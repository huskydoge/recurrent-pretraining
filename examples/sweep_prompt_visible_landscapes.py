from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from export_position_pca_review_pairgrid import (
    DEFAULT_SYSTEM_PROMPT,
    build_prefix_ids,
    compute_pair_end_variances,
    compute_pair_score,
    compute_pca_coords,
    select_farthest_endpoint_representatives,
)
from export_position_pca_review_figures import apply_publication_style
from vis import clean_output_text, collect_generated_token_traces, sample_next_token_predictions, collect_next_token_trajectories


DEFAULT_PROMPTS = [
    "Continue this sentence in one clear sentence: The best way to learn programming is",
    "Finish this thought in one sentence: A good leader should",
    "Explain why the sky appears blue in one sentence.",
    "Explain photosynthesis in one sentence.",
    "State one benefit of regular exercise in one sentence.",
    "Describe gravity to a child in one sentence.",
    "Why do leaves change color in autumn? Answer in one sentence.",
    "What happens when water freezes? Answer in one sentence.",
    "A train travels 60 miles in 1.5 hours. What is its average speed? Explain briefly.",
    "A grove has 15 trees. Workers plant more trees and now there are 21. How many trees did they plant?",
    "Is 37 a prime number? Explain briefly.",
    "What is the capital of France? Answer in one sentence.",
]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep multiple prompts to find a visibly separated PCA landscape.")
    parser.add_argument("--model-dir", type=Path, default=Path("/data/hansen/serv12/HRM-Deq/models/huginn-0125"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/rebuttal_ar_figures/prompt_sweep_visible_landscape.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--search-max-positions", type=int, default=8)
    parser.add_argument("--search-num-samples", type=int, default=24)
    parser.add_argument("--search-num-steps", type=int, default=32)
    parser.add_argument("--full-num-samples", type=int, default=256)
    parser.add_argument("--full-num-steps", type=int, default=32)
    parser.add_argument("--sample-batch-size", type=int, default=16)
    parser.add_argument("--init-std-scales", default="12,16")
    parser.add_argument("--pca-dims", default="6,8,10,12")
    parser.add_argument("--min-minority-fraction", type=float, default=0.10)
    parser.add_argument("--ranking-objective", choices=("end_var", "pair_score"), default="pair_score")
    return parser.parse_args()


def decode_counts(tokenizer: AutoTokenizer, counts: list[tuple[int, int]]) -> list[tuple[str, int]]:
    return [(tokenizer.decode([token_id], skip_special_tokens=False), int(count)) for token_id, count in counts]


def main() -> None:
    args = parse_args()
    apply_publication_style()
    torch.manual_seed(args.seed)

    init_scales = parse_float_list(args.init_std_scales)
    pca_dims = parse_int_list(args.pca_dims)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[prompt-sweep] loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    )
    model.eval()
    print("[prompt-sweep] model loaded")

    generation_config = GenerationConfig(
        max_new_tokens=96,
        stop_strings=["<|end_text|>", "<|end_turn|>"],
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        return_dict_in_generate=True,
        eos_token_id=65505,
        bos_token_id=65504,
        pad_token_id=65509,
    )

    all_results: list[dict] = []

    for prompt_idx, question in enumerate(DEFAULT_PROMPTS):
        print(f"[prompt-sweep] prompt={prompt_idx}: {question}")
        messages = []
        if args.system_prompt.strip():
            messages.append({"role": "system", "content": args.system_prompt.strip()})
        messages.append({"role": "user", "content": question.strip()})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
        generated_trace = collect_generated_token_traces(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids.clone(),
            generation_config=generation_config,
            num_steps=12,
            max_new_tokens=96,
            init_scale=0.0,
        )
        generated_token_ids = generated_trace["generated_token_ids"].tolist()
        generated_text = clean_output_text(tokenizer.decode(generated_token_ids, skip_special_tokens=False))
        max_positions = min(args.search_max_positions, len(generated_token_ids))

        for init_scale in init_scales:
            print(f"[prompt-sweep] prompt={prompt_idx} init_scale={init_scale}")
            for position in range(max_positions):
                prefix_ids = build_prefix_ids(input_ids, generated_token_ids, position)
                pred_search = sample_next_token_predictions(
                    model=model,
                    input_ids=prefix_ids,
                    num_samples=args.search_num_samples,
                    num_steps=args.search_num_steps,
                    init_scale=init_scale,
                    batch_size=args.sample_batch_size,
                )
                search_counts = Counter(pred_search.tolist()).most_common(4)
                if len(search_counts) < 2:
                    continue
                search_minority_fraction = search_counts[1][1] / args.search_num_samples
                if search_minority_fraction < args.min_minority_fraction:
                    continue

                trace = collect_next_token_trajectories(
                    model=model,
                    input_ids=prefix_ids,
                    num_samples=args.full_num_samples,
                    num_steps=args.full_num_steps,
                    init_scale=init_scale,
                    batch_size=args.sample_batch_size,
                )
                pred_full = trace["predicted_token_ids"].cpu()
                full_counts = Counter(pred_full.tolist()).most_common(4)
                if len(full_counts) < 2:
                    continue
                full_minority_fraction = full_counts[1][1] / args.full_num_samples
                if full_minority_fraction < args.min_minority_fraction:
                    continue

                candidate = {
                    "prompt_index": prompt_idx,
                    "question": question,
                    "generated_text": generated_text,
                    "position": position,
                    "reference_token": tokenizer.decode([generated_token_ids[position]], skip_special_tokens=False),
                    "init_scale": init_scale,
                    "search_top_tokens": decode_counts(tokenizer, search_counts),
                    "full_top_tokens": decode_counts(tokenizer, full_counts),
                    "full_minority_fraction": full_minority_fraction,
                    "dims": [],
                }

                for dim in pca_dims:
                    projected = compute_pca_coords(trace["states"], n_components=dim, whiten=False)
                    coords = projected["coords"].numpy()
                    final_coords = coords[:, -1, :]
                    overlay = select_farthest_endpoint_representatives(final_coords, pred_full, max_tokens=2)
                    pair_vars = compute_pair_end_variances({"coords": coords, "predicted_token_ids": pred_full}, overlay)
                    pair_scores = []
                    for row in range(dim):
                        for col in range(dim):
                            if row == col:
                                continue
                            score = compute_pair_score(
                                final_coords,
                                pred_full,
                                (row, col),
                                min_group_size=2,
                            )
                            if score is not None:
                                pair_scores.append(score)
                    pair_scores.sort(key=lambda item: item["score"], reverse=True)
                    endpoints = final_coords[overlay]
                    endpoint_distance = float(np.linalg.norm(endpoints[0] - endpoints[1])) if len(endpoints) >= 2 else 0.0
                    top_pair = pair_vars[0] if pair_vars else None
                    top_pair_score = pair_scores[0] if pair_scores else None
                    if args.ranking_objective == "pair_score":
                        ranking_score = float((top_pair_score["score"] if top_pair_score else 0.0) * full_minority_fraction)
                    else:
                        ranking_score = float((top_pair["end_var"] if top_pair else 0.0) * full_minority_fraction)
                    candidate["dims"].append(
                        {
                            "pca_dim": dim,
                            "overlay_indices": overlay,
                            "rep_endpoint_distance": endpoint_distance,
                            "top_pair": top_pair,
                            "top_pair_score": top_pair_score,
                            "score": ranking_score,
                        }
                    )

                candidate["best_dim"] = max(candidate["dims"], key=lambda item: item["score"])
                all_results.append(candidate)
                print(
                    "[prompt-sweep] candidate",
                    json.dumps(
                        {
                            "prompt_index": prompt_idx,
                            "position": position,
                            "init_scale": init_scale,
                            "full_top_tokens": candidate["full_top_tokens"][:2],
                            "best_dim": candidate["best_dim"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    all_results.sort(
        key=lambda item: (
            item["best_dim"]["score"],
            item["best_dim"]["top_pair_score"]["score"] if item["best_dim"]["top_pair_score"] else 0.0,
            item["best_dim"]["top_pair"]["end_var"] if item["best_dim"]["top_pair"] else 0.0,
            item["best_dim"]["rep_endpoint_distance"],
        ),
        reverse=True,
    )

    payload = {
        "prompts": DEFAULT_PROMPTS,
        "search_num_samples": args.search_num_samples,
        "search_num_steps": args.search_num_steps,
        "full_num_samples": args.full_num_samples,
        "full_num_steps": args.full_num_steps,
        "init_scales": init_scales,
        "pca_dims": pca_dims,
        "ranking_objective": args.ranking_objective,
        "results": all_results,
    }
    args.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[prompt-sweep] wrote {args.output_path}")
    if all_results:
        print("[prompt-sweep] best candidate:", json.dumps(all_results[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
