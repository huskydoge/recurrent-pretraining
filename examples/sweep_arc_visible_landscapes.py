from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
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


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def decode_counts(tokenizer: AutoTokenizer, counts: list[tuple[int, int]]) -> list[tuple[str, int]]:
    return [(tokenizer.decode([token_id], skip_special_tokens=False), int(count)) for token_id, count in counts]


def format_arc_prompt(example: dict, answer_format: str) -> str:
    choices = example["choices"]
    choice_lines = "\n".join(
        f"{label}. {text}" for label, text in zip(choices["label"], choices["text"], strict=True)
    )
    if answer_format == "choice_label":
        instruction = "Answer with only the correct choice label."
    elif answer_format == "explanatory":
        instruction = "Answer the multiple-choice science question."
    else:
        instruction = "Respond with exactly one choice text from the list above and nothing else. Do not explain. Do not add punctuation."
    return (
        f"Question: {example['question']}\n"
        f"Choices:\n{choice_lines}\n"
        f"{instruction}\n"
        "Answer:"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep ARC-Challenge examples to find visibly separated PCA landscapes.")
    parser.add_argument("--model-dir", type=Path, default=Path("/data/hansen/serv12/HRM-Deq/models/huginn-0125"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/rebuttal_ar_figures/arc_challenge_visible_sweep.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--answer-format", choices=("concise_text", "choice_label", "explanatory"), default="concise_text")
    parser.add_argument("--question-start", type=int, default=0)
    parser.add_argument("--question-count", type=int, default=8)
    parser.add_argument("--search-max-positions", type=int, default=20)
    parser.add_argument("--search-num-samples", type=int, default=16)
    parser.add_argument("--search-num-steps", type=int, default=32)
    parser.add_argument("--full-num-samples", type=int, default=128)
    parser.add_argument("--full-num-steps", type=int, default=32)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--init-std-scales", default="8,10,12")
    parser.add_argument("--pca-dims", default="6,8,10,12")
    parser.add_argument("--max-candidates-per-scale", type=int, default=6)
    parser.add_argument("--min-minority-fraction", type=float, default=0.02)
    parser.add_argument("--ranking-objective", choices=("end_var", "pair_score"), default="pair_score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_publication_style()
    torch.manual_seed(args.seed)

    init_scales = parse_float_list(args.init_std_scales)
    pca_dims = parse_int_list(args.pca_dims)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_payload(results: list[dict], completed_questions: list[int]) -> None:
        payload = {
            "dataset": "allenai/ai2_arc",
            "config": "ARC-Challenge",
            "dataset_split": args.dataset_split,
            "answer_format": args.answer_format,
            "question_start": args.question_start,
            "question_count": args.question_count,
            "search_num_samples": args.search_num_samples,
            "search_num_steps": args.search_num_steps,
            "full_num_samples": args.full_num_samples,
            "full_num_steps": args.full_num_steps,
            "init_scales": init_scales,
            "pca_dims": pca_dims,
            "ranking_objective": args.ranking_objective,
            "completed_questions": completed_questions,
            "results": results,
        }
        args.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print("[arc-sweep] loading dataset...")
    ds = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split=f"{args.dataset_split}[{args.question_start}:{args.question_start + args.question_count}]",
    )
    print(f"[arc-sweep] loaded {len(ds)} questions")

    print("[arc-sweep] loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    )
    model.eval()
    print("[arc-sweep] model loaded")

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
    completed_questions: list[int] = []

    for local_idx, example in enumerate(ds):
        question_index = args.question_start + local_idx
        prompt_body = format_arc_prompt(example, args.answer_format)
        print(f"[arc-sweep] question={question_index}")
        messages = []
        if args.system_prompt.strip():
            messages.append({"role": "system", "content": args.system_prompt.strip()})
        messages.append({"role": "user", "content": prompt_body})
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

        question_results: list[dict] = []

        for init_scale in init_scales:
            search_candidates: list[dict] = []
            for position in range(max_positions):
                prefix_ids = build_prefix_ids(input_ids, generated_token_ids, position)
                pred = sample_next_token_predictions(
                    model=model,
                    input_ids=prefix_ids,
                    num_samples=args.search_num_samples,
                    num_steps=args.search_num_steps,
                    init_scale=init_scale,
                    batch_size=args.sample_batch_size,
                )
                counts = Counter(pred.tolist()).most_common(5)
                if len(counts) < 2:
                    continue
                mode_fraction = counts[0][1] / args.search_num_samples
                minority_fraction = counts[1][1] / args.search_num_samples
                if minority_fraction < args.min_minority_fraction:
                    continue
                search_candidates.append(
                    {
                        "position": position,
                        "reference_token": tokenizer.decode([generated_token_ids[position]], skip_special_tokens=False),
                        "search_top_tokens": decode_counts(tokenizer, counts),
                        "search_mode_fraction": mode_fraction,
                        "search_minority_fraction": minority_fraction,
                    }
                )

            search_candidates.sort(
                key=lambda item: (
                    -item["search_minority_fraction"],
                    item["search_mode_fraction"],
                    item["position"],
                )
            )
            search_candidates = search_candidates[: args.max_candidates_per_scale]
            print(f"[arc-sweep] question={question_index} init_scale={init_scale} candidates={len(search_candidates)}")

            for candidate in search_candidates:
                position = candidate["position"]
                prefix_ids = build_prefix_ids(input_ids, generated_token_ids, position)
                trace = collect_next_token_trajectories(
                    model=model,
                    input_ids=prefix_ids,
                    num_samples=args.full_num_samples,
                    num_steps=args.full_num_steps,
                    init_scale=init_scale,
                    batch_size=args.sample_batch_size,
                )
                pred = trace["predicted_token_ids"].cpu()
                counts = Counter(pred.tolist()).most_common(5)
                if len(counts) < 2:
                    continue
                minority_fraction = counts[1][1] / args.full_num_samples
                if minority_fraction < args.min_minority_fraction:
                    continue

                candidate_result = {
                    "question_index": question_index,
                    "id": example["id"],
                    "question": example["question"],
                    "choices": example["choices"],
                    "answerKey": example["answerKey"],
                    "generated_text": generated_text,
                    "position": position,
                    "reference_token": candidate["reference_token"],
                    "init_scale": init_scale,
                    "full_top_tokens": decode_counts(tokenizer, counts),
                    "full_mode_fraction": counts[0][1] / args.full_num_samples,
                    "full_minority_fraction": minority_fraction,
                    "dims": [],
                }

                for dim in pca_dims:
                    projected = compute_pca_coords(trace["states"], n_components=dim, whiten=False)
                    coords = projected["coords"].numpy()
                    final_coords = coords[:, -1, :]
                    overlay = select_farthest_endpoint_representatives(final_coords, pred, max_tokens=2)
                    pair_vars = compute_pair_end_variances({"coords": coords, "predicted_token_ids": pred}, overlay)

                    pair_scores = []
                    for row in range(dim):
                        for col in range(dim):
                            if row == col:
                                continue
                            score = compute_pair_score(final_coords, pred, (row, col), min_group_size=2)
                            if score is not None:
                                pair_scores.append(score)
                    pair_scores.sort(key=lambda item: item["score"], reverse=True)

                    rep_endpoints = final_coords[overlay]
                    endpoint_distance = float(np.linalg.norm(rep_endpoints[0] - rep_endpoints[1])) if len(overlay) >= 2 else 0.0
                    top_pair = pair_vars[0] if pair_vars else None
                    top_pair_score = pair_scores[0] if pair_scores else None
                    if args.ranking_objective == "pair_score":
                        ranking_score = float((top_pair_score["score"] if top_pair_score else 0.0) * minority_fraction)
                    else:
                        ranking_score = float((top_pair["end_var"] if top_pair else 0.0) * minority_fraction)

                    candidate_result["dims"].append(
                        {
                            "pca_dim": dim,
                            "overlay_indices": overlay,
                            "rep_endpoint_distance": endpoint_distance,
                            "top_pair": top_pair,
                            "top_pair_score": top_pair_score,
                            "score": ranking_score,
                        }
                    )

                candidate_result["best_dim"] = max(candidate_result["dims"], key=lambda item: item["score"])
                question_results.append(candidate_result)
                print(
                    "[arc-sweep] result",
                    json.dumps(
                        {
                            "question_index": question_index,
                            "position": position,
                            "init_scale": init_scale,
                            "full_top_tokens": candidate_result["full_top_tokens"][:2],
                            "best_dim": candidate_result["best_dim"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        question_results.sort(
            key=lambda item: (
                item["best_dim"]["score"],
                item["best_dim"]["top_pair_score"]["score"] if item["best_dim"]["top_pair_score"] else 0.0,
                item["best_dim"]["top_pair"]["end_var"] if item["best_dim"]["top_pair"] else 0.0,
                item["best_dim"]["rep_endpoint_distance"],
            ),
            reverse=True,
        )
        all_results.extend(question_results[:1])
        all_results.sort(
            key=lambda item: (
                item["best_dim"]["score"],
                item["best_dim"]["top_pair_score"]["score"] if item["best_dim"]["top_pair_score"] else 0.0,
                item["best_dim"]["top_pair"]["end_var"] if item["best_dim"]["top_pair"] else 0.0,
                item["best_dim"]["rep_endpoint_distance"],
            ),
            reverse=True,
        )
        completed_questions.append(question_index)
        write_payload(all_results, completed_questions)
        print(f"[arc-sweep] checkpoint wrote {args.output_path}")

    write_payload(all_results, completed_questions)
    print(f"[arc-sweep] wrote {args.output_path}")
    if all_results:
        print("[arc-sweep] best candidate:", json.dumps(all_results[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
