from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from export_position_pca_review_pairgrid import (
    DEFAULT_QUESTION,
    DEFAULT_SYSTEM_PROMPT,
    analyze_position,
    apply_publication_style,
    compute_pair_end_variances,
    export_single_pairgrid_landscape,
    export_unstable_gallery,
    format_metric,
    select_pairgrid_overlay,
)
from vis import clean_output_text, collect_generated_token_traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export only the unstable-position pairgrid/gallery for a fixed output token.")
    parser.add_argument("--model-dir", type=Path, default=Path("/data/hansen/serv12/HRM-Deq/models/huginn-0125"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rebuttal_ar_figures"))
    parser.add_argument("--run-tag", default="unstable_only")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--pairgrid-trajectories", type=int, default=512)
    parser.add_argument("--plot-grid-size", type=int, default=120)
    parser.add_argument("--landscape-fit-max-points", type=int, default=2048)
    parser.add_argument("--full-output-num-steps", type=int, default=12)
    parser.add_argument("--full-output-max-new-tokens", type=int, default=128)
    parser.add_argument("--init-std-scale", type=float, default=12.0)
    parser.add_argument("--pairgrid-components", type=int, default=4)
    parser.add_argument("--pairgrid-min-group-size", type=int, default=2)
    parser.add_argument(
        "--pairgrid-overlay-strategy",
        choices=("round_robin", "per_token_centroid", "per_token_farthest_end"),
        default="round_robin",
    )
    parser.add_argument("--pairgrid-only", action="store_true")
    parser.add_argument("--gallery-only", action="store_true")
    parser.add_argument("--gallery-max-trajectories", type=int, default=12)
    parser.add_argument("--gallery-focus-tail-steps", type=int, default=None)
    parser.add_argument("--gallery-landscape-tail-steps", type=int, default=None)
    parser.add_argument("--no-pca-whiten", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_publication_style()
    torch.manual_seed(args.seed)
    print(f"[export] run_tag={args.run_tag} position={args.position} samples={args.num_samples} steps={args.num_steps}")

    run_dir = args.output_dir / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    for path in run_dir.iterdir():
        if path.is_file():
            path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    print("[export] loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    )
    model.eval()
    print("[export] model loaded")

    messages = []
    if args.system_prompt.strip():
        messages.append({"role": "system", "content": args.system_prompt.strip()})
    messages.append({"role": "user", "content": args.question.strip()})
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)

    generation_config = GenerationConfig(
        max_new_tokens=args.full_output_max_new_tokens,
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
    generated_trace = collect_generated_token_traces(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids.clone(),
        generation_config=generation_config,
        num_steps=args.full_output_num_steps,
        max_new_tokens=args.full_output_max_new_tokens,
        init_scale=0.0,
    )
    print("[export] collected full output trace")
    generated_token_ids = generated_trace["generated_token_ids"].tolist()
    if len(generated_token_ids) <= args.position:
        raise RuntimeError(f"Need output token position {args.position}, but only got {len(generated_token_ids)} generated tokens.")

    base_init_std = float(model.config.init_values["std"])
    analysis_init_scale = (base_init_std * args.init_std_scale) / base_init_std if base_init_std != 0 else 0.0
    info = {
        "position": args.position,
        "reference_token_id": int(generated_token_ids[args.position]),
        "mode_fraction": -1.0,
    }
    analysis = analyze_position(
        model=model,
        tokenizer=tokenizer,
        prompt_input_ids=input_ids,
        generated_token_ids=generated_token_ids,
        info=info,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        init_scale=analysis_init_scale,
        batch_size=args.sample_batch_size,
        n_components=args.pairgrid_components,
        min_group_size=args.pairgrid_min_group_size,
        pca_whiten=not args.no_pca_whiten,
    )
    print("[export] analyzed fixed position")
    counts = Counter(analysis["predicted_token_ids"].tolist())
    mode_count = counts.most_common(1)[0][1]
    analysis["info"]["mode_fraction"] = mode_count / analysis["predicted_token_ids"].numel()
    overlay_indices, _ = select_pairgrid_overlay(
        analysis,
        pairgrid_trajectories=args.pairgrid_trajectories,
        overlay_strategy=args.pairgrid_overlay_strategy,
    )
    pair_end_variances = compute_pair_end_variances(analysis, overlay_indices)

    if not args.gallery_only:
        export_single_pairgrid_landscape(
            out_path=run_dir / "unstable_pairgrid.pdf",
            analysis=analysis,
            tokenizer=tokenizer,
            grid_size=args.plot_grid_size,
            grid_pad=0.24,
            smooth=0.16,
            level_count=12,
            max_fit_points=args.landscape_fit_max_points,
            pairgrid_trajectories=args.pairgrid_trajectories,
            gallery_max_tokens=3,
            overlay_strategy=args.pairgrid_overlay_strategy,
            title_prefix="Unstable output token",
            token_color_legend_title="Unstable Token Colors",
        )
        print("[export] wrote pairgrid")
    if not args.pairgrid_only:
        export_unstable_gallery(
            out_path=run_dir / "unstable_gallery.pdf",
            unstable_analysis=analysis,
            tokenizer=tokenizer,
            grid_size=args.plot_grid_size,
            grid_pad=0.24,
            smooth=0.16,
            level_count=12,
            max_fit_points=args.landscape_fit_max_points,
            gallery_max_tokens=3,
            gallery_max_trajectories=args.gallery_max_trajectories,
            overlay_indices=overlay_indices if args.pairgrid_overlay_strategy != "round_robin" else None,
            focus_tail_steps=args.gallery_focus_tail_steps,
            landscape_tail_steps=args.gallery_landscape_tail_steps,
        )
        print("[export] wrote gallery")

    summary = {
        "position": args.position,
        "reference_token_id": int(generated_token_ids[args.position]),
        "mode_fraction": analysis["info"]["mode_fraction"],
        "top_tokens": analysis["token_count_summary"],
        "best_pair": list(analysis["best_pair"]),
        "best_pair_score": analysis["pair_scores"][0]["score"] if analysis["pair_scores"] else None,
        "pca_whiten": analysis["pca_whiten"],
        "generated_text": clean_output_text(tokenizer.decode(generated_token_ids, skip_special_tokens=False)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    ranking_lines = [
        f"Run tag: {args.run_tag}",
        f"Position: {args.position}",
        f"Reference token: {repr(tokenizer.decode([generated_token_ids[args.position]], skip_special_tokens=False))}",
        f"Samples: {args.num_samples}",
        f"Steps: {args.num_steps}",
        f"PCA components: {args.pairgrid_components}",
        f"PCA whiten: {not args.no_pca_whiten}",
        f"Overlay strategy: {args.pairgrid_overlay_strategy}",
        f"Overlay count: {len(overlay_indices)}",
        "",
        "Representative-end variance ranking (descending):",
    ]
    for rank, item in enumerate(pair_end_variances, start=1):
        ranking_lines.append(f"{rank:02d}. {item['label']}: End var={format_metric(item['end_var'])}")
    (run_dir / "pair_end_variance_ranking.txt").write_text("\n".join(ranking_lines).rstrip() + "\n")
    input_text = []
    if args.system_prompt.strip():
        input_text.append("[System]")
        input_text.append(args.system_prompt.strip())
        input_text.append("")
    input_text.append("[User]")
    input_text.append(args.question.strip())
    (run_dir / "input.txt").write_text("\n".join(input_text).strip() + "\n")
    (run_dir / "output.txt").write_text(summary["generated_text"].strip() + "\n")
    print("[export] wrote text artifacts")

    zip_path = args.output_dir / f"{args.run_tag}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.iterdir()):
            zf.write(path, arcname=path.name)
    print("[export] wrote zip")

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()
