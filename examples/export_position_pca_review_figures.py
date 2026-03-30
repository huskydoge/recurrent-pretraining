from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from vis import (
    DEFAULT_PALETTE,
    clean_output_text,
    collect_generated_token_traces,
    collect_next_token_trajectories,
    compact_labels,
    decode_token_ids,
    draw_landscape_with_trajectories,
    find_variable_generated_positions,
    plot_landscape_with_trajectories,
    plot_residual_heatmap,
    project_states_with_pca,
    sample_next_token_predictions,
)


PROFILE_CONFIGS = {
    "smoke": {
        "NUM_SAMPLES": 16,
        "SAMPLE_BATCH_SIZE": 4,
        "NUM_STEPS": 8,
        "MAX_TRAJECTORIES_TO_OVERLAY": 6,
        "PLOT_GRID_SIZE": 48,
        "GRID_PAD": 0.10,
        "RBF_SMOOTH": 0.22,
        "WHITEN_COORDS": True,
        "HIDE_AXES": False,
        "SHOW_COLORBAR": True,
        "CONTOUR_LEVEL_COUNT": 8,
        "INIT_STD_SCALE": 4.0,
        "INIT_STD_OVERRIDE": None,
        "FULL_OUTPUT_NUM_STEPS": 8,
        "FULL_OUTPUT_MAX_NEW_TOKENS": 24,
        "SINGLE_TRAJECTORY_COUNT": 3,
        "SEARCH_NUM_SAMPLES": 16,
        "SEARCH_NUM_STEPS": 8,
    },
    "fast": {
        "NUM_SAMPLES": 48,
        "SAMPLE_BATCH_SIZE": 8,
        "NUM_STEPS": 16,
        "MAX_TRAJECTORIES_TO_OVERLAY": 10,
        "PLOT_GRID_SIZE": 72,
        "GRID_PAD": 0.10,
        "RBF_SMOOTH": 0.20,
        "WHITEN_COORDS": True,
        "HIDE_AXES": False,
        "SHOW_COLORBAR": True,
        "CONTOUR_LEVEL_COUNT": 10,
        "INIT_STD_SCALE": 12.0,
        "INIT_STD_OVERRIDE": None,
        "FULL_OUTPUT_NUM_STEPS": 12,
        "FULL_OUTPUT_MAX_NEW_TOKENS": 64,
        "SINGLE_TRAJECTORY_COUNT": 4,
        "SEARCH_NUM_SAMPLES": 24,
        "SEARCH_NUM_STEPS": 12,
    },
    "balanced": {
        "NUM_SAMPLES": 96,
        "SAMPLE_BATCH_SIZE": 16,
        "NUM_STEPS": 24,
        "MAX_TRAJECTORIES_TO_OVERLAY": 16,
        "PLOT_GRID_SIZE": 96,
        "GRID_PAD": 0.08,
        "RBF_SMOOTH": 0.18,
        "WHITEN_COORDS": True,
        "HIDE_AXES": False,
        "SHOW_COLORBAR": True,
        "CONTOUR_LEVEL_COUNT": 12,
        "INIT_STD_SCALE": 14.0,
        "INIT_STD_OVERRIDE": None,
        "FULL_OUTPUT_NUM_STEPS": 16,
        "FULL_OUTPUT_MAX_NEW_TOKENS": 128,
        "SINGLE_TRAJECTORY_COUNT": 5,
        "SEARCH_NUM_SAMPLES": 32,
        "SEARCH_NUM_STEPS": 12,
    },
    "full": {
        "NUM_SAMPLES": 256,
        "SAMPLE_BATCH_SIZE": 32,
        "NUM_STEPS": 64,
        "MAX_TRAJECTORIES_TO_OVERLAY": 24,
        "PLOT_GRID_SIZE": 140,
        "GRID_PAD": 0.08,
        "RBF_SMOOTH": 0.15,
        "WHITEN_COORDS": True,
        "HIDE_AXES": False,
        "SHOW_COLORBAR": True,
        "CONTOUR_LEVEL_COUNT": 16,
        "INIT_STD_SCALE": 16.0,
        "INIT_STD_OVERRIDE": None,
        "FULL_OUTPUT_NUM_STEPS": 32,
        "FULL_OUTPUT_MAX_NEW_TOKENS": 256,
        "SINGLE_TRAJECTORY_COUNT": 6,
        "SEARCH_NUM_SAMPLES": 48,
        "SEARCH_NUM_STEPS": 16,
    },
}


DEFAULT_SYSTEM_PROMPT = """You are Huginn, an AI assistant who embodies careful thought and deliberation.

Your responses demonstrate:
- Methodical reasoning, breaking complex problems into clear steps
- Mathematical and programming expertise grounded in fundamentals
- The ability to acknowledge uncertainty and correct course when needed
- Clear communication that illuminates rather than just informs
"""


DEFAULT_QUESTION = (
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
    "After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rebuttal-ready AR trajectory figures to PDF and ZIP.")
    parser.add_argument("--model-dir", type=Path, default=Path("/data/hansen/serv12/HRM-Deq/models/huginn-0125"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rebuttal_ar_figures"))
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="balanced")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--preview-init-scale", type=float, default=0.0)
    parser.add_argument("--search-max-positions", type=int, default=24)
    parser.add_argument("--gallery-max-samples", type=int, default=5)
    return parser.parse_args()


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "font.weight": "bold",
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.8,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def style_figure(fig: plt.Figure) -> None:
    for ax in fig.axes:
        style_axis(ax)


def style_last_colorbar(fig: plt.Figure, label: str | None = None) -> None:
    if not fig.axes:
        return
    cax = fig.axes[-1]
    if label:
        cax.set_ylabel(label, fontsize=15, fontweight="bold")
    for tick in cax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(12)


def choose_gallery_indices(labels: list[str], count: int) -> list[int]:
    chosen = []
    seen = set()
    for idx, label in enumerate(labels):
        if label not in seen:
            chosen.append(idx)
            seen.add(label)
        if len(chosen) == count:
            return chosen
    extras = np.linspace(0, len(labels) - 1, num=min(len(labels), count), dtype=int).tolist()
    for idx in extras:
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) == count:
            break
    return chosen[:count]


def collect_position_scan(
    *,
    model: Any,
    prompt_input_ids: torch.Tensor,
    reference_generated_token_ids: torch.Tensor,
    num_samples: int,
    num_steps: int,
    init_scale: float,
    batch_size: int,
    max_positions: int,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for position in range(min(max_positions, len(reference_generated_token_ids))):
        if position == 0:
            prefix_ids = prompt_input_ids
        else:
            prefix_ids = torch.cat([prompt_input_ids, reference_generated_token_ids[:position].unsqueeze(0)], dim=-1)
        predicted = sample_next_token_predictions(
            model=model,
            input_ids=prefix_ids,
            num_samples=num_samples,
            num_steps=num_steps,
            init_scale=init_scale,
            batch_size=batch_size,
        )
        unique_ids, counts = torch.unique(predicted, sorted=False, return_counts=True)
        order = torch.argsort(counts, descending=True)
        unique_ids = unique_ids[order]
        counts = counts[order]
        findings.append(
            {
                "position": position,
                "predicted_token_ids": predicted,
                "unique_token_ids": unique_ids,
                "counts": counts,
                "mode_fraction": float(counts[0].item() / predicted.numel()),
                "num_unique": int(unique_ids.numel()),
                "reference_token_id": int(reference_generated_token_ids[position].item()),
            }
        )
    return findings


def normalize_variable_finding(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    normalized["num_unique"] = int(entry["unique_token_ids"].numel())
    return normalized


def pick_stable_and_unstable_positions(scan: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    unstable_candidates = [entry for entry in scan if entry["num_unique"] >= 2]
    if unstable_candidates:
        unstable = min(
            unstable_candidates,
            key=lambda entry: (entry["mode_fraction"], entry["position"]),
        )
    else:
        unstable = min(
            scan,
            key=lambda entry: (entry["mode_fraction"], -entry["num_unique"], entry["position"]),
        )
    stable_candidates = [entry for entry in scan if entry["num_unique"] == 1]
    if stable_candidates:
        stable = stable_candidates[0]
    else:
        stable = max(scan, key=lambda entry: (entry["mode_fraction"], -entry["position"]))
    return stable, unstable


def build_prefix_ids(prompt_input_ids: torch.Tensor, generated_token_ids: list[int], position: int) -> torch.Tensor:
    if position == 0:
        return prompt_input_ids
    prefix_suffix = torch.tensor(generated_token_ids[:position], device=prompt_input_ids.device).unsqueeze(0)
    return torch.cat([prompt_input_ids, prefix_suffix], dim=-1)


def export_heatmap(
    out_path: Path,
    generated_step_residual: np.ndarray,
    generated_token_labels: list[str],
) -> None:
    fig, ax = plot_residual_heatmap(
        generated_step_residual,
        compact_labels(generated_token_labels, max_chars=12),
        title="Residual Heatmap Across Generated Output Tokens",
        cmap="magma",
        max_label_chars=12,
    )
    ax.set_xlabel("Generated Token Position", fontsize=16, fontweight="bold")
    ax.set_ylabel("Recurrent Iteration", fontsize=16, fontweight="bold")
    ax.set_title("Residual Heatmap Across Generated Output Tokens", fontsize=18, fontweight="bold")
    style_figure(fig)
    style_last_colorbar(fig, "||f_t - f_{t-1}||_2")
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def export_stable_vs_unstable(
    *,
    out_path: Path,
    model: Any,
    tokenizer: Any,
    prompt_input_ids: torch.Tensor,
    generated_token_ids: list[int],
    stable_info: dict[str, Any],
    unstable_info: dict[str, Any],
    num_samples: int,
    num_steps: int,
    init_scale: float,
    batch_size: int,
    grid_size: int,
    grid_pad: float,
    smooth: float,
    level_count: int,
    max_trajectories: int,
    whiten_coords: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), dpi=200)
    palette = DEFAULT_PALETTE.copy()

    for ax, info, panel_title in zip(
        axes,
        [stable_info, unstable_info],
        ["Stable Output Token", "Unstable Output Token"],
        strict=True,
    ):
        prefix_ids = build_prefix_ids(prompt_input_ids, generated_token_ids, info["position"])
        trace = collect_next_token_trajectories(
            model=model,
            input_ids=prefix_ids,
            num_samples=num_samples,
            num_steps=num_steps,
            init_scale=init_scale,
            batch_size=batch_size,
        )
        predicted_labels = decode_token_ids(tokenizer, trace["predicted_token_ids"].tolist(), max_chars=24)
        pca = project_states_with_pca(trace["states"].unsqueeze(2), whiten_coords=whiten_coords)[0]
        contour = draw_landscape_with_trajectories(
            ax,
            coords=pca["coords"].numpy(),
            residuals=trace["residual_norms"].numpy(),
            title=None,
            grid_size=grid_size,
            grid_pad=grid_pad,
            smooth=smooth,
            level_count=level_count,
            cmap="viridis",
            hide_axes=False,
            max_trajectories_to_overlay=max_trajectories,
            palette=palette,
        )
        ref_label = decode_token_ids(tokenizer, [info["reference_token_id"]], max_chars=24)[0]
        top_counter = Counter(predicted_labels).most_common(3)
        summary = ", ".join(f"{label}:{count}" for label, count in top_counter)
        ax.set_title(
            f"{panel_title}\nposition={info['position']}, ref={ref_label!r}, mode={info['mode_fraction']:.2f}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("PC 1", fontsize=15, fontweight="bold")
        ax.set_ylabel("PC 2", fontsize=15, fontweight="bold")
        ax.text(
            0.5,
            -0.18,
            f"Decoded tokens: {summary}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        style_axis(ax)

    cbar = fig.colorbar(contour, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("||f_t - f_{t-1}||_2", fontsize=15, fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(12)
    fig.suptitle("Stable vs Unstable Token-Conditional Local Landscapes", fontsize=19, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.24, top=0.82, wspace=0.28)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def export_unstable_gallery(
    *,
    out_path: Path,
    model: Any,
    tokenizer: Any,
    prompt_input_ids: torch.Tensor,
    generated_token_ids: list[int],
    unstable_info: dict[str, Any],
    num_samples: int,
    num_steps: int,
    init_scale: float,
    batch_size: int,
    grid_size: int,
    grid_pad: float,
    smooth: float,
    level_count: int,
    whiten_coords: bool,
    gallery_count: int,
) -> None:
    prefix_ids = build_prefix_ids(prompt_input_ids, generated_token_ids, unstable_info["position"])
    trace = collect_next_token_trajectories(
        model=model,
        input_ids=prefix_ids,
        num_samples=num_samples,
        num_steps=num_steps,
        init_scale=init_scale,
        batch_size=batch_size,
    )
    predicted_labels = decode_token_ids(tokenizer, trace["predicted_token_ids"].tolist(), max_chars=24)
    gallery_indices = choose_gallery_indices(predicted_labels, gallery_count)
    pca = project_states_with_pca(trace["states"].unsqueeze(2), whiten_coords=whiten_coords)[0]
    coords = pca["coords"].numpy()
    residuals = trace["residual_norms"].numpy()

    ncols = min(3, len(gallery_indices))
    nrows = math.ceil(len(gallery_indices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6.8 * nrows), dpi=200)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    palette = DEFAULT_PALETTE.copy()

    contour = None
    for ax, sample_idx in zip(axes.flat, gallery_indices, strict=False):
        contour = draw_landscape_with_trajectories(
            ax,
            coords=coords,
            residuals=residuals,
            title=None,
            grid_size=grid_size,
            grid_pad=grid_pad,
            smooth=smooth,
            level_count=level_count,
            cmap="viridis",
            hide_axes=False,
            trajectory_sample_indices=[sample_idx],
            palette=palette,
        )
        ax.set_xlabel("PC 1", fontsize=15, fontweight="bold")
        ax.set_ylabel("PC 2", fontsize=15, fontweight="bold")
        ax.set_title(
            f"Sample {sample_idx}",
            fontsize=15,
            fontweight="bold",
        )
        ax.text(
            0.5,
            -0.16,
            f"Decoded token: {predicted_labels[sample_idx]!r}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        style_axis(ax)

    for ax in axes.flat[len(gallery_indices):]:
        ax.set_axis_off()

    if contour is not None:
        cbar = fig.colorbar(contour, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("||f_t - f_{t-1}||_2", fontsize=15, fontweight="bold")
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")
            tick.set_fontsize(12)

    ref_label = decode_token_ids(tokenizer, [unstable_info["reference_token_id"]], max_chars=24)[0]
    fig.suptitle(
        (
            "Unstable Token Gallery\n"
            f"position={unstable_info['position']}, reference={ref_label!r}, mode={unstable_info['mode_fraction']:.2f}"
        ),
        fontsize=19,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.16, top=0.86, wspace=0.28, hspace=0.40)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def write_summary(
    out_path: Path,
    *,
    profile: str,
    question: str,
    generated_text: str,
    generated_token_labels: list[str],
    stable_info: dict[str, Any],
    unstable_info: dict[str, Any],
) -> None:
    payload = {
        "profile": profile,
        "question": question,
        "generated_text": generated_text,
        "generated_token_labels": generated_token_labels,
        "stable_position": {
            "position": stable_info["position"],
            "mode_fraction": stable_info["mode_fraction"],
            "num_unique": stable_info["num_unique"],
            "counts": stable_info["counts"].tolist(),
            "unique_token_ids": stable_info["unique_token_ids"].tolist(),
            "reference_token_id": stable_info["reference_token_id"],
        },
        "unstable_position": {
            "position": unstable_info["position"],
            "mode_fraction": unstable_info["mode_fraction"],
            "num_unique": unstable_info["num_unique"],
            "counts": unstable_info["counts"].tolist(),
            "unique_token_ids": unstable_info["unique_token_ids"].tolist(),
            "reference_token_id": unstable_info["reference_token_id"],
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    apply_publication_style()
    torch.manual_seed(args.seed)

    config = PROFILE_CONFIGS[args.profile].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / f"review_export_{args.profile}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    )
    model.eval()

    preview_messages = []
    if args.system_prompt.strip():
        preview_messages.append({"role": "system", "content": args.system_prompt.strip()})
    preview_messages.append({"role": "user", "content": args.question.strip()})
    preview_prompt_text = tokenizer.apply_chat_template(preview_messages, tokenize=False, add_generation_prompt=True)
    preview_input_ids = tokenizer(preview_prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        args.device
    )

    base_init_std = float(model.config.init_values["std"])
    analysis_init_std = (
        float(config["INIT_STD_OVERRIDE"]) if config["INIT_STD_OVERRIDE"] is not None else base_init_std * config["INIT_STD_SCALE"]
    )
    analysis_init_scale = 0.0 if base_init_std == 0 else analysis_init_std / base_init_std

    generation_config = GenerationConfig(
        max_new_tokens=config["FULL_OUTPUT_MAX_NEW_TOKENS"],
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
        input_ids=preview_input_ids.clone(),
        generation_config=generation_config,
        num_steps=config["FULL_OUTPUT_NUM_STEPS"],
        max_new_tokens=config["FULL_OUTPUT_MAX_NEW_TOKENS"],
        init_scale=args.preview_init_scale,
    )
    generated_token_ids = generated_trace["generated_token_ids"]
    generated_token_id_list = generated_token_ids.tolist()
    generated_token_labels = decode_token_ids(tokenizer, generated_token_id_list, max_chars=24)
    generated_text = clean_output_text(tokenizer.decode(generated_token_id_list, skip_special_tokens=False))

    stable_scan = collect_position_scan(
        model=model,
        prompt_input_ids=preview_input_ids,
        reference_generated_token_ids=generated_token_ids.to(preview_input_ids.device),
        num_samples=config["SEARCH_NUM_SAMPLES"],
        num_steps=config["SEARCH_NUM_STEPS"],
        init_scale=analysis_init_scale,
        batch_size=min(config["SAMPLE_BATCH_SIZE"], 8),
        max_positions=min(args.search_max_positions, len(generated_token_id_list)),
    )

    unstable_findings = find_variable_generated_positions(
        model=model,
        prompt_input_ids=preview_input_ids,
        reference_generated_token_ids=generated_token_ids.to(preview_input_ids.device),
        num_samples=config["SEARCH_NUM_SAMPLES"],
        num_steps=config["SEARCH_NUM_STEPS"],
        init_scale=analysis_init_scale,
        batch_size=min(config["SAMPLE_BATCH_SIZE"], 8),
        max_positions=min(args.search_max_positions, len(generated_token_id_list)),
        min_unique_tokens=2,
        max_results=1,
    )
    if not unstable_findings:
        unstable_findings = find_variable_generated_positions(
            model=model,
            prompt_input_ids=preview_input_ids,
            reference_generated_token_ids=generated_token_ids.to(preview_input_ids.device),
            num_samples=config["SEARCH_NUM_SAMPLES"],
            num_steps=config["SEARCH_NUM_STEPS"],
            init_scale=analysis_init_scale,
            batch_size=min(config["SAMPLE_BATCH_SIZE"], 8),
            max_positions=len(generated_token_id_list),
            min_unique_tokens=2,
            max_results=1,
        )
    if unstable_findings:
        unstable_info = normalize_variable_finding(unstable_findings[0])
    else:
        unstable_info = min(
            stable_scan,
            key=lambda entry: (entry["mode_fraction"], -entry["num_unique"], entry["position"]),
        )

    stable_candidates = [entry for entry in stable_scan if entry["num_unique"] == 1]
    if stable_candidates:
        stable_info = stable_candidates[0]
    else:
        stable_info = max(stable_scan, key=lambda entry: (entry["mode_fraction"], -entry["position"]))

    export_heatmap(
        run_dir / "01_output_token_residual_heatmap.pdf",
        generated_trace["step_residual_norm"].numpy(),
        generated_token_labels,
    )
    export_stable_vs_unstable(
        out_path=run_dir / "02_stable_vs_unstable_landscapes.pdf",
        model=model,
        tokenizer=tokenizer,
        prompt_input_ids=preview_input_ids,
        generated_token_ids=generated_token_id_list,
        stable_info=stable_info,
        unstable_info=unstable_info,
        num_samples=config["NUM_SAMPLES"],
        num_steps=config["NUM_STEPS"],
        init_scale=analysis_init_scale,
        batch_size=config["SAMPLE_BATCH_SIZE"],
        grid_size=config["PLOT_GRID_SIZE"],
        grid_pad=config["GRID_PAD"],
        smooth=config["RBF_SMOOTH"],
        level_count=config["CONTOUR_LEVEL_COUNT"],
        max_trajectories=config["MAX_TRAJECTORIES_TO_OVERLAY"],
        whiten_coords=config["WHITEN_COORDS"],
    )
    export_unstable_gallery(
        out_path=run_dir / "03_unstable_token_gallery.pdf",
        model=model,
        tokenizer=tokenizer,
        prompt_input_ids=preview_input_ids,
        generated_token_ids=generated_token_id_list,
        unstable_info=unstable_info,
        num_samples=min(max(config["NUM_SAMPLES"], 48), 64),
        num_steps=max(config["NUM_STEPS"], config["SEARCH_NUM_STEPS"]),
        init_scale=analysis_init_scale,
        batch_size=min(config["SAMPLE_BATCH_SIZE"], 8),
        grid_size=config["PLOT_GRID_SIZE"],
        grid_pad=config["GRID_PAD"],
        smooth=config["RBF_SMOOTH"],
        level_count=config["CONTOUR_LEVEL_COUNT"],
        whiten_coords=config["WHITEN_COORDS"],
        gallery_count=args.gallery_max_samples,
    )

    write_summary(
        run_dir / "summary.json",
        profile=args.profile,
        question=args.question,
        generated_text=generated_text,
        generated_token_labels=generated_token_labels,
        stable_info=stable_info,
        unstable_info=unstable_info,
    )

    zip_path = args.output_dir / f"review_export_{args.profile}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.iterdir()):
            zf.write(path, arcname=path.name)

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()
