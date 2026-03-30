from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from vis import (
    DEFAULT_PALETTE,
    DEFAULT_TRAJECTORY_COLORS,
    axis_ranges,
    clean_output_text,
    collect_generated_token_traces,
    collect_next_token_trajectories,
    compact_labels,
    decode_token_ids,
    draw_landscape_with_trajectories,
    find_variable_generated_positions,
    make_levels,
    make_metric_grid_square,
    square_ranges,
)

from export_position_pca_review_figures import (
    DEFAULT_QUESTION,
    DEFAULT_SYSTEM_PROMPT,
    apply_publication_style,
    build_prefix_ids,
    collect_position_scan,
    normalize_variable_finding,
    style_axis,
)


PROFILE_CONFIGS = {
    "light": {
        "NUM_SAMPLES": 24,
        "SAMPLE_BATCH_SIZE": 8,
        "NUM_STEPS": 10,
        "PLOT_GRID_SIZE": 64,
        "GRID_PAD": 0.24,
        "RBF_SMOOTH": 0.20,
        "CONTOUR_LEVEL_COUNT": 10,
        "INIT_STD_SCALE": 12.0,
        "FULL_OUTPUT_NUM_STEPS": 10,
        "FULL_OUTPUT_MAX_NEW_TOKENS": 128,
        "SEARCH_NUM_SAMPLES": 16,
        "SEARCH_NUM_STEPS": 10,
        "UNSTABLE_MAX_RESULTS": 4,
        "PAIRGRID_COMPONENTS": 4,
        "PAIRGRID_TRAJECTORIES": 5,
        "PAIRGRID_MIN_GROUP_SIZE": 2,
        "GALLERY_NUM_SAMPLES": 24,
        "GALLERY_NUM_STEPS": 10,
        "GALLERY_MAX_TOKENS": 3,
        "GALLERY_MAX_TRAJECTORIES": 5,
    },
    "fast": {
        "NUM_SAMPLES": 48,
        "SAMPLE_BATCH_SIZE": 8,
        "NUM_STEPS": 16,
        "PLOT_GRID_SIZE": 84,
        "GRID_PAD": 0.24,
        "RBF_SMOOTH": 0.16,
        "CONTOUR_LEVEL_COUNT": 12,
        "INIT_STD_SCALE": 12.0,
        "FULL_OUTPUT_NUM_STEPS": 16,
        "FULL_OUTPUT_MAX_NEW_TOKENS": 256,
        "SEARCH_NUM_SAMPLES": 24,
        "SEARCH_NUM_STEPS": 12,
        "UNSTABLE_MAX_RESULTS": 8,
        "PAIRGRID_COMPONENTS": 4,
        "PAIRGRID_TRAJECTORIES": 8,
        "PAIRGRID_MIN_GROUP_SIZE": 2,
        "GALLERY_NUM_SAMPLES": 40,
        "GALLERY_NUM_STEPS": 14,
        "GALLERY_MAX_TOKENS": 3,
        "GALLERY_MAX_TRAJECTORIES": 8,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reviewer figures with PCA pair-grid landscapes.")
    parser.add_argument("--model-dir", type=Path, default=Path("/data/hansen/serv12/HRM-Deq/models/huginn-0125"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rebuttal_ar_figures"))
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="light")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--preview-init-scale", type=float, default=0.0)
    parser.add_argument("--search-max-positions", type=int, default=96)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--sample-batch-size", type=int)
    parser.add_argument("--search-num-samples", type=int)
    parser.add_argument("--search-num-steps", type=int)
    parser.add_argument("--full-output-num-steps", type=int)
    parser.add_argument("--full-output-max-new-tokens", type=int)
    parser.add_argument("--pairgrid-trajectories", type=int)
    parser.add_argument("--plot-grid-size", type=int)
    parser.add_argument("--init-std-scale", type=float)
    parser.add_argument("--no-pca-whiten", action="store_true")
    return parser.parse_args()


def compute_pca_coords(states: torch.Tensor, n_components: int, whiten: bool = True) -> dict[str, torch.Tensor]:
    flat = states.reshape(-1, states.shape[-1])
    center = flat.mean(dim=0)
    centered = flat - center
    q = min(n_components, centered.shape[0], centered.shape[1])
    if q < 2:
        raise ValueError(f"Need at least 2 PCA components, got q={q}.")
    _, _, v = torch.pca_lowrank(centered, q=q)
    coords = centered @ v[:, :q]
    if whiten:
        coord_scale = coords.std(dim=0).clamp_min(1e-6)
        coords = coords / coord_scale
    return {
        "coords": coords.reshape(states.shape[0], states.shape[1], q),
        "basis": v[:, :q],
        "center": center,
    }


def pair_indices(n_components: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_components) for j in range(i + 1, n_components)]


def summarize_token_counts(token_ids: torch.Tensor, tokenizer: Any, max_items: int = 4) -> list[tuple[str, int]]:
    counts = Counter(token_ids.tolist()).most_common(max_items)
    labels = decode_token_ids(tokenizer, [item[0] for item in counts], max_chars=20)
    return list(zip(labels, [item[1] for item in counts], strict=True))


def select_representative_indices(token_ids: torch.Tensor, max_trajectories: int) -> list[int]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, token_id in enumerate(token_ids.tolist()):
        groups[token_id].append(idx)
    ordered_tokens = [token for token, _ in Counter(token_ids.tolist()).most_common()]
    chosen = []
    round_idx = 0
    while len(chosen) < max_trajectories:
        added = False
        for token in ordered_tokens:
            bucket = groups[token]
            if round_idx < len(bucket):
                chosen.append(bucket[round_idx])
                added = True
                if len(chosen) == max_trajectories:
                    break
        if not added:
            break
        round_idx += 1
    return chosen


def select_centroid_representatives(
    final_coords: np.ndarray,
    token_ids: torch.Tensor,
    max_tokens: int | None = None,
) -> list[int]:
    ordered_tokens = [token for token, _ in Counter(token_ids.tolist()).most_common(max_tokens)]
    token_ids_np = token_ids.numpy()
    chosen: list[int] = []
    for token_id in ordered_tokens:
        group_indices = np.flatnonzero(token_ids_np == token_id)
        if group_indices.size == 0:
            continue
        group_points = final_coords[group_indices]
        centroid = group_points.mean(axis=0)
        distances = np.linalg.norm(group_points - centroid, axis=1)
        chosen.append(int(group_indices[int(np.argmin(distances))]))
    return chosen


def select_farthest_endpoint_representatives(
    final_coords: np.ndarray,
    token_ids: torch.Tensor,
    max_tokens: int | None = None,
) -> list[int]:
    ordered_tokens = [token for token, _ in Counter(token_ids.tolist()).most_common(max_tokens)]
    if not ordered_tokens:
        return []

    token_ids_np = token_ids.numpy()
    groups = {token: np.flatnonzero(token_ids_np == token) for token in ordered_tokens}

    if len(ordered_tokens) == 1:
        group_indices = groups[ordered_tokens[0]]
        return [int(group_indices[0])] if group_indices.size else []

    selected: dict[int, int] = {}
    best_pair: tuple[int, int] | None = None
    best_distance = -1.0

    for i, token_i in enumerate(ordered_tokens):
        idx_i = groups[token_i]
        if idx_i.size == 0:
            continue
        points_i = final_coords[idx_i]
        for token_j in ordered_tokens[i + 1 :]:
            idx_j = groups[token_j]
            if idx_j.size == 0:
                continue
            points_j = final_coords[idx_j]
            pair_dist = np.linalg.norm(points_i[:, None, :] - points_j[None, :, :], axis=-1)
            flat_best = int(np.argmax(pair_dist))
            local_i, local_j = np.unravel_index(flat_best, pair_dist.shape)
            dist = float(pair_dist[local_i, local_j])
            if dist > best_distance:
                best_distance = dist
                best_pair = (int(idx_i[local_i]), int(idx_j[local_j]))
                selected = {
                    token_i: int(idx_i[local_i]),
                    token_j: int(idx_j[local_j]),
                }

    if best_pair is None:
        return select_centroid_representatives(final_coords, token_ids, max_tokens=max_tokens)

    selected_points = [final_coords[idx] for idx in selected.values()]
    for token in ordered_tokens:
        if token in selected:
            continue
        idxs = groups[token]
        if idxs.size == 0:
            continue
        points = final_coords[idxs]
        distances = []
        for point in points:
            min_dist = min(float(np.linalg.norm(point - chosen)) for chosen in selected_points)
            distances.append(min_dist)
        best_local = int(np.argmax(np.asarray(distances)))
        chosen_idx = int(idxs[best_local])
        selected[token] = chosen_idx
        selected_points.append(final_coords[chosen_idx])

    return [selected[token] for token in ordered_tokens if token in selected]


def format_metric(value: float) -> str:
    value = float(value)
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value >= 1e3 or abs_value < 1e-2:
        return f"{value:.2e}"
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def select_pairgrid_overlay(
    analysis: dict[str, Any],
    pairgrid_trajectories: int,
    overlay_strategy: str,
) -> tuple[list[int], str]:
    if overlay_strategy == "per_token_centroid":
        overlay = select_centroid_representatives(
            analysis["coords"][:, -1, :],
            analysis["predicted_token_ids"],
            max_tokens=None,
        )
        overlay_mode = "token"
    elif overlay_strategy == "per_token_farthest_end":
        overlay = select_farthest_endpoint_representatives(
            analysis["coords"][:, -1, :],
            analysis["predicted_token_ids"],
            max_tokens=None,
        )
        overlay_mode = "token"
    else:
        overlay = select_representative_indices(analysis["predicted_token_ids"], pairgrid_trajectories)
        overlay_mode = "gray"
    return overlay, overlay_mode


def compute_pair_end_variances(
    analysis: dict[str, Any],
    overlay_indices: list[int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if len(overlay_indices) == 0:
        return results
    n_components = analysis["coords"].shape[-1]
    for row in range(n_components):
        for col in range(n_components):
            if row == col:
                continue
            coords_pair = analysis["coords"][:, :, [row, col]]
            rep_end_points = coords_pair[overlay_indices, -1, :]
            end_var = float(np.var(rep_end_points, axis=0).sum()) if len(overlay_indices) >= 2 else 0.0
            results.append(
                {
                    "pair": (row, col),
                    "label": f"PC {row + 1} vs PC {col + 1}",
                    "end_var": end_var,
                }
            )
    results.sort(key=lambda item: item["end_var"], reverse=True)
    return results


def compute_pair_score(
    final_coords: np.ndarray,
    predicted_token_ids: torch.Tensor,
    pair: tuple[int, int],
    min_group_size: int,
) -> dict[str, Any] | None:
    pair_points = final_coords[:, [pair[0], pair[1]]]
    unique_ids, counts = torch.unique(predicted_token_ids, return_counts=True)
    keep = counts >= min_group_size
    unique_ids = unique_ids[keep]
    counts = counts[keep]
    if unique_ids.numel() < 2:
        return None

    centroids = []
    within = []
    for token_id in unique_ids.tolist():
        group_points = pair_points[predicted_token_ids.numpy() == token_id]
        centroid = group_points.mean(axis=0)
        centroids.append(centroid)
        within.append(float(np.linalg.norm(group_points - centroid, axis=1).mean()))

    min_inter = math.inf
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            min_inter = min(min_inter, dist)

    avg_within = max(float(np.mean(within)), 1e-6)
    return {
        "pair": pair,
        "score": float(min_inter / avg_within),
        "min_inter": float(min_inter),
        "avg_within": float(avg_within),
        "num_groups": int(unique_ids.numel()),
    }


def analyze_position(
    *,
    model: Any,
    tokenizer: Any,
    prompt_input_ids: torch.Tensor,
    generated_token_ids: list[int],
    info: dict[str, Any],
    num_samples: int,
    num_steps: int,
    init_scale: float,
    batch_size: int,
    n_components: int,
    min_group_size: int,
    pca_whiten: bool,
) -> dict[str, Any]:
    prefix_ids = build_prefix_ids(prompt_input_ids, generated_token_ids, info["position"])
    trace = collect_next_token_trajectories(
        model=model,
        input_ids=prefix_ids,
        num_samples=num_samples,
        num_steps=num_steps,
        init_scale=init_scale,
        batch_size=batch_size,
    )
    projected = compute_pca_coords(trace["states"], n_components=n_components, whiten=pca_whiten)
    coords = projected["coords"].numpy()
    final_coords = coords[:, -1, :]
    predicted_token_ids = trace["predicted_token_ids"].cpu()
    pair_scores = []
    for pair in pair_indices(coords.shape[-1]):
        score = compute_pair_score(final_coords, predicted_token_ids, pair, min_group_size=min_group_size)
        if score is not None:
            pair_scores.append(score)
    pair_scores.sort(key=lambda item: item["score"], reverse=True)
    return {
        "info": info,
        "trace": trace,
        "coords": coords,
        "predicted_token_ids": predicted_token_ids,
        "predicted_token_labels": decode_token_ids(tokenizer, predicted_token_ids.tolist(), max_chars=24),
        "token_count_summary": summarize_token_counts(predicted_token_ids, tokenizer),
        "pair_scores": pair_scores,
        "best_pair": pair_scores[0]["pair"] if pair_scores else (0, 1),
        "pca_whiten": pca_whiten,
    }


def make_token_style_maps(
    predicted_token_ids: torch.Tensor,
    tokenizer: Any,
    max_tokens: int,
) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    counts = Counter(predicted_token_ids.tolist()).most_common(max_tokens)
    token_ids = [item[0] for item in counts]
    labels = decode_token_ids(tokenizer, token_ids, max_chars=20)
    palette = ("#2563EB", "#DC2626", "#16A34A", "#B45309", "#7C3AED", "#0891B2")
    marker_cycle = ("*", "^", "s", "P", "X", "D")
    color_map = {token_id: palette[i % len(palette)] for i, token_id in enumerate(token_ids)}
    marker_map = {token_id: marker_cycle[i % len(marker_cycle)] for i, token_id in enumerate(token_ids)}
    label_map = {token_id: label for token_id, label in zip(token_ids, labels, strict=True)}
    return color_map, marker_map, label_map


def scatter_token_points(
    ax: plt.Axes,
    *,
    points: np.ndarray,
    token_ids: list[int] | np.ndarray,
    token_color_map: dict[int, str],
    token_marker_map: dict[int, str],
    size: float,
    alpha: float,
    edgecolors: str,
    linewidths: float,
    zorder: int,
) -> None:
    token_ids_np = np.asarray(token_ids)
    for token_id in token_color_map:
        mask = token_ids_np == token_id
        if not np.any(mask):
            continue
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            c=token_color_map[token_id],
            marker=token_marker_map[token_id],
            s=size,
            alpha=alpha,
            edgecolors=edgecolors,
            linewidths=linewidths,
            zorder=zorder,
        )


def add_common_legends(
    fig: plt.Figure,
    token_color_map: dict[int, str],
    token_marker_map: dict[int, str],
    token_label_map: dict[int, str],
    anchor_y: float,
) -> None:
    marker_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.0, label="Trajectory"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=DEFAULT_PALETTE["highlight"], markeredgecolor="#111111", markersize=10, label="Start"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=DEFAULT_PALETTE["accent"], markeredgecolor="#111111", markersize=9, label="End"),
    ]
    marker_legend = fig.legend(
        handles=marker_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, anchor_y),
        frameon=True,
        title="Trajectory Markers",
        title_fontsize=13,
    )
    fig.add_artist(marker_legend)

    token_handles = [
        Line2D(
            [0],
            [0],
            marker=token_marker_map[token_id],
            color="w",
            markerfacecolor=color_map,
            markeredgecolor="#222222",
            markersize=9,
            label=token_label_map[token_id],
        )
        for token_id, color_map in token_color_map.items()
    ]
    if token_handles:
        fig.legend(
            handles=token_handles,
            loc="upper right",
            bbox_to_anchor=(0.985, anchor_y - 0.14),
            frameon=True,
            title="Decoded Tokens",
            title_fontsize=13,
        )


def robust_value_range(values: np.ndarray, low_q: float = 0.05, high_q: float = 0.95) -> tuple[float, float]:
    value_min = float(np.quantile(values, low_q))
    value_max = float(np.quantile(values, high_q))
    if not np.isfinite(value_min) or not np.isfinite(value_max) or value_max <= value_min:
        value_min = float(np.min(values))
        value_max = float(np.max(values))
    if value_max <= value_min:
        value_max = value_min + 1e-3
    return value_min, value_max


def draw_pair_landscape(
    ax: plt.Axes,
    *,
    coords_pair: np.ndarray,
    residuals: np.ndarray,
    grid_size: int,
    grid_pad: float,
    smooth: float,
    level_count: int,
    max_fit_points: int | None = 2048,
    bounds_points_override: np.ndarray | None = None,
    metric_tail_steps: int | None = None,
    cmap: str = "viridis",
    square_axes: bool = True,
) -> Any:
    if metric_tail_steps is not None and metric_tail_steps > 0:
        metric_slice = slice(-metric_tail_steps, None)
        metric_points = coords_pair[:, metric_slice, :].reshape(-1, 2)
        values = np.log10(np.maximum(residuals[:, metric_slice].reshape(-1), 1e-6))
    else:
        metric_points = coords_pair[:, 1:, :].reshape(-1, 2)
        values = np.log10(np.maximum(residuals.reshape(-1), 1e-6))
    bounds_points = coords_pair.reshape(-1, 2) if bounds_points_override is None else bounds_points_override
    if max_fit_points is not None and metric_points.shape[0] > max_fit_points:
        keep = np.linspace(0, metric_points.shape[0] - 1, num=max_fit_points, dtype=int)
        metric_points = metric_points[keep]
        values = values[keep]
    value_min, value_max = robust_value_range(values)
    levels = make_levels(value_min, value_max, level_count=level_count)
    xs, ys, grid_z = make_metric_grid_square(
        metric_points,
        values,
        grid_size=grid_size,
        pad=grid_pad,
        smooth=smooth,
        value_min=value_min,
        value_max=value_max,
        bounds_points=bounds_points,
        square=square_axes,
    )
    if square_axes:
        x_range, y_range = square_ranges(bounds_points[:, 0], bounds_points[:, 1], pad=grid_pad)
    else:
        x_range, y_range = axis_ranges(bounds_points[:, 0], bounds_points[:, 1], pad=grid_pad)
    contour = ax.contourf(xs, ys, grid_z, levels=levels, cmap=cmap)
    ax.contour(xs, ys, grid_z, levels=levels, colors="white", linewidths=0.8, alpha=0.65)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal", adjustable="box")
    return contour


def overlay_pair_trajectories(
    ax: plt.Axes,
    *,
    coords_pair: np.ndarray,
    overlay_indices: list[int],
    predicted_token_ids: torch.Tensor,
    token_color_map: dict[int, str] | None,
    mode: str,
) -> None:
    overlay_count = len(overlay_indices)
    for sample_idx in overlay_indices:
        traj = coords_pair[sample_idx]
        token_id = int(predicted_token_ids[sample_idx].item())
        if mode == "token" and token_color_map is not None:
            color = token_color_map.get(token_id, "#9A9A9A")
            alpha = 0.44
            linewidth = 1.0
            start_kwargs = {
                "s": 26,
                "c": DEFAULT_PALETTE["highlight"],
                "edgecolors": "#111111",
                "linewidths": 0.7,
                "marker": "^",
            }
            end_kwargs = {
                "s": 24,
                "c": DEFAULT_PALETTE["accent"],
                "edgecolors": "#111111",
                "linewidths": 0.7,
                "marker": "D",
            }
        else:
            color = "#B8B8B8"
            alpha = 0.65 if overlay_count <= 12 else 0.28
            linewidth = 1.4 if overlay_count <= 12 else 0.85
            if overlay_count <= 12:
                start_kwargs = {
                    "s": 52,
                    "c": DEFAULT_PALETTE["highlight"],
                    "edgecolors": "#111111",
                    "linewidths": 0.9,
                    "marker": "^",
                }
                end_kwargs = {
                    "s": 48,
                    "c": DEFAULT_PALETTE["accent"],
                    "edgecolors": "#111111",
                    "linewidths": 0.9,
                    "marker": "D",
                }
            else:
                start_kwargs = {
                    "s": 28,
                    "facecolors": "none",
                    "edgecolors": DEFAULT_PALETTE["highlight"],
                    "linewidths": 0.9,
                    "marker": "^",
                }
                end_kwargs = {
                    "s": 24,
                    "facecolors": "none",
                    "edgecolors": DEFAULT_PALETTE["accent"],
                    "linewidths": 0.9,
                    "marker": "D",
                }
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=linewidth, alpha=alpha, zorder=3)
        ax.scatter(
            [traj[0, 0]],
            [traj[0, 1]],
            **start_kwargs,
            zorder=4,
        )
        ax.scatter(
            [traj[-1, 0]],
            [traj[-1, 1]],
            **end_kwargs,
            zorder=4,
        )


def export_full_answer_heatmap(
    out_path: Path,
    generated_step_residual: np.ndarray,
) -> None:
    heatmap = generated_step_residual.T
    fig_width = max(16.0, min(28.0, heatmap.shape[1] * 0.18))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5), dpi=220)
    im = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Residual Heatmap Across the Full Generated Answer", fontsize=18, fontweight="bold")
    ax.set_xlabel("Generated Token Position", fontsize=16, fontweight="bold")
    ax.set_ylabel("Recurrent Iteration", fontsize=16, fontweight="bold")
    xtick_step = max(1, int(math.ceil(heatmap.shape[1] / 18)))
    xticks = np.arange(0, heatmap.shape[1], xtick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks])
    yticks = np.arange(heatmap.shape[0])
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(y + 1)) for y in yticks])
    style_axis(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("||f_t - f_{t-1}||_2", fontsize=15, fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(12)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", pad_inches=0.08)
    plt.close(fig)


def plot_pair_subplot(
    ax: plt.Axes,
    *,
    analysis: dict[str, Any],
    pair: tuple[int, int],
    grid_size: int,
    grid_pad: float,
    smooth: float,
    level_count: int,
    max_fit_points: int,
    overlay_indices: list[int],
    token_color_map: dict[int, str],
    token_marker_map: dict[int, str],
    overlay_mode: str,
    show_x_label: bool,
    show_y_label: bool,
    show_trajectories: bool = True,
    show_end_var_text: bool = True,
    focus_tail_steps: int | None = None,
    landscape_tail_steps: int | None = None,
    show_all_endpoints: bool = True,
) -> Any:
    coords_pair = analysis["coords"][:, :, [pair[0], pair[1]]]
    final_points = coords_pair[:, -1, :]
    bounds_points_override = None
    if focus_tail_steps is not None and focus_tail_steps > 0:
        focus_tail = min(focus_tail_steps, coords_pair.shape[1])
        focus_parts = [
            final_points,
            coords_pair[:, -focus_tail:, :].reshape(-1, 2),
        ]
        if overlay_indices:
            focus_parts.append(coords_pair[overlay_indices, -focus_tail:, :].reshape(-1, 2))
        bounds_points_override = np.concatenate(focus_parts, axis=0)
    contour = draw_pair_landscape(
        ax,
        coords_pair=coords_pair,
        residuals=analysis["trace"]["residual_norms"].numpy(),
        grid_size=grid_size,
        grid_pad=grid_pad,
        smooth=smooth,
        level_count=level_count,
        max_fit_points=max_fit_points,
        bounds_points_override=bounds_points_override,
        metric_tail_steps=landscape_tail_steps,
    )
    if show_trajectories and overlay_indices:
        overlay_pair_trajectories(
            ax,
            coords_pair=coords_pair,
            overlay_indices=overlay_indices,
            predicted_token_ids=analysis["predicted_token_ids"],
            token_color_map=token_color_map,
            mode=overlay_mode,
        )
    final_token_ids = analysis["predicted_token_ids"].tolist()
    if show_all_endpoints or not overlay_indices:
        endpoint_points = final_points
        endpoint_token_ids = final_token_ids
    else:
        endpoint_points = final_points[overlay_indices]
        endpoint_token_ids = [final_token_ids[idx] for idx in overlay_indices]
    scatter_token_points(
        ax,
        points=endpoint_points,
        token_ids=endpoint_token_ids,
        token_color_map=token_color_map,
        token_marker_map=token_marker_map,
        size=70 if not show_all_endpoints else 26,
        alpha=0.96 if not show_all_endpoints else 0.92,
        edgecolors="#111111",
        linewidths=0.42 if not show_all_endpoints else 0.22,
        zorder=5,
    )
    rep_end_points = coords_pair[overlay_indices, -1, :]
    rep_end_var = float(np.var(rep_end_points, axis=0).sum()) if len(overlay_indices) >= 2 else 0.0
    if show_end_var_text:
        ax.text(
            0.5,
            -0.18,
            f"End var={format_metric(rep_end_var)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color="#444444",
            clip_on=False,
        )
    if show_x_label:
        ax.set_xlabel(f"PC {pair[0] + 1}", fontsize=13, fontweight="bold")
    else:
        ax.set_xlabel("")
    ax.set_ylabel("")
    style_axis(ax)
    return contour


def export_single_pairgrid_landscape(
    *,
    out_path: Path,
    analysis: dict[str, Any],
    tokenizer: Any,
    grid_size: int,
    grid_pad: float,
    smooth: float,
    level_count: int,
    max_fit_points: int,
    pairgrid_trajectories: int,
    gallery_max_tokens: int,
    overlay_strategy: str,
    title_prefix: str,
    token_color_legend_title: str,
    show_trajectories: bool = True,
    show_end_var_text: bool = True,
    title_text: str | None = None,
    focus_tail_steps: int | None = None,
    landscape_tail_steps: int | None = None,
    show_all_endpoints: bool = True,
    show_header_text: bool = True,
) -> None:
    overlay, overlay_mode = select_pairgrid_overlay(
        analysis,
        pairgrid_trajectories=pairgrid_trajectories,
        overlay_strategy=overlay_strategy,
    )
    token_color_map, token_marker_map, token_label_map = make_token_style_maps(
        analysis["predicted_token_ids"], tokenizer, max_tokens=gallery_max_tokens
    )

    n_components = analysis["coords"].shape[-1]
    fig_width = max(17.0, 2.55 * n_components + 7.0)
    fig_height = max(12.0, 2.55 * n_components + 3.0)
    hspace = 0.34 if n_components >= 6 else 0.28
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=220)
    gs = fig.add_gridspec(n_components, n_components, left=0.07, right=0.79, top=0.90, bottom=0.07, hspace=hspace, wspace=0.18)

    contour = None
    for row in range(n_components):
        for col in range(n_components):
            ax = fig.add_subplot(gs[row, col])
            if row == col:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    f"PC {row + 1}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color="#444444",
                )
                continue
            contour = plot_pair_subplot(
                ax,
                analysis=analysis,
                pair=(col, row),
                grid_size=grid_size,
                grid_pad=grid_pad,
                smooth=smooth,
                level_count=level_count,
                max_fit_points=max_fit_points,
                overlay_indices=overlay,
                token_color_map=token_color_map,
                token_marker_map=token_marker_map,
                overlay_mode=overlay_mode,
                show_x_label=row == n_components - 1,
                show_y_label=col == 0,
                show_trajectories=show_trajectories,
                show_end_var_text=show_end_var_text,
                focus_tail_steps=focus_tail_steps,
                landscape_tail_steps=landscape_tail_steps,
                show_all_endpoints=show_all_endpoints,
            )
            if row != n_components - 1:
                ax.set_xticklabels([])
            if col != 0:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f"PC {col + 1}", fontsize=16, fontweight="bold", pad=6)
            if col == 0:
                ax.text(
                    -0.42,
                    0.5,
                    f"PC {row + 1}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=16,
                    fontweight="bold",
                )

    if show_header_text:
        ref_label = decode_token_ids(tokenizer, [analysis["info"]["reference_token_id"]], max_chars=24)[0]
        fig.text(
            0.07,
            0.93,
            (
                f"{title_prefix}: position={analysis['info']['position']}, ref={ref_label!r}, "
                f"mode={analysis['info']['mode_fraction']:.2f}, top={analysis['token_count_summary']}"
            ),
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
        )
    if title_text == "":
        pass
    elif title_text is None:
        fig.suptitle(f"{title_prefix} Token-Conditional PCA Pair Landscape", fontsize=20, fontweight="bold", y=0.985)
    else:
        fig.suptitle(title_text, fontsize=20, fontweight="bold", y=0.985)

    if show_trajectories:
        marker_handles = [
            Line2D([0], [0], color="#B8B8B8", linewidth=1.8, label="Trajectory"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor=DEFAULT_PALETTE["highlight"], markeredgecolor="#111111", markersize=10, label="Start"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor=DEFAULT_PALETTE["accent"], markeredgecolor="#111111", markersize=9, label="End"),
        ]
        fig.legend(
            handles=marker_handles,
            loc="upper left",
            bbox_to_anchor=(0.80, 0.90),
            frameon=True,
            title="Trajectory Markers",
            fontsize=14,
            title_fontsize=15,
        )

    token_handles = [
        Line2D(
            [0],
            [0],
            marker=token_marker_map[token_id],
            color="w",
            markerfacecolor=color,
            markeredgecolor="#222222",
            markersize=12,
            label=token_label_map[token_id],
        )
        for token_id, color in token_color_map.items()
    ]
    if token_handles:
        fig.legend(
            handles=token_handles,
            loc="upper left",
            bbox_to_anchor=(0.80, 0.74 if show_trajectories else 0.88),
            frameon=True,
            title=token_color_legend_title,
            fontsize=14,
            title_fontsize=15,
        )

    if contour is not None:
        cax = fig.add_axes([0.91, 0.19, 0.028, 0.36])
        cbar = fig.colorbar(contour, cax=cax)
        cbar.set_label("log10 ||f_t - f_{t-1}||_2", fontsize=17, fontweight="bold")
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")
            tick.set_fontsize(14)

    fig.savefig(out_path, format="pdf", pad_inches=0.08)
    plt.close(fig)


def export_unstable_gallery(
    *,
    out_path: Path,
    unstable_analysis: dict[str, Any],
    tokenizer: Any,
    grid_size: int,
    grid_pad: float,
    smooth: float,
    level_count: int,
    max_fit_points: int,
    gallery_max_tokens: int,
    gallery_max_trajectories: int,
    overlay_indices: list[int] | None = None,
    focus_tail_steps: int | None = None,
    landscape_tail_steps: int | None = None,
    show_trajectories: bool = True,
    title_text: str | None = None,
    use_latex_axis_labels: bool = False,
    show_all_endpoints: bool = True,
) -> None:
    best_pair = unstable_analysis["best_pair"]
    coords_pair = unstable_analysis["coords"][:, :, [best_pair[0], best_pair[1]]]
    residuals = unstable_analysis["trace"]["residual_norms"].numpy()
    predicted_token_ids = unstable_analysis["predicted_token_ids"]
    token_color_map, token_marker_map, token_label_map = make_token_style_maps(
        predicted_token_ids, tokenizer, max_tokens=gallery_max_tokens
    )

    groups: dict[int, list[int]] = defaultdict(list)
    for idx, token_id in enumerate(predicted_token_ids.tolist()):
        if token_id in token_color_map:
            groups[token_id].append(idx)

    if overlay_indices is None:
        overlay_indices = []
        for token_id, _ in Counter(predicted_token_ids.tolist()).most_common(gallery_max_tokens):
            overlay_indices.extend(groups.get(token_id, [])[:2])
        overlay_indices = overlay_indices[:gallery_max_trajectories]
        if not overlay_indices:
            overlay_indices = select_representative_indices(predicted_token_ids, gallery_max_trajectories)

    final_points = coords_pair[:, -1, :]
    fig, ax = plt.subplots(figsize=(9.6, 8.2), dpi=220)
    focus_tail = min(focus_tail_steps or 8, coords_pair.shape[1])
    focus_bounds = np.concatenate(
        [
            final_points,
            coords_pair[:, -focus_tail:, :].reshape(-1, 2),
            coords_pair[overlay_indices, -focus_tail:, :].reshape(-1, 2),
        ],
        axis=0,
    )
    contour = draw_pair_landscape(
        ax,
        coords_pair=coords_pair,
        residuals=residuals,
        grid_size=grid_size,
        grid_pad=grid_pad + 0.08,
        smooth=smooth,
        level_count=level_count,
        max_fit_points=max_fit_points,
        bounds_points_override=focus_bounds,
        metric_tail_steps=landscape_tail_steps,
    )
    if show_trajectories:
        overlay_pair_trajectories(
            ax,
            coords_pair=coords_pair,
            overlay_indices=overlay_indices,
            predicted_token_ids=predicted_token_ids,
            token_color_map=token_color_map,
            mode="token",
        )
    if show_all_endpoints or not overlay_indices:
        endpoint_points = final_points
        endpoint_token_ids = predicted_token_ids.tolist()
        point_size = 42
        point_alpha = 0.88
        point_lw = 0.28
    else:
        endpoint_points = final_points[overlay_indices]
        all_token_ids = predicted_token_ids.tolist()
        endpoint_token_ids = [all_token_ids[idx] for idx in overlay_indices]
        point_size = 86
        point_alpha = 0.96
        point_lw = 0.42
    scatter_token_points(
        ax,
        points=endpoint_points,
        token_ids=endpoint_token_ids,
        token_color_map=token_color_map,
        token_marker_map=token_marker_map,
        size=point_size,
        alpha=point_alpha,
        edgecolors="#111111",
        linewidths=point_lw,
        zorder=5,
    )
    if use_latex_axis_labels:
        ax.set_xlabel(fr"$\mathrm{{PC}}_{{{best_pair[0] + 1}}}$", fontsize=15, fontweight="bold")
        ax.set_ylabel(fr"$\mathrm{{PC}}_{{{best_pair[1] + 1}}}$", fontsize=15, fontweight="bold")
    else:
        ax.set_xlabel(f"PC {best_pair[0] + 1}", fontsize=15, fontweight="bold")
        ax.set_ylabel(f"PC {best_pair[1] + 1}", fontsize=15, fontweight="bold")
    ref_label = decode_token_ids(tokenizer, [unstable_analysis["info"]["reference_token_id"]], max_chars=24)[0]
    best_pair_score = unstable_analysis["pair_scores"][0]["score"] if unstable_analysis["pair_scores"] else float("nan")
    if title_text is None:
        ax.set_title(
            (
                f"Unstable Gallery on Best-Separated Pair: PC {best_pair[0] + 1} vs PC {best_pair[1] + 1}\n"
                f"position={unstable_analysis['info']['position']}, ref={ref_label!r}, "
                f"mode={unstable_analysis['info']['mode_fraction']:.2f}, score={best_pair_score:.2f}"
            ),
            fontsize=17,
            fontweight="bold",
        )
    else:
        ax.set_title(title_text, fontsize=17, fontweight="bold")
    style_axis(ax)

    token_handles = [
        Line2D(
            [0],
            [0],
            marker=token_marker_map[token_id],
            color="w",
            markerfacecolor=color,
            markeredgecolor="#222222",
            markersize=12,
            label=token_label_map[token_id],
        )
        for token_id, color in token_color_map.items()
    ]
    if token_handles:
        legend_loc = "upper left" if not show_trajectories else "upper right"
        ax.legend(handles=token_handles, loc=legend_loc, frameon=True, title="Decoded Tokens", fontsize=14, title_fontsize=15)
    if show_trajectories:
        marker_handles = [
            Line2D([0], [0], color="#444444", linewidth=2.0, label="Trajectory"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor=DEFAULT_PALETTE["highlight"], markeredgecolor="#111111", markersize=10, label="Start"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor=DEFAULT_PALETTE["accent"], markeredgecolor="#111111", markersize=9, label="End"),
        ]
        legend1 = ax.legend(handles=marker_handles, loc="upper left", frameon=True, title="Trajectory Markers", fontsize=14, title_fontsize=15)
        ax.add_artist(legend1)
        if token_handles:
            ax.legend(handles=token_handles, loc="upper right", frameon=True, title="Decoded Tokens", fontsize=14, title_fontsize=15)

    cbar = fig.colorbar(contour, ax=ax, fraction=0.055, pad=0.035)
    cbar.set_label(r"$\log_{10}\,\|f_t - f_{t-1}\|_2$", fontsize=17, fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(14)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", pad_inches=0.08)
    plt.close(fig)


def export_single_pair_figure(
    *,
    out_path: Path,
    analysis: dict[str, Any],
    tokenizer: Any,
    pair: tuple[int, int],
    grid_size: int,
    grid_pad: float,
    smooth: float,
    level_count: int,
    max_fit_points: int,
    gallery_max_tokens: int,
    title_text: str | None = None,
    use_latex_axis_labels: bool = False,
    show_trajectories: bool = False,
    overlay_indices: list[int] | None = None,
    overlay_strategy: str = "per_token_farthest_end",
    focus_tail_steps: int | None = None,
    landscape_tail_steps: int | None = None,
    show_all_endpoints: bool = True,
) -> None:
    if overlay_indices is None:
        overlay_indices, overlay_mode = select_pairgrid_overlay(
            analysis,
            pairgrid_trajectories=512,
            overlay_strategy=overlay_strategy,
        )
    else:
        overlay_mode = "token"

    coords_pair = analysis["coords"][:, :, [pair[0], pair[1]]]
    residuals = analysis["trace"]["residual_norms"].numpy()
    predicted_token_ids = analysis["predicted_token_ids"]
    token_color_map, token_marker_map, token_label_map = make_token_style_maps(
        predicted_token_ids, tokenizer, max_tokens=gallery_max_tokens
    )
    final_points = coords_pair[:, -1, :]
    bounds_points_override = None
    if focus_tail_steps is not None and focus_tail_steps > 0:
        focus_tail = min(focus_tail_steps, coords_pair.shape[1])
        focus_parts = [
            final_points,
            coords_pair[:, -focus_tail:, :].reshape(-1, 2),
        ]
        if overlay_indices:
            focus_parts.append(coords_pair[overlay_indices, -focus_tail:, :].reshape(-1, 2))
        bounds_points_override = np.concatenate(focus_parts, axis=0)

    fig, ax = plt.subplots(figsize=(8.8, 7.6), dpi=220)
    contour = draw_pair_landscape(
        ax,
        coords_pair=coords_pair,
        residuals=residuals,
        grid_size=grid_size,
        grid_pad=grid_pad,
        smooth=smooth,
        level_count=level_count,
        max_fit_points=max_fit_points,
        bounds_points_override=bounds_points_override,
        metric_tail_steps=landscape_tail_steps,
    )
    if show_trajectories and overlay_indices:
        overlay_pair_trajectories(
            ax,
            coords_pair=coords_pair,
            overlay_indices=overlay_indices,
            predicted_token_ids=predicted_token_ids,
            token_color_map=token_color_map,
            mode=overlay_mode,
        )
    if show_all_endpoints or not overlay_indices:
        endpoint_points = final_points
        endpoint_token_ids = predicted_token_ids.tolist()
        point_size = 46
        point_alpha = 0.92
        point_lw = 0.24
    else:
        endpoint_points = final_points[overlay_indices]
        all_token_ids = predicted_token_ids.tolist()
        endpoint_token_ids = [all_token_ids[idx] for idx in overlay_indices]
        point_size = 86
        point_alpha = 0.96
        point_lw = 0.42
    scatter_token_points(
        ax,
        points=endpoint_points,
        token_ids=endpoint_token_ids,
        token_color_map=token_color_map,
        token_marker_map=token_marker_map,
        size=point_size,
        alpha=point_alpha,
        edgecolors="#111111",
        linewidths=point_lw,
        zorder=5,
    )

    if use_latex_axis_labels:
        ax.set_xlabel(fr"$\mathrm{{PC}}_{{{pair[0] + 1}}}$", fontsize=16, fontweight="bold")
        ax.set_ylabel(fr"$\mathrm{{PC}}_{{{pair[1] + 1}}}$", fontsize=16, fontweight="bold")
    else:
        ax.set_xlabel(f"PC {pair[0] + 1}", fontsize=16, fontweight="bold")
        ax.set_ylabel(f"PC {pair[1] + 1}", fontsize=16, fontweight="bold")

    if title_text is None:
        ax.set_title(f"PC {pair[0] + 1} vs PC {pair[1] + 1}", fontsize=18, fontweight="bold")
    else:
        ax.set_title(title_text, fontsize=18, fontweight="bold")
    style_axis(ax)

    token_handles = [
        Line2D(
            [0],
            [0],
            marker=token_marker_map[token_id],
            color="w",
            markerfacecolor=color,
            markeredgecolor="#222222",
            markersize=12,
            label=token_label_map[token_id],
        )
        for token_id, color in token_color_map.items()
    ]
    if token_handles:
        ax.legend(handles=token_handles, loc="upper left", frameon=True, title="Decoded Tokens", fontsize=14, title_fontsize=15)

    cbar = fig.colorbar(contour, ax=ax, fraction=0.055, pad=0.035)
    cbar.set_label(r"$\log_{10}\,\|f_t - f_{t-1}\|_2$", fontsize=17, fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(14)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", pad_inches=0.08)
    plt.close(fig)


def write_summary(
    out_path: Path,
    *,
    profile: str,
    generated_text: str,
    stable_analysis: dict[str, Any],
    unstable_analysis: dict[str, Any],
) -> None:
    payload = {
        "profile": profile,
        "generated_text": generated_text,
        "stable_position": {
            "position": stable_analysis["info"]["position"],
            "mode_fraction": stable_analysis["info"]["mode_fraction"],
            "top_tokens": stable_analysis["token_count_summary"],
            "pca_whiten": stable_analysis["pca_whiten"],
        },
        "unstable_position": {
            "position": unstable_analysis["info"]["position"],
            "mode_fraction": unstable_analysis["info"]["mode_fraction"],
            "top_tokens": unstable_analysis["token_count_summary"],
            "pca_whiten": unstable_analysis["pca_whiten"],
            "best_pair": list(unstable_analysis["best_pair"]),
            "best_pair_score": unstable_analysis["pair_scores"][0]["score"] if unstable_analysis["pair_scores"] else None,
            "pair_scores": unstable_analysis["pair_scores"][:6],
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    apply_publication_style()
    torch.manual_seed(args.seed)
    config = PROFILE_CONFIGS[args.profile].copy()
    override_map = {
        "NUM_SAMPLES": args.num_samples,
        "NUM_STEPS": args.num_steps,
        "SAMPLE_BATCH_SIZE": args.sample_batch_size,
        "SEARCH_NUM_SAMPLES": args.search_num_samples,
        "SEARCH_NUM_STEPS": args.search_num_steps,
        "FULL_OUTPUT_NUM_STEPS": args.full_output_num_steps,
        "FULL_OUTPUT_MAX_NEW_TOKENS": args.full_output_max_new_tokens,
        "PAIRGRID_TRAJECTORIES": args.pairgrid_trajectories,
        "PLOT_GRID_SIZE": args.plot_grid_size,
        "INIT_STD_SCALE": args.init_std_scale,
    }
    for key, value in override_map.items():
        if value is not None:
            config[key] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"review_pairgrid_{args.profile}"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in run_dir.iterdir():
        if stale_path.is_file():
            stale_path.unlink()

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
    analysis_init_std = base_init_std * config["INIT_STD_SCALE"]
    analysis_init_scale = 0.0 if base_init_std == 0 else analysis_init_std / base_init_std
    pca_whiten = not args.no_pca_whiten

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
    generated_text = clean_output_text(tokenizer.decode(generated_token_id_list, skip_special_tokens=False))

    stable_scan = collect_position_scan(
        model=model,
        prompt_input_ids=preview_input_ids,
        reference_generated_token_ids=generated_token_ids.to(preview_input_ids.device),
        num_samples=config["SEARCH_NUM_SAMPLES"],
        num_steps=config["SEARCH_NUM_STEPS"],
        init_scale=analysis_init_scale,
        batch_size=min(config["SAMPLE_BATCH_SIZE"], 8),
        max_positions=min(24, len(generated_token_id_list)),
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
        max_results=config["UNSTABLE_MAX_RESULTS"],
    )
    if not unstable_findings:
        raise RuntimeError("No unstable generated token positions found under the current lightweight setting.")

    unstable_candidates = [normalize_variable_finding(item) for item in unstable_findings]
    unstable_analyses = []
    for info in unstable_candidates:
        unstable_analyses.append(
            analyze_position(
                model=model,
                tokenizer=tokenizer,
                prompt_input_ids=preview_input_ids,
                generated_token_ids=generated_token_id_list,
                info=info,
                num_samples=config["NUM_SAMPLES"],
                num_steps=config["NUM_STEPS"],
                init_scale=analysis_init_scale,
                batch_size=config["SAMPLE_BATCH_SIZE"],
                n_components=config["PAIRGRID_COMPONENTS"],
                min_group_size=config["PAIRGRID_MIN_GROUP_SIZE"],
                pca_whiten=pca_whiten,
            )
        )
    unstable_analysis = max(
        unstable_analyses,
        key=lambda item: item["pair_scores"][0]["score"] if item["pair_scores"] else -1.0,
    )

    stable_candidates = [entry for entry in stable_scan if entry["num_unique"] == 1]
    if stable_candidates:
        stable_info = min(
            stable_candidates,
            key=lambda entry: (abs(entry["position"] - unstable_analysis["info"]["position"]), -entry["position"]),
        )
    else:
        stable_info = max(stable_scan, key=lambda entry: entry["mode_fraction"])

    stable_analysis = analyze_position(
        model=model,
        tokenizer=tokenizer,
        prompt_input_ids=preview_input_ids,
        generated_token_ids=generated_token_id_list,
        info=stable_info,
        num_samples=config["NUM_SAMPLES"],
        num_steps=config["NUM_STEPS"],
        init_scale=analysis_init_scale,
        batch_size=config["SAMPLE_BATCH_SIZE"],
        n_components=config["PAIRGRID_COMPONENTS"],
        min_group_size=config["PAIRGRID_MIN_GROUP_SIZE"],
        pca_whiten=pca_whiten,
    )

    export_full_answer_heatmap(
        run_dir / "01_full_answer_residual_heatmap.pdf",
        generated_trace["step_residual_norm"].numpy(),
    )
    export_single_pairgrid_landscape(
        out_path=run_dir / "02_stable_pairgrid.pdf",
        analysis=stable_analysis,
        tokenizer=tokenizer,
        grid_size=config["PLOT_GRID_SIZE"],
        grid_pad=config["GRID_PAD"],
        smooth=config["RBF_SMOOTH"],
        level_count=config["CONTOUR_LEVEL_COUNT"],
        max_fit_points=2048,
        pairgrid_trajectories=config["PAIRGRID_TRAJECTORIES"],
        gallery_max_tokens=config["GALLERY_MAX_TOKENS"],
        overlay_strategy="round_robin",
        title_prefix="Stable output token",
        token_color_legend_title="Stable Token Colors",
    )
    export_single_pairgrid_landscape(
        out_path=run_dir / "03_unstable_pairgrid.pdf",
        analysis=unstable_analysis,
        tokenizer=tokenizer,
        grid_size=config["PLOT_GRID_SIZE"],
        grid_pad=config["GRID_PAD"],
        smooth=config["RBF_SMOOTH"],
        level_count=config["CONTOUR_LEVEL_COUNT"],
        max_fit_points=2048,
        pairgrid_trajectories=config["PAIRGRID_TRAJECTORIES"],
        gallery_max_tokens=config["GALLERY_MAX_TOKENS"],
        overlay_strategy="round_robin",
        title_prefix="Unstable output token",
        token_color_legend_title="Unstable Token Colors",
    )
    export_unstable_gallery(
        out_path=run_dir / "04_unstable_gallery_best_pair.pdf",
        unstable_analysis=unstable_analysis,
        tokenizer=tokenizer,
        grid_size=config["PLOT_GRID_SIZE"],
        grid_pad=config["GRID_PAD"],
        smooth=config["RBF_SMOOTH"],
        level_count=config["CONTOUR_LEVEL_COUNT"],
        max_fit_points=2048,
        gallery_max_tokens=config["GALLERY_MAX_TOKENS"],
        gallery_max_trajectories=config["GALLERY_MAX_TRAJECTORIES"],
    )
    write_summary(
        run_dir / "summary.json",
        profile=args.profile,
        generated_text=generated_text,
        stable_analysis=stable_analysis,
        unstable_analysis=unstable_analysis,
    )

    zip_name = run_name + ".zip"
    zip_path = args.output_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.iterdir()):
            zf.write(path, arcname=path.name)

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()
