from __future__ import annotations

from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers.cache_utils import DynamicCache

try:
    from scipy.interpolate import Rbf, griddata
except Exception:  # pragma: no cover
    Rbf = None
    griddata = None


DEFAULT_PALETTE = {
    "deep": "#4D2B8C",
    "mid": "#85409D",
    "accent": "#EEA727",
    "highlight": "#FFEF5F",
}

DEFAULT_TRAJECTORY_COLORS = ("#B0B0B0", "#C0C0C0", "#D0D0D0", "#A8A8A8")


def clean_output_text(text: str) -> str:
    for marker in ("<|end_turn|>", "<|end_text|>"):
        text = text.replace(marker, "")
    return text.strip()


def compact_labels(labels: Sequence[str], max_chars: int = 14) -> list[str]:
    compact: list[str] = []
    for label in labels:
        label = label.replace("\n", "\\n")
        if len(label) > max_chars:
            label = label[: max_chars - 3] + "..."
        compact.append(label)
    return compact


def decode_token_ids(tokenizer: Any, token_ids: Iterable[int], max_chars: int | None = None) -> list[str]:
    labels = []
    for token_id in token_ids:
        label = tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")
        if max_chars is not None and len(label) > max_chars:
            label = label[: max_chars - 3] + "..."
        labels.append(label)
    return labels


@torch.inference_mode()
def collect_next_token_trajectories(
    model: Any,
    input_ids: torch.Tensor,
    num_samples: int,
    num_steps: int,
    init_scale: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    embedded_inputs, block_idx = model.embed_inputs(input_ids)
    all_states = []
    all_residuals = []
    all_predicted_token_ids = []

    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        local_batch = stop - start
        batched_embeds = embedded_inputs.repeat(local_batch, 1, 1)
        current = model.initialize_state(batched_embeds, scale=init_scale)
        local_block_idx = block_idx.clone()

        states = [current[:, -1, :].detach().cpu().float()]
        residuals = []

        for step in range(num_steps):
            prev_last = current[:, -1, :].clone()
            current, local_block_idx, _ = model.iterate_one_step(
                batched_embeds,
                current,
                block_idx=local_block_idx,
                current_step=step,
            )
            delta_last = current[:, -1, :] - prev_last
            states.append(current[:, -1, :].detach().cpu().float())
            residuals.append(torch.linalg.vector_norm(delta_last.float(), dim=-1).cpu())

        logits = model.predict_from_latents(current).logits[:, -1, :]
        all_states.append(torch.stack(states, dim=1))
        all_residuals.append(torch.stack(residuals, dim=1))
        all_predicted_token_ids.append(logits.argmax(dim=-1).cpu())

    return {
        "states": torch.cat(all_states, dim=0),
        "residual_norms": torch.cat(all_residuals, dim=0),
        "predicted_token_ids": torch.cat(all_predicted_token_ids, dim=0),
    }


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1) + 0.5 * torch.sum(
        q * (torch.log(q) - torch.log(m)),
        dim=-1,
    )


@torch.inference_mode()
def collect_coda_token_distribution_trace(
    model: Any,
    input_ids: torch.Tensor,
    num_steps: int,
    init_scale: float = 0.0,
    attention_mask: torch.Tensor | None = None,
    include_step_zero: bool = True,
    position: int = -1,
) -> dict[str, torch.Tensor]:
    if input_ids.shape[0] != 1:
        raise ValueError("collect_coda_token_distribution_trace only supports batch size 1.")

    embedded_inputs, block_idx = model.embed_inputs(input_ids, attention_mask=attention_mask)
    current = model.initialize_state(embedded_inputs, scale=init_scale)

    probs_by_step = []
    entropy_by_step = []
    argmax_ids = []
    argmax_probs = []

    position_index = position if position >= 0 else embedded_inputs.shape[1] + position
    if position_index < 0 or position_index >= embedded_inputs.shape[1]:
        raise IndexError(f"position={position} is out of range for sequence length {embedded_inputs.shape[1]}.")

    def record_distribution(latents: torch.Tensor) -> None:
        outputs = model.predict_from_latents(latents, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits[:, position_index, :].float(), dim=-1).cpu()
        entropy = torch.where(probs > 0, -probs * probs.log(), 0).sum(dim=-1)
        top_prob, top_id = probs.max(dim=-1)

        probs_by_step.append(probs.squeeze(0))
        entropy_by_step.append(entropy.squeeze(0))
        argmax_ids.append(top_id.squeeze(0))
        argmax_probs.append(top_prob.squeeze(0))

    if include_step_zero:
        record_distribution(current)

    for step in range(num_steps):
        current, block_idx, _ = model.iterate_one_step(
            embedded_inputs,
            current,
            block_idx=block_idx,
            attention_mask=attention_mask,
            current_step=step,
        )
        record_distribution(current)

    probs = torch.stack(probs_by_step, dim=0)
    entropy = torch.stack(entropy_by_step, dim=0)
    argmax_token_ids = torch.stack(argmax_ids, dim=0).to(dtype=torch.long)
    argmax_prob = torch.stack(argmax_probs, dim=0)
    steps = torch.arange(probs.shape[0], dtype=torch.long)
    if not include_step_zero:
        steps = steps + 1

    l1_delta_from_prev = torch.zeros(probs.shape[0], dtype=probs.dtype)
    js_delta_from_prev = torch.zeros(probs.shape[0], dtype=probs.dtype)
    if probs.shape[0] > 1:
        l1_delta_from_prev[1:] = (probs[1:] - probs[:-1]).abs().sum(dim=-1)
        js_delta_from_prev[1:] = _js_divergence(probs[1:], probs[:-1])

    js_to_final = _js_divergence(probs, probs[-1:].expand_as(probs))

    return {
        "steps": steps,
        "position_index": torch.as_tensor(position_index, dtype=torch.long),
        "probs": probs,
        "entropy": entropy,
        "argmax_token_ids": argmax_token_ids,
        "argmax_prob": argmax_prob,
        "l1_delta_from_prev": l1_delta_from_prev,
        "js_delta_from_prev": js_delta_from_prev,
        "js_to_final": js_to_final,
    }


@torch.inference_mode()
def sample_next_token_predictions(
    model: Any,
    input_ids: torch.Tensor,
    num_samples: int,
    num_steps: int,
    init_scale: float,
    batch_size: int,
) -> torch.Tensor:
    trace = collect_next_token_trajectories(
        model=model,
        input_ids=input_ids,
        num_samples=num_samples,
        num_steps=num_steps,
        init_scale=init_scale,
        batch_size=batch_size,
    )
    return trace["predicted_token_ids"]


def select_salient_token_ids(
    probs_by_step: torch.Tensor,
    top_k_per_step: int = 8,
    max_tokens: int = 18,
    always_include: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    if probs_by_step.ndim != 2:
        raise ValueError(f"Expected probs_by_step to have shape [steps, vocab], got {tuple(probs_by_step.shape)}.")

    k = min(top_k_per_step, probs_by_step.shape[-1])
    top_ids = torch.topk(probs_by_step, k=k, dim=-1).indices.reshape(-1)
    candidate_ids = torch.unique(top_ids, sorted=False)

    if always_include is not None:
        if isinstance(always_include, torch.Tensor):
            include_ids = always_include.to(dtype=torch.long).reshape(-1)
        else:
            include_ids = torch.as_tensor(list(always_include), dtype=torch.long)
        candidate_ids = torch.unique(torch.cat([candidate_ids, include_ids]), sorted=False)

    candidate_scores = probs_by_step.index_select(1, candidate_ids).max(dim=0).values
    order = torch.argsort(candidate_scores, descending=True)
    candidate_ids = candidate_ids[order]

    if candidate_ids.numel() > max_tokens:
        candidate_ids = candidate_ids[:max_tokens]

    return candidate_ids.to(dtype=torch.long)


@torch.inference_mode()
def find_variable_generated_positions(
    model: Any,
    prompt_input_ids: torch.Tensor,
    reference_generated_token_ids: torch.Tensor | Sequence[int],
    num_samples: int,
    num_steps: int,
    init_scale: float,
    batch_size: int,
    max_positions: int | None = None,
    min_unique_tokens: int = 2,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    if prompt_input_ids.shape[0] != 1:
        raise ValueError("find_variable_generated_positions only supports batch size 1.")

    if isinstance(reference_generated_token_ids, torch.Tensor):
        reference = reference_generated_token_ids.to(device=prompt_input_ids.device, dtype=torch.long).flatten()
    else:
        reference = torch.tensor(list(reference_generated_token_ids), device=prompt_input_ids.device, dtype=torch.long)

    positions_to_scan = len(reference) if max_positions is None else min(len(reference), max_positions)
    findings: list[dict[str, Any]] = []

    for position in range(positions_to_scan):
        if position == 0:
            prefix_ids = prompt_input_ids
        else:
            prefix_ids = torch.cat([prompt_input_ids, reference[:position].unsqueeze(0)], dim=-1)

        predicted_token_ids = sample_next_token_predictions(
            model=model,
            input_ids=prefix_ids,
            num_samples=num_samples,
            num_steps=num_steps,
            init_scale=init_scale,
            batch_size=batch_size,
        )
        unique_token_ids, counts = torch.unique(predicted_token_ids, sorted=False, return_counts=True)
        order = torch.argsort(counts, descending=True)
        unique_token_ids = unique_token_ids[order]
        counts = counts[order]

        if unique_token_ids.numel() < min_unique_tokens:
            continue

        findings.append(
            {
                "position": position,
                "reference_token_id": int(reference[position].item()),
                "predicted_token_ids": predicted_token_ids,
                "unique_token_ids": unique_token_ids,
                "counts": counts,
                "mode_fraction": float(counts[0].item() / predicted_token_ids.numel()),
            }
        )

        if max_results is not None and len(findings) >= max_results:
            break

    return findings


@torch.inference_mode()
def collect_generated_token_traces(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    generation_config: Any,
    num_steps: int,
    max_new_tokens: int,
    init_scale: float = 1.0,
) -> dict[str, Any]:
    model_kwargs = {
        "use_cache": True,
        "past_key_values": DynamicCache(),
        "cache_position": torch.arange(input_ids.shape[1], device=input_ids.device),
    }
    stop_tokens = model._get_stops(generation_config, tokenizer, model_kwargs).to(input_ids.device)

    step_residuals = []
    generated_token_ids: list[int] = []

    for _ in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        aux_inputs = {
            key: model_inputs[key]
            for key in ("cache_position", "past_key_values", "attention_mask")
            if key in model_inputs
        }
        embedded_inputs, block_idx = model.embed_inputs(
            model_inputs["input_ids"],
            attention_mask=model_inputs.get("attention_mask"),
            past_key_values=model_inputs.get("past_key_values"),
            use_cache=model_kwargs.get("use_cache", True),
            cache_position=model_inputs.get("cache_position"),
        )

        current_latents = model.initialize_state(embedded_inputs, scale=init_scale)
        token_residuals = [current_latents[:, -1, :].detach().float().cpu()]

        for step_idx in range(num_steps):
            current_latents, block_idx, _ = model.iterate_one_step(
                embedded_inputs,
                current_latents,
                block_idx=block_idx,
                attention_mask=model_inputs.get("attention_mask"),
                past_key_values=model_inputs.get("past_key_values"),
                cache_position=model_inputs.get("cache_position"),
                current_step=step_idx,
            )
            token_residuals.append(current_latents[:, -1, :].detach().float().cpu())

        outputs = model.predict_from_latents(current_latents, **aux_inputs)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        next_token = model._sample_next_token(next_token_logits, generation_config)

        step_residuals.append(torch.cat(token_residuals, dim=0))
        generated_token_ids.append(int(next_token[0, 0].item()))

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs)

        if next_token[0, 0].item() in stop_tokens.tolist():
            break

    residual_stream = torch.stack(step_residuals, dim=0)  # [generated_position, steps+1, hidden]
    step_residual_norm = torch.linalg.vector_norm(residual_stream[:, 1:, :] - residual_stream[:, :-1, :], dim=-1)

    return {
        "all_input_ids": input_ids.detach().cpu(),
        "generated_token_ids": torch.tensor(generated_token_ids, dtype=torch.long),
        "residual_stream": residual_stream,
        "step_residual_norm": step_residual_norm,
    }


@torch.inference_mode()
def collect_position_trajectories(
    model: Any,
    embedded_inputs: torch.Tensor,
    block_idx: torch.Tensor,
    positions: Sequence[int],
    num_samples: int,
    num_steps: int,
    init_scale: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    all_states = []
    all_selected_residuals = []
    all_position_residuals = []
    all_predicted_token_ids = []

    positions_tensor = torch.as_tensor(list(positions), dtype=torch.long)

    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        local_batch = stop - start
        batched_embeds = embedded_inputs.repeat(local_batch, 1, 1)
        current = model.initialize_state(batched_embeds, scale=init_scale)
        local_block_idx = block_idx.clone()

        states = [current[:, positions, :].detach().cpu().float()]
        selected_residuals = []
        position_residuals = []

        for step in range(num_steps):
            prev = current.clone()
            current, local_block_idx, _ = model.iterate_one_step(
                batched_embeds,
                current,
                block_idx=local_block_idx,
                current_step=step,
            )
            delta = current - prev
            states.append(current[:, positions, :].detach().cpu().float())
            selected_residuals.append(torch.linalg.vector_norm(delta[:, positions, :].float(), dim=-1).cpu())
            position_residuals.append(torch.linalg.vector_norm(delta.float(), dim=-1).cpu())

        logits = model.predict_from_latents(current).logits
        predicted_token_ids = logits.argmax(dim=-1).index_select(dim=1, index=positions_tensor.to(logits.device)).cpu()

        all_states.append(torch.stack(states, dim=1))
        all_selected_residuals.append(torch.stack(selected_residuals, dim=1))
        all_position_residuals.append(torch.stack(position_residuals, dim=1))
        all_predicted_token_ids.append(predicted_token_ids)

    return {
        "states": torch.cat(all_states, dim=0),
        "residual_norms": torch.cat(all_selected_residuals, dim=0),
        "all_position_residual_norms": torch.cat(all_position_residuals, dim=0),
        "predicted_token_ids": torch.cat(all_predicted_token_ids, dim=0),
    }


def project_states_with_pca(states: torch.Tensor, whiten_coords: bool = True) -> list[dict[str, torch.Tensor]]:
    projections = []
    for position_index in range(states.shape[2]):
        position_states = states[:, :, position_index, :]
        flat = position_states.reshape(-1, position_states.shape[-1])
        center = flat.mean(dim=0)
        centered = flat - center
        _, _, v = torch.pca_lowrank(centered, q=2)
        coords = centered @ v[:, :2]
        coord_scale = coords.std(dim=0).clamp_min(1e-6)
        if whiten_coords:
            coords = coords / coord_scale
        projections.append(
            {
                "coords": coords.reshape(position_states.shape[0], position_states.shape[1], 2),
                "center": center,
                "basis": v[:, :2],
                "coord_scale": coord_scale,
            }
        )
    return projections


def _padded_range_1d(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    value_min, value_max = float(np.min(values)), float(np.max(values))
    center = (value_min + value_max) / 2.0
    half = (value_max - value_min) / 2.0
    if half <= 0:
        half = 1.0
    half *= 1.0 + pad
    return center - half, center + half


def axis_ranges(xs: np.ndarray, ys: np.ndarray, pad: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    return _padded_range_1d(xs, pad=pad), _padded_range_1d(ys, pad=pad)


def square_ranges(xs: np.ndarray, ys: np.ndarray, pad: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    half = max(x_max - x_min, y_max - y_min) / 2.0
    if half <= 0:
        half = 1.0
    half *= 1.0 + pad
    return (x_center - half, x_center + half), (y_center - half, y_center + half)


def _idw_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    neighbors: int = 24,
    power: float = 2.0,
    eps: float = 1e-8,
    chunk_size: int = 2048,
) -> np.ndarray:
    points = points.astype(np.float32)
    values = values.astype(np.float32)
    targets = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    k = min(neighbors, points.shape[0])
    out = np.empty(targets.shape[0], dtype=np.float32)

    for start in range(0, targets.shape[0], chunk_size):
        stop = min(start + chunk_size, targets.shape[0])
        chunk = targets[start:stop]
        diff = chunk[:, None, :] - points[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        nn_idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        nn_dist2 = np.take_along_axis(dist2, nn_idx, axis=1)
        nn_values = values[nn_idx]

        exact_mask = nn_dist2 <= eps
        weights = 1.0 / np.maximum(nn_dist2, eps) ** (power / 2.0)
        chunk_out = np.sum(weights * nn_values, axis=1) / np.sum(weights, axis=1)

        has_exact = exact_mask.any(axis=1)
        if has_exact.any():
            exact_rows = np.where(has_exact)[0]
            exact_cols = exact_mask[has_exact].argmax(axis=1)
            chunk_out[exact_rows] = nn_values[has_exact, exact_cols]

        out[start:stop] = chunk_out

    return out.reshape(xx.shape)


def make_metric_grid_square(
    points: np.ndarray,
    values: np.ndarray,
    grid_size: int,
    pad: float = 0.08,
    smooth: float = 0.15,
    value_min: float | None = None,
    value_max: float | None = None,
    bounds_points: np.ndarray | None = None,
    idw_neighbors: int = 24,
    idw_power: float = 2.0,
    square: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    range_points = points if bounds_points is None else bounds_points
    if square:
        x_range, y_range = square_ranges(range_points[:, 0], range_points[:, 1], pad=pad)
    else:
        x_range, y_range = axis_ranges(range_points[:, 0], range_points[:, 1], pad=pad)
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    x = points[:, 0]
    y = points[:, 1]
    grid_z = None
    if Rbf is not None:
        try:
            rbf = Rbf(x, y, values, function="multiquadric", smooth=smooth)
            grid_z = rbf(xx, yy)
        except Exception:
            grid_z = None

    if grid_z is None and griddata is not None:
        try:
            grid_z = griddata((x, y), values, (xx, yy), method="linear")
            if np.isnan(grid_z).any():
                grid_z = griddata((x, y), values, (xx, yy), method="nearest")
        except Exception:
            grid_z = None

    if grid_z is None:
        grid_z = _idw_interpolate(points, values, xx, yy, neighbors=idw_neighbors, power=idw_power)

    if value_min is not None and value_max is not None:
        grid_z = np.clip(grid_z, value_min, value_max)
    return xs, ys, grid_z


def make_levels(value_min: float, value_max: float, level_count: int = 16) -> np.ndarray:
    if value_max <= value_min:
        value_max = value_min + 1e-3
    return np.linspace(value_min, value_max, level_count + 1)


def plot_residual_heatmap(
    residual_by_position_and_step: np.ndarray,
    token_labels: Sequence[str],
    title: str,
    cmap: str = "magma",
    max_label_chars: int = 14,
    all_labels_threshold: int = 40,
) -> tuple[plt.Figure, plt.Axes]:
    heatmap = residual_by_position_and_step.T
    display_labels = compact_labels(token_labels, max_chars=max_label_chars)
    fig_width = max(10.0, len(display_labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    im = ax.imshow(heatmap, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Iteration t")
    ax.set_xticks(np.arange(len(display_labels)))
    if len(display_labels) <= all_labels_threshold:
        ax.set_xticklabels(display_labels, rotation=90, fontsize=8)
    else:
        ax.set_xticklabels([str(i) for i in range(len(display_labels))], rotation=90, fontsize=8)
    ax.set_yticks(np.arange(heatmap.shape[0]))
    ax.set_yticklabels(np.arange(1, heatmap.shape[0] + 1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("L2 norm of f_t - f_{t-1}")
    fig.tight_layout()
    return fig, ax


def plot_coda_distribution_heatmap(
    trace: dict[str, torch.Tensor],
    tokenizer: Any,
    title: str,
    selected_token_ids: Sequence[int] | torch.Tensor | None = None,
    top_k_per_step: int = 8,
    max_tokens: int = 18,
    include_other_mass: bool = True,
    cmap: str = "magma",
    max_label_chars: int = 24,
) -> tuple[plt.Figure, plt.Axes]:
    probs = trace["probs"].float().cpu()
    steps = trace["steps"].cpu().numpy()

    if selected_token_ids is None:
        selected_ids = select_salient_token_ids(
            probs,
            top_k_per_step=top_k_per_step,
            max_tokens=max_tokens,
            always_include=trace["argmax_token_ids"],
        )
    elif isinstance(selected_token_ids, torch.Tensor):
        selected_ids = selected_token_ids.to(dtype=torch.long).cpu().reshape(-1)
    else:
        selected_ids = torch.as_tensor(list(selected_token_ids), dtype=torch.long)

    token_probs = probs.index_select(1, selected_ids)
    token_labels = decode_token_ids(tokenizer, selected_ids.tolist(), max_chars=max_label_chars)

    if include_other_mass:
        other_mass = (1.0 - token_probs.sum(dim=1)).clamp_min(0.0).unsqueeze(1)
        token_probs = torch.cat([token_probs, other_mass], dim=1)
        token_labels.append("<other>")

    heatmap = token_probs.T.numpy()
    fig_width = max(10.0, len(steps) * 0.65)
    fig_height = max(5.5, 0.42 * len(token_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)

    vmax = max(0.20, float(heatmap.max()))
    im = ax.imshow(heatmap, aspect="auto", origin="upper", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Loop t")
    ax.set_ylabel("Token")
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_yticks(np.arange(len(token_labels)))
    ax.set_yticklabels(token_labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Probability after coda + lm_head")

    fig.tight_layout()
    return fig, ax


def draw_landscape_with_trajectories(
    ax: plt.Axes,
    coords: np.ndarray,
    residuals: np.ndarray,
    title: str | None = None,
    grid_size: int = 140,
    grid_pad: float = 0.08,
    smooth: float = 0.15,
    level_count: int = 16,
    cmap: str = "viridis",
    hide_axes: bool = True,
    trajectory_sample_indices: Sequence[int] | None = None,
    max_trajectories_to_overlay: int = 18,
    palette: dict[str, str] | None = None,
    trajectory_colors: Sequence[str] = DEFAULT_TRAJECTORY_COLORS,
) -> Any:
    palette = DEFAULT_PALETTE if palette is None else palette
    metric_points = coords[:, 1:, :].reshape(-1, 2)
    bounds_points = coords.reshape(-1, 2)
    values = residuals.reshape(-1)

    value_min = float(values.min())
    value_max = float(values.max())
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
    )
    x_range, y_range = square_ranges(bounds_points[:, 0], bounds_points[:, 1], pad=grid_pad)

    contour = ax.contourf(xs, ys, grid_z, levels=levels, cmap=cmap)
    ax.contour(xs, ys, grid_z, levels=levels, colors="white", linewidths=1.0)

    if trajectory_sample_indices is None:
        overlay_indices = list(range(min(max_trajectories_to_overlay, coords.shape[0])))
    else:
        overlay_indices = list(trajectory_sample_indices)

    for overlay_idx, sample_idx in enumerate(overlay_indices):
        traj = coords[sample_idx]
        color = trajectory_colors[overlay_idx % len(trajectory_colors)]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.6, alpha=0.72)
        ax.scatter(
            [traj[0, 0]],
            [traj[0, 1]],
            s=120,
            c=palette["highlight"],
            edgecolors="#111111",
            linewidths=1.3,
            marker="^",
            zorder=4,
        )
        ax.scatter(
            [traj[-1, 0]],
            [traj[-1, 1]],
            s=120,
            c=palette["accent"],
            edgecolors="#111111",
            linewidths=1.3,
            marker="D",
            zorder=4,
        )

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal", adjustable="box")
    if hide_axes:
        ax.set_axis_off()
    else:
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
    if title:
        ax.set_title(title)
    return contour


def plot_landscape_with_trajectories(
    coords: np.ndarray,
    residuals: np.ndarray,
    title: str | None = None,
    grid_size: int = 140,
    grid_pad: float = 0.08,
    smooth: float = 0.15,
    level_count: int = 16,
    cmap: str = "viridis",
    hide_axes: bool = True,
    show_colorbar: bool = True,
    trajectory_sample_indices: Sequence[int] | None = None,
    max_trajectories_to_overlay: int = 18,
    palette: dict[str, str] | None = None,
    trajectory_colors: Sequence[str] = DEFAULT_TRAJECTORY_COLORS,
    footer_text: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    contour = draw_landscape_with_trajectories(
        ax,
        coords,
        residuals,
        title=title if not hide_axes else None,
        grid_size=grid_size,
        grid_pad=grid_pad,
        smooth=smooth,
        level_count=level_count,
        cmap=cmap,
        hide_axes=hide_axes,
        trajectory_sample_indices=trajectory_sample_indices,
        max_trajectories_to_overlay=max_trajectories_to_overlay,
        palette=palette,
        trajectory_colors=trajectory_colors,
    )

    if hide_axes:
        bottom = 0.06 if footer_text else 0.0
        fig.subplots_adjust(left=0, right=1, bottom=bottom, top=1)
    else:
        if title:
            ax.set_title(title)
        fig.tight_layout()

    if show_colorbar:
        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("||f_t - f_{t-1}||_2")

    if title and hide_axes:
        fig.suptitle(title, y=0.995, fontsize=11)
    if footer_text:
        fig.text(0.5, 0.02, footer_text, ha="center", va="bottom", fontsize=10, family="monospace")

    fig.patch.set_facecolor("white")
    return fig, ax
