#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


DEFAULT_SYSTEM_PROMPT = """You are Huginn, an AI assistant who embodies careful thought and deliberation.

Your responses demonstrate:
- Methodical reasoning, breaking complex problems into clear steps
- Mathematical and programming expertise grounded in fundamentals
- The ability to acknowledge uncertainty and correct course when needed
- Clear communication that illuminates rather than just informs
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-step residual traces for Huginn generation.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/data/hansen/serv12/HRM-Deq/models/huginn-0125"),
        help="Local path to the Huginn model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/residual_traces"),
        help="Directory for saved traces and figures.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device string.")
    parser.add_argument(
        "--example-source",
        choices=["repo_gsm8k_yaml", "hf_gsm8k_test", "manual"],
        default="repo_gsm8k_yaml",
        help="Where to load the evaluation instance from.",
    )
    parser.add_argument("--example-index", type=int, default=0, help="Index within the chosen source.")
    parser.add_argument("--prompt", type=str, default=None, help="Manual prompt when --example-source=manual.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for chat formatting.",
    )
    parser.add_argument("--num-steps", type=int, default=32, help="Recurrent steps per predicted token.")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Maximum new tokens to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_repo_gsm8k_yaml_example(repo_root: Path, index: int) -> dict[str, Any]:
    yaml_path = repo_root / "evaluate_raven" / "misc_benchmark_variants" / "gms8k_long_cot.yaml"
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    samples = data["fewshot_config"]["samples"]
    sample = samples[index]
    return {
        "task_name": data.get("task", "gsm8k_cot_long"),
        "split": "fewshot_config.samples",
        "index": index,
        "question": sample["question"],
        "target": sample["target"],
        "source_path": str(yaml_path),
    }


def load_hf_gsm8k_test_example(index: int) -> dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    sample = ds[index]
    return {
        "task_name": "gsm8k",
        "split": "test",
        "index": index,
        "question": sample["question"],
        "target": sample["answer"],
        "source_path": "hf://datasets/gsm8k/main/test",
    }


def load_example(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    if args.example_source == "repo_gsm8k_yaml":
        return load_repo_gsm8k_yaml_example(repo_root, args.example_index)
    if args.example_source == "hf_gsm8k_test":
        return load_hf_gsm8k_test_example(args.example_index)
    if not args.prompt:
        raise ValueError("--prompt is required when --example-source=manual")
    return {
        "task_name": "manual",
        "split": "manual",
        "index": args.example_index,
        "question": args.prompt,
        "target": None,
        "source_path": "manual",
    }


def build_chat_input(tokenizer, system_prompt: str, question: str) -> str:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def clean_text(text: str) -> str:
    for marker in ("<|end_turn|>", "<|end_text|>", "<|begin_text|>"):
        text = text.replace(marker, "")
    return text.strip()


def token_labels(tokenizer, token_ids: list[int]) -> list[str]:
    labels = []
    for token_id in token_ids:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        text = text.replace("\n", "\\n")
        if len(text) > 14:
            text = text[:11] + "..."
        labels.append(text)
    return labels


@torch.inference_mode()
def trace_generation(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    generation_config: GenerationConfig,
    num_steps: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    model_kwargs = {
        "use_cache": True,
        "past_key_values": DynamicCache(),
        "cache_position": torch.arange(input_ids.shape[1], device=input_ids.device),
    }
    stop_tokens = model._get_stops(generation_config, tokenizer, model_kwargs).to(input_ids.device)

    step_residuals = []
    step_residuals_ln = []
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

        current_latents = model.initialize_state(embedded_inputs)
        token_residuals = [current_latents[:, -1, :].detach().float().cpu()]
        token_residuals_ln = [model.transformer.ln_f(current_latents[:, -1:, :]).squeeze(1).detach().float().cpu()]

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
            token_residuals_ln.append(
                model.transformer.ln_f(current_latents[:, -1:, :]).squeeze(1).detach().float().cpu()
            )

        outputs = model.predict_from_latents(current_latents, **aux_inputs)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        next_token = model._sample_next_token(next_token_logits, generation_config)

        step_residuals.append(torch.cat(token_residuals, dim=0))
        step_residuals_ln.append(torch.cat(token_residuals_ln, dim=0))
        generated_token_ids.append(int(next_token[0, 0].item()))

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs)

        if next_token[0, 0].item() in stop_tokens.tolist():
            break

    residual_stream = torch.stack(step_residuals, dim=0)  # [positions, steps+1, hidden]
    residual_stream_ln = torch.stack(step_residuals_ln, dim=0)
    final_state = residual_stream[:, -1:, :]
    final_state_ln = residual_stream_ln[:, -1:, :]

    return {
        "all_input_ids": input_ids.detach().cpu(),
        "generated_token_ids": torch.tensor(generated_token_ids, dtype=torch.long),
        "residual_stream": residual_stream,
        "residual_stream_ln": residual_stream_ln,
        "residual_l2": torch.linalg.vector_norm(residual_stream, dim=-1),
        "residual_l2_ln": torch.linalg.vector_norm(residual_stream_ln, dim=-1),
        "delta_to_final_l2": torch.linalg.vector_norm(residual_stream - final_state, dim=-1),
        "delta_to_final_l2_ln": torch.linalg.vector_norm(residual_stream_ln - final_state_ln, dim=-1),
    }


def save_heatmap(matrix_steps_by_pos: np.ndarray, labels: list[str], title: str, out_path: Path, cmap: str) -> None:
    fig_width = max(8, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    image = ax.imshow(matrix_steps_by_pos, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xlabel("Generated token position")
    ax.set_ylabel("Recurrent step")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    if len(labels) <= 40:
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
    else:
        ax.set_xticklabels([str(i) for i in range(len(labels))], rotation=90, fontsize=8)
    ax.set_yticks(np.arange(matrix_steps_by_pos.shape[0]))
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    example = load_example(args, repo_root)
    run_slug = f"{example['task_name']}_{example['split'].replace('/', '_')}_{example['index']}_steps{args.num_steps}"
    run_dir = args.output_dir / run_slug
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

    prompt_text = build_chat_input(tokenizer, args.system_prompt, example["question"])
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
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

    traces = trace_generation(
        model=model,
        tokenizer=tokenizer,
        input_ids=prompt_ids,
        generation_config=generation_config,
        num_steps=args.num_steps,
        max_new_tokens=args.max_new_tokens,
    )

    generated_ids = traces["generated_token_ids"].tolist()
    generated_text = clean_text(tokenizer.decode(generated_ids, skip_special_tokens=False))
    labels = token_labels(tokenizer, generated_ids)

    residual_l2 = traces["residual_l2"].numpy().T
    delta_to_final_l2 = traces["delta_to_final_l2"].numpy().T

    save_heatmap(
        residual_l2,
        labels,
        title=f"Residual L2 Norm per Step ({example['task_name']} idx={example['index']})",
        out_path=run_dir / "residual_l2_heatmap.png",
        cmap="viridis",
    )
    save_heatmap(
        delta_to_final_l2,
        labels,
        title=f"Delta-to-final Residual L2 ({example['task_name']} idx={example['index']})",
        out_path=run_dir / "delta_to_final_l2_heatmap.png",
        cmap="magma",
    )

    torch.save(
        {
            "metadata": {
                "task_name": example["task_name"],
                "split": example["split"],
                "index": example["index"],
                "source_path": example["source_path"],
                "question": example["question"],
                "target": example["target"],
                "prompt_text": prompt_text,
                "generated_text": generated_text,
                "generated_token_labels": labels,
                "num_steps_recorded": args.num_steps + 1,
                "max_new_tokens_requested": args.max_new_tokens,
            },
            "prompt_ids": prompt_ids.detach().cpu(),
            **traces,
        },
        run_dir / "trace.pt",
    )

    with (run_dir / "metadata.json").open("w") as f:
        json.dump(
            {
                "task_name": example["task_name"],
                "split": example["split"],
                "index": example["index"],
                "source_path": example["source_path"],
                "question": example["question"],
                "target": example["target"],
                "generated_text": generated_text,
                "generated_token_ids": generated_ids,
                "generated_token_labels": labels,
                "num_generated_tokens": len(generated_ids),
                "num_steps_recorded": args.num_steps + 1,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved run to: {run_dir}")
    print(f"Question: {example['question']}")
    print(f"Target: {example['target']}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
