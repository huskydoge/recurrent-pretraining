#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer


DEFAULT_SYSTEM_PROMPT = """You are Huginn, an AI assistant who embodies careful thought and deliberation.

Your responses demonstrate:
- Methodical reasoning, breaking complex problems into clear steps
- Mathematical and programming expertise grounded in fundamentals
- The ability to acknowledge uncertainty and correct course when needed
- Clear communication that illuminates rather than just informs
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local inference for Huginn-0125.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/data/hansen/serv12/HRM-Deq/models/huginn-0125"),
        help="Local path to the downloaded Huginn-0125 model.",
    )
    parser.add_argument(
        "--device",
        default="cuda:7",
        help="Torch device string. Pick a GPU with enough free memory, e.g. cuda:6.",
    )
    parser.add_argument("--prompt", required=True, help="User prompt to send to the model.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used when applying the tokenizer chat template.",
    )
    parser.add_argument("--num-steps", type=int, default=32, help="Recurrent depth steps at inference time.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p used when sampling is enabled.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming and print the final completion only.",
    )
    return parser.parse_args()


def build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt.strip()})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def clean_output(text: str) -> str:
    for marker in ("<|end_turn|>", "<|end_text|>"):
        text = text.replace(marker, "")
    return text.strip()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    if not args.device.startswith("cuda"):
        raise ValueError(f"Expected a CUDA device, got: {args.device}")

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    )
    model.eval()

    rendered_prompt = build_prompt(tokenizer, args.system_prompt, args.prompt)
    inputs = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(args.device)

    do_sample = args.temperature > 0.0
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        stop_strings=["<|end_text|>", "<|end_turn|>"],
        do_sample=do_sample,
        temperature=None if not do_sample else args.temperature,
        top_p=None if not do_sample else args.top_p,
        top_k=None,
        min_p=None,
        return_dict_in_generate=True,
        eos_token_id=65505,
        bos_token_id=65504,
        pad_token_id=65509,
    )

    streamer = None if args.no_stream else TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            generation_config=generation_config,
            num_steps=args.num_steps,
            tokenizer=tokenizer,
            streamer=streamer,
        )

    if args.no_stream:
        new_tokens = outputs.sequences[0, input_ids.shape[1] :]
        print(clean_output(tokenizer.decode(new_tokens, skip_special_tokens=False)))


if __name__ == "__main__":
    main()
