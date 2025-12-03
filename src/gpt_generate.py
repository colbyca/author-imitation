"""Generate text using GPT-2 models (fine-tuned or raw).

This script can load either:
- A fine-tuned GPT-2 model from a directory (e.g., G0 trained by "src.g0_train")
- A raw (untrained) GPT-2 model by name (e.g., gpt2-large)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, PreTrainedTokenizerFast, set_seed

from config import CONFIG
from utils import (
    collect_texts,
    ensure_split,
    GeneratedSample,
    build_word_prompts,
    generate_sequences,
    load_dataset_dict,
    load_generator,
    select_device,
    setup_logging,
)

LOGGER = logging.getLogger("gpt_generate")


def parse_args() -> argparse.Namespace:
    cfg = CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-dir",
        type=Path,
        help="Path to the fine-tuned GPT-2 model directory.",
    )
    model_group.add_argument(
        "--model-name",
        type=str,
        help="Name of a pre-trained GPT-2 model (e.g., 'gpt2-large', 'gpt2', 'gpt2-medium').",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        action="append",
        help="Prompt text to seed generation. Can be provided multiple times.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="File containing prompts. Each non-empty line becomes a prompt.",
    )
    parser.add_argument(
        "--auto-prompt",
        action="store_true",
        help="Automatically generate prompts from target author texts in the discriminator dataset.",
    )
    parser.add_argument(
        "--num-auto-prompts",
        type=int,
        default=1,
        help="Number of prompts to automatically generate when --auto-prompt is used.",
    )
    parser.add_argument(
        "--prompt-num-words",
        type=int,
        default=7,
        help="Number of words to extract from the beginning of texts for prompts (default: 7, range: 5-10).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=cfg.generation.max_new_tokens,
        help="Maximum number of tokens to generate beyond each prompt.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=cfg.generation.min_new_tokens,
        help="Minimum number of tokens to generate beyond each prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=cfg.generation.temperature,
        help="Sampling temperature (ignored when --greedy is set).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=cfg.generation.top_k,
        help="Top-k sampling threshold.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=cfg.generation.top_p,
        help="Top-p (nucleus) sampling threshold.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=cfg.generation.repetition_penalty,
        help="Penalty for repeated tokens.",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=cfg.generation.num_return_sequences,
        help="Number of completions to sample per prompt.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=cfg.generation.num_beams,
        help="Beam search width (set >1 to enable beam search).",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=cfg.generation.no_repeat_ngram_size,
        help="Block repetition of n-grams of this size (0 to disable).",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=cfg.generation.length_penalty,
        help="Length penalty for beam search.",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=cfg.generation.early_stopping,
        help="Stop beam search when at least num_beams sentences are finished.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Disable sampling and use greedy decoding (num_return_sequences must be 1).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run generation on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.random_seed,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional file to save generations to.",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "jsonl"],
        default="jsonl",
        help="Format to use when writing outputs to --output-file.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of overwriting it.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the active generation configuration before sampling.",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_dir: Optional[Path],
    model_name: Optional[str],
) -> tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    """Load GPT-2 model and tokenizer from either a directory or model name."""
    start_time = time.time()
    if model_dir is not None:
        LOGGER.info("Loading fine-tuned model from directory: %s", model_dir)
        model, tokenizer = load_generator(model_dir)
    elif model_name is not None:
        LOGGER.info("Loading raw GPT-2 model: %s", model_name)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        raise ValueError("Either --model-dir or --model-name must be provided.")
    
    load_time = time.time() - start_time
    LOGGER.info("Model loading completed in %.2f seconds", load_time)
    return model, tokenizer


def load_all_author_texts() -> Sequence[str]:
    """Load texts from all authors (target and distractors) from the discriminator dataset."""
    dataset = load_dataset_dict(CONFIG.paths.discriminator_dataset_dir)
    train_split = ensure_split(dataset, "train")
    return collect_texts(train_split)


def collect_prompts(args: argparse.Namespace, rng: Optional[np.random.Generator] = None) -> List[str]:
    prompts: List[str] = []

    if args.prompt:
        prompts.extend(args.prompt)

    if args.prompt_file:
        if not args.prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        file_prompts = [
            line.rstrip("\n")
            for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        ]
        prompts.extend(file_prompts)

    if not prompts and not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
        if stdin_text:
            prompts = [line.rstrip("\n") for line in stdin_text.splitlines()]

    if args.auto_prompt:
        if rng is None:
            rng = np.random.default_rng(args.seed)
        all_texts = load_all_author_texts()
        num_words = max(5, min(10, args.prompt_num_words))
        auto_prompts = build_word_prompts(all_texts, args.num_auto_prompts, num_words, rng)
        prompts.extend(auto_prompts)
        LOGGER.info("Generated %d automatic prompts from all author texts (first %d words).", len(auto_prompts), num_words)

    if not prompts:
        LOGGER.info("No prompts supplied; defaulting to empty prompt for unconditional generation.")
        prompts = [""]

    return prompts


def build_generation_kwargs(args: argparse.Namespace, tokenizer: PreTrainedTokenizerFast) -> dict:
    if args.greedy and args.num_return_sequences != 1:
        raise ValueError("Greedy decoding only supports num_return_sequences=1.")

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "do_sample": not args.greedy,
        "num_beams": args.num_beams,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "length_penalty": args.length_penalty,
        "early_stopping": args.early_stopping,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if args.min_new_tokens > 0:
        generation_kwargs["min_new_tokens"] = args.min_new_tokens

    # Avoid conflicting sampling parameters when using beam search.
    if args.num_beams > 1:
        generation_kwargs["do_sample"] = not args.greedy
        if generation_kwargs["do_sample"]:
            generation_kwargs.setdefault("temperature", args.temperature)
        else:
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_k", None)
            generation_kwargs.pop("top_p", None)

    if generation_kwargs["do_sample"]:
        generation_kwargs.setdefault("temperature", args.temperature)
    else:
        generation_kwargs.pop("temperature", None)
        generation_kwargs.pop("top_k", None)
        generation_kwargs.pop("top_p", None)

    return generation_kwargs


def group_generations(
    prompts: Sequence[str],
    samples: Sequence[GeneratedSample],
    num_return_sequences: int,
) -> List[dict]:
    """Organise flat generated samples into prompt-grouped structures."""

    results: List[dict] = []
    sample_index = 0
    for prompt in prompts:
        prompt_entries: List[dict] = []
        for _ in range(num_return_sequences):
            if sample_index >= len(samples):
                break
            sample = samples[sample_index]
            prompt_entries.append(
                {
                    "prompt_text": sample.prompt,
                    "completion": sample.completion,
                    "full_text": sample.full_text,
                }
            )
            sample_index += 1
        if not prompt_entries:
            prompt_entries.append({"prompt_text": "", "completion": "", "full_text": ""})
        results.append({"prompt": prompt, "samples": prompt_entries})
    return results


def save_results(results: List[dict], path: Path, format_: str, append: bool, generation_kwargs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    if format_ == "text":
        with path.open(mode, encoding="utf-8") as handle:
            for idx, entry in enumerate(results, start=1):
                header = f"=== Prompt {idx} ===\n"
                handle.write(header)
                handle.write(entry["prompt"] + "\n")
                for sample_idx, sample in enumerate(entry["samples"], start=1):
                    prefix = (
                        f"--- Completion {sample_idx} ---\n"
                        if len(entry["samples"]) > 1
                        else "--- Completion ---\n"
                    )
                    handle.write(prefix)
                    handle.write(sample["completion"].strip() + "\n")
                handle.write("\n")
    else:
        with path.open(mode, encoding="utf-8") as handle:
            for entry in results:
                record = {
                    "prompt": entry["prompt"],
                    "samples": entry["samples"],
                    "generation_kwargs": generation_kwargs,
                }
                handle.write(json.dumps(record) + "\n")

    LOGGER.info("Saved generations to %s", path)


def main() -> None:
    args = parse_args()
    setup_logging(args.logging_level)
    if args.show_config:
        LOGGER.info("Active generation config: %s", json.dumps(asdict(CONFIG.generation), indent=2))

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    prompts = collect_prompts(args, rng)
    device = select_device(args.device)
    LOGGER.info("Using device: %s", device)

    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.model_name)
    
    model = model.to(device)

    generation_kwargs = build_generation_kwargs(args, tokenizer)
    LOGGER.debug("Generation kwargs: %s", generation_kwargs)

    generation_start = time.time()
    samples = generate_sequences(
        model,
        tokenizer,
        prompts,
        device=device,
        generation_kwargs=generation_kwargs,
        logger=LOGGER,
    )
    generation_time = time.time() - generation_start
    
    num_prompts = len(prompts)
    total_generations = len(samples)
    max_new_tokens = generation_kwargs.get("max_new_tokens", 0)
    
    total_tokens_approx = total_generations * max_new_tokens
    tokens_per_second = total_tokens_approx / generation_time if generation_time > 0 else 0
    time_per_prompt = generation_time / num_prompts if num_prompts > 0 else 0
    time_per_generation = generation_time / total_generations if total_generations > 0 else 0
    
    # Log timing results
    LOGGER.info("=" * 60)
    LOGGER.info("Generation timing results:")
    LOGGER.info("  - Total generation time: %.2f seconds", generation_time)
    LOGGER.info("  - Time per prompt: %.3f seconds", time_per_prompt)
    LOGGER.info("  - Time per generation: %.3f seconds", time_per_generation)
    LOGGER.info("  - Approximate tokens per second: %.1f", tokens_per_second)
    LOGGER.info("  - Total prompts processed: %d", num_prompts)
    LOGGER.info("  - Total generations: %d", total_generations)
    LOGGER.info("=" * 60)

    results = group_generations(
        prompts,
        samples,
        int(generation_kwargs.get("num_return_sequences", 1)),
    )

    if args.output_file:
        save_results(results, args.output_file, args.output_format, args.append, generation_kwargs)


if __name__ == "__main__":
    main()

