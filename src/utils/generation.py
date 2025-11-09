"""Utilities for building prompts and sampling text from generator models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@dataclass
class GeneratedSample:
    prompt: str
    completion: str
    full_text: str


def ensure_tokenizer_padding(tokenizer: GPT2TokenizerFast) -> GPT2TokenizerFast:
    """Ensure the tokenizer has a pad token defined (falling back to EOS)."""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_generator(model_dir: Union[str, Path]) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    """Load a GPT-2 generator checkpoint alongside its tokenizer."""

    tokenizer = GPT2TokenizerFast.from_pretrained(str(model_dir))
    ensure_tokenizer_padding(tokenizer)

    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def build_word_prompts(
    texts: Sequence[str],
    num_prompts: int,
    num_words: int,
    rng: np.random.Generator,
) -> List[str]:
    """Create prompts by sampling texts and taking the first ``num_words`` words."""

    if num_prompts <= 0:
        return []
    if not texts:
        return [""] * num_prompts

    indices = rng.choice(len(texts), size=num_prompts, replace=True)
    prompts: List[str] = []
    for idx in indices:
        sample = texts[int(idx)].replace("\n", " ").strip()
        if not sample:
            prompts.append("")
            continue
        words = sample.split()
        prompts.append(" ".join(words[:num_words]) if words else "")
    return prompts


def build_char_prompts(
    texts: Sequence[str],
    num_prompts: int,
    max_chars: int,
    rng: np.random.Generator,
) -> List[str]:
    """Create prompts by sampling texts and truncating to ``max_chars``."""

    if num_prompts <= 0:
        return []
    if max_chars <= 0:
        return ["" for _ in range(num_prompts)]
    if not texts:
        return [""] * num_prompts

    indices = rng.choice(len(texts), size=num_prompts, replace=True)
    prompts: List[str] = []
    for idx in indices:
        sample = texts[int(idx)].replace("\n", " ").strip()
        prompts.append(sample[:max_chars].strip())
    return prompts


def generate_sequences(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompts: Sequence[str],
    *,
    device: torch.device,
    generation_kwargs: Optional[dict] = None,
    batch_size: Optional[int] = 2,
    logger: Optional[logging.Logger] = None,
) -> List[GeneratedSample]:
    """Generate continuations for prompts using a GPT-2 model."""

    if not prompts:
        return []

    ensure_tokenizer_padding(tokenizer)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        kwargs = dict(generation_kwargs or {})
        num_return = int(kwargs.get("num_return_sequences", 1))
        kwargs["num_return_sequences"] = num_return

        pad_token_id = kwargs.get("pad_token_id")
        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError("Tokenizer is missing a pad token id.")
            kwargs["pad_token_id"] = pad_token_id

        kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

        model = model.to(device)
        model.eval()

        results: List[GeneratedSample] = []

        for start in range(0, len(prompts), batch_size):
            end = min(len(prompts), start + batch_size)
            batch_prompts = prompts[start:end]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            prompt_lengths = attention_mask.sum(dim=1).tolist()

            if input_ids.size(1) == 0:
                batch_size_current = input_ids.size(0)
                input_ids = torch.full((batch_size_current, 1), pad_token_id, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
            else:
                zero_length_indices = [i for i, length in enumerate(prompt_lengths) if length == 0]
                if zero_length_indices:
                    input_ids[zero_length_indices, 0] = pad_token_id
                    attention_mask[zero_length_indices, 0] = 1

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            input_length = input_ids.size(1)
            # Only log every 50 generations, regardless of batch size.
            if logger:
                # Calculate how many generations we've made so far.
                if start == 0 or start % 50 < batch_size:
                    logger.info(
                        "Generating batch %d-%d: %d prompts, input_length=%d, num_return_sequences=%d",
                        start + 1,
                        end,
                        len(batch_prompts),
                        input_length,
                        num_return,
                    )

            with torch.no_grad():
                try:
                    output_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **kwargs,
                    )
                except Exception as e:
                    if logger:
                        logger.error(
                            "Error during generation for batch %d-%d: %s",
                            start + 1,
                            end,
                            str(e),
                        )
                    raise

            output_ids = output_ids.cpu()
            
            for i in range(len(batch_prompts)):
                for j in range(num_return):
                    idx = i * num_return + j
                    sequence = output_ids[idx]
                    prompt_len = int(prompt_lengths[i])
                    prompt_tokens = sequence[:prompt_len]
                    continuation_tokens = sequence[prompt_len:]
                    prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                    completion_text = tokenizer.decode(continuation_tokens, skip_special_tokens=True)
                    full_text = tokenizer.decode(sequence, skip_special_tokens=True)
                    results.append(GeneratedSample(prompt_text, completion_text, full_text))


        return results
    finally:
        tokenizer.padding_side = original_padding_side

