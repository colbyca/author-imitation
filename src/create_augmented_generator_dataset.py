"""Create an augmented generator dataset by combining original texts with hard negatives.

This script:
1. Loads the original 50 training texts from the generator_g0 dataset
2. Extracts hard negative texts from generator_texts.json (of the last iteration)
3. Combines them into a new dataset
4. Saves the augmented dataset for training
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset
from transformers import GPT2TokenizerFast

from config import CONFIG
from utils import setup_logging

LOGGER = logging.getLogger("create_augmented_generator_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator-texts-file",
        type=Path,
        required=True,
        help="Path to generator_texts.json file containing texts and hard_neg_indices.",
    )
    parser.add_argument(
        "--original-texts-file",
        type=Path,
        default=CONFIG.paths.metadata_dir / "target_corpus_train.txt",
        help="Path to file containing original 50 training texts (default: metadata/target_corpus_train.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG.paths.reuters_processed_dir / "generator_Nplus",
        help="Directory to save the augmented dataset (default: data/processed/reuter/generator_Nplus).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists.",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def load_original_texts(path: Path) -> list[str]:
    """Load the original 50 training texts from the metadata file."""
    if not path.exists():
        raise FileNotFoundError(f"Original texts file not found: {path}")
    
    content = path.read_text(encoding="utf-8")
    # Texts are separated by double newlines
    texts = [text.strip() for text in content.split("\n\n") if text.strip()]
    
    LOGGER.info("Loaded %d original texts from %s", len(texts), path)
    return texts


def load_generator_texts_with_hard_negatives(path: Path) -> tuple[list[str], list[int]]:
    """Load generator texts and extract hard negative indices."""
    if not path.exists():
        raise FileNotFoundError(f"Generator texts file not found: {path}")
    
    data = json.loads(path.read_text(encoding="utf-8"))
    texts = data.get("texts", [])
    hard_neg_indices = data.get("hard_neg_indices", [])
    
    LOGGER.info(
        "Loaded %d texts and %d hard negative indices from %s",
        len(texts),
        len(hard_neg_indices),
        path,
    )
    
    return texts, hard_neg_indices


def extract_hard_negative_texts(texts: list[str], indices: list[int]) -> list[str]:
    """Extract hard negative texts using the provided indices."""
    hard_negatives = []
    for idx in indices:
        if idx < 0 or idx >= len(texts):
            LOGGER.warning("Invalid index %d (texts length: %d), skipping", idx, len(texts))
            continue
        hard_negatives.append(texts[idx])
    
    LOGGER.info("Extracted %d hard negative texts", len(hard_negatives))
    return hard_negatives


def create_generator_dataset_from_texts(
    texts: list[str],
    tokenizer: GPT2TokenizerFast,
    block_size: int,
    stride: int,
) -> Dataset:
    """Create a generator dataset from a list of texts."""
    if not texts:
        raise ValueError("No texts provided for generator dataset.")
    
    concatenated = "\n\n".join(texts)
    encoding = tokenizer(
        concatenated,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids = encoding["input_ids"]
    if not input_ids:
        raise ValueError("Tokenisation produced an empty sequence for generator.")
    
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if stride >= block_size:
        raise ValueError("stride must be smaller than block_size.")
    
    total_length = len(input_ids)
    if total_length < block_size:
        raise ValueError(
            "Generator dataset too small for the requested block size. "
            f"Have {total_length} tokens but require at least {block_size}."
        )
    
    step = block_size - stride if stride > 0 else block_size
    max_start = total_length - block_size + 1
    examples = [input_ids[i : i + block_size] for i in range(0, max_start, step)]
    
    dataset = Dataset.from_dict(
        {
            "input_ids": examples,
            "labels": [example.copy() for example in examples],
        }
    )
    return dataset


def main() -> None:
    args = parse_args()
    setup_logging(args.logging_level)
    
    original_texts = load_original_texts(args.original_texts_file)
    if len(original_texts) != 50:
        LOGGER.warning(
            "Expected 50 original texts, but found %d. Continuing anyway.",
            len(original_texts),
        )
    
    # Load generator texts and hard negative indices
    generator_texts, hard_neg_indices = load_generator_texts_with_hard_negatives(
        args.generator_texts_file
    )
    
    # Extract hard negative texts
    hard_negative_texts = extract_hard_negative_texts(generator_texts, hard_neg_indices)
    
    # Combine original texts with hard negatives
    combined_texts = original_texts + hard_negative_texts
    LOGGER.info(
        "Combined dataset: %d original texts + %d hard negatives = %d total texts",
        len(original_texts),
        len(hard_negative_texts),
        len(combined_texts),
    )
    
    # Prepare tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(CONFIG.tokenizers.gpt2_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    LOGGER.info("Creating generator dataset...")
    dataset = create_generator_dataset_from_texts(
        combined_texts,
        tokenizer,
        CONFIG.tokenizers.gpt2_block_size,
        CONFIG.tokenizers.gpt2_stride,
    )
    
    LOGGER.info("Created dataset with %d examples", len(dataset))
    
    # Prepare output directory
    output_dir = args.output_dir
    if output_dir.exists():
        if args.overwrite:
            LOGGER.info("Removing existing directory: %s", output_dir)
            import shutil
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    LOGGER.info("Saving dataset to %s", output_dir)
    dataset.save_to_disk(str(output_dir))
    
    # Save metadata
    metadata = {
        "num_original_texts": len(original_texts),
        "num_hard_negatives": len(hard_negative_texts),
        "num_total_texts": len(combined_texts),
        "num_examples": len(dataset),
        "hard_neg_indices": hard_neg_indices,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved metadata to %s", metadata_path)
    
    LOGGER.info("Augmented dataset created successfully at %s", output_dir)


if __name__ == "__main__":
    main()

