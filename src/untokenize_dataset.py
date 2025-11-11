"""Script to load a .arrow dataset, untokenize it, and export to a readable format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import BertTokenizerFast, GPT2TokenizerFast, AutoTokenizer


def detect_tokenizer_type(dataset: Dataset) -> Optional[str]:
    """Try to detect which tokenizer was used based on dataset columns."""
    columns = dataset.column_names
    
    # BERT datasets have token_type_ids
    if "token_type_ids" in columns:
        return "bert"
    
    # GPT2 datasets typically don't have token_type_ids
    # Check if we have input_ids but no token_type_ids
    if "input_ids" in columns and "token_type_ids" not in columns:
        # Could be GPT2 or other tokenizer
        # Check dataset info or try to infer
        return "gpt2"
    
    return None


def load_tokenizer(tokenizer_type: Optional[str], model_name: Optional[str] = None) -> Optional[AutoTokenizer]:
    """Load the appropriate tokenizer."""
    if model_name:
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {model_name}: {e}")
            return None
    
    if tokenizer_type == "bert":
        return BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif tokenizer_type == "gpt2":
        return GPT2TokenizerFast.from_pretrained("gpt2")
    
    return None


def untokenize_sample(
    sample: Dict,
    tokenizer: Optional[AutoTokenizer],
    use_attention_mask: bool = True,
) -> Dict:
    """Untokenize a single sample."""
    result = sample.copy()
    
    if tokenizer is None or "input_ids" not in sample:
        return result
    
    input_ids = sample["input_ids"]
    
    # Use attention mask if available to avoid decoding padding tokens
    if use_attention_mask and "attention_mask" in sample:
        attention_mask = sample["attention_mask"]
        # Filter out padding tokens (where attention_mask is 0)
        input_ids = [token_id for token_id, mask in zip(input_ids, attention_mask) if mask == 1]
    
    try:
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        result["decoded_text"] = decoded_text
    except Exception as e:
        result["decoded_text"] = f"<ERROR: {e}>"
    
    return result


def export_dataset(
    dataset: Dataset,
    output_path: Path,
    tokenizer: Optional[AutoTokenizer] = None,
    max_samples: Optional[int] = None,
    format: str = "json",
) -> None:
    """Export dataset to a readable format."""
    num_samples = len(dataset)
    if max_samples:
        num_samples = min(num_samples, max_samples)
    
    print(f"Processing {num_samples} samples...")
    
    # Collect all samples
    samples = []
    for i in range(num_samples):
        sample = {col: dataset[i][col] for col in dataset.column_names}
        if tokenizer:
            sample = untokenize_sample(sample, tokenizer)
        samples.append(sample)
    
    # Export based on format
    if format == "json":
        output_path.write_text(
            json.dumps(samples, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    elif format == "txt":
        lines = []
        for i, sample in enumerate(samples):
            lines.append(f"{'='*80}")
            lines.append(f"Sample {i+1}/{num_samples}")
            lines.append(f"{'='*80}")
            for key, value in sample.items():
                if key == "decoded_text":
                    lines.append(f"\n{key}:")
                    lines.append(f"{value}\n")
                elif isinstance(value, list):
                    # Truncate long lists
                    if len(value) > 50:
                        lines.append(f"{key}: [{value[0]}, {value[1]}, ..., {value[-1]}] (length: {len(value)})")
                    else:
                        lines.append(f"{key}: {value}")
                else:
                    lines.append(f"{key}: {value}")
            lines.append("")
        
        output_path.write_text("\n".join(lines), encoding="utf-8")
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Exported {num_samples} samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a .arrow dataset, untokenize it, and export to a readable format."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to dataset directory (or specific split directory)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Specific split to export (if dataset is a DatasetDict). If not specified, exports all splits.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer model name (e.g., 'bert-base-uncased', 'gpt2'). If not specified, will try to auto-detect.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "txt"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to export (default: all)",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    
    print(f"Loading dataset from {args.dataset_path}...")
    loaded = load_from_disk(str(args.dataset_path))
    
    # Handle DatasetDict vs Dataset
    if isinstance(loaded, DatasetDict):
        if args.split:
            if args.split not in loaded:
                available = ", ".join(loaded.keys())
                raise ValueError(f"Split '{args.split}' not found. Available: {available}")
            dataset = loaded[args.split]
            print(f"Using split: {args.split}")
        else:
            # Export all splits
            for split_name, split_dataset in loaded.items():
                output_path = args.output.parent / f"{args.output.stem}_{split_name}{args.output.suffix}"
                print(f"\nProcessing split: {split_name}")
                
                # Detect tokenizer
                tokenizer_type = detect_tokenizer_type(split_dataset)
                print(f"Detected tokenizer type: {tokenizer_type}")
                
                tokenizer = load_tokenizer(tokenizer_type, args.tokenizer)
                if tokenizer:
                    print(f"Loaded tokenizer: {tokenizer.__class__.__name__}")
                
                export_dataset(split_dataset, output_path, tokenizer, args.max_samples, args.format)
            return
    else:
        dataset = loaded
    
    # Detect tokenizer
    tokenizer_type = detect_tokenizer_type(dataset)
    print(f"Detected tokenizer type: {tokenizer_type}")
    
    tokenizer = load_tokenizer(tokenizer_type, args.tokenizer)
    if tokenizer:
        print(f"Loaded tokenizer: {tokenizer.__class__.__name__}")
    else:
        print("Warning: No tokenizer available. Will export without decoding.")
    
    # Export
    export_dataset(dataset, args.output, tokenizer, args.max_samples, args.format)


if __name__ == "__main__":
    main()

