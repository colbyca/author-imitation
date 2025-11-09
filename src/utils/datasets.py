"""Dataset loading helpers shared across scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_from_disk


def load_dataset_dict(path: Path) -> DatasetDict:
    """Load a HuggingFace DatasetDict from disk with basic validation."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.")
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {path}, received {type(dataset)!r}.")
    return dataset


def ensure_split(dataset: DatasetDict, split: str) -> Dataset:
    """Return a specific split from a DatasetDict, raising if missing."""

    if split not in dataset:
        available = ", ".join(sorted(dataset.keys()))
        raise ValueError(f"Split '{split}' not found in dataset. Available splits: {available}")
    return dataset[split]


def read_label_mapping(metadata_dir: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Read label mapping metadata produced during preprocessing."""

    mapping_path = metadata_dir / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found at {mapping_path}.")

    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    raw_label2id = payload.get("label2id", {})
    raw_id2label = payload.get("id2label", {})
    if not raw_label2id or not raw_id2label:
        raise ValueError(f"Invalid label mapping contents in {mapping_path}.")

    label2id = {str(name): int(idx) for name, idx in raw_label2id.items()}
    id2label = {int(idx): str(name) for idx, name in raw_id2label.items()}
    return label2id, id2label


def collect_texts(dataset: Dataset, column: str = "text") -> List[str]:
    """Extract a textual column from a dataset."""

    if column not in dataset.column_names:
        raise ValueError(f"Dataset is missing required column '{column}'.")
    return [str(text) for text in dataset[column]]


def texts_for_label(dataset: Dataset, label_id: int, *, label_column: str = "label") -> List[str]:
    """Collect texts belonging to a specific numeric label id."""

    if label_column not in dataset.column_names:
        raise ValueError(f"Dataset is missing required column '{label_column}'.")
    indices = [
        idx
        for idx, label in enumerate(dataset[label_column])
        if int(label) == int(label_id)
    ]
    text_column = dataset.column_names[0] if "text" not in dataset.column_names else "text"
    return [str(dataset[text_column][idx]) for idx in indices]


def load_target_author_texts(
    dataset_dir: Path,
    metadata_dir: Path,
    target_author: str,
    *,
    split: str = "train",
) -> List[str]:
    """Convenience wrapper to load texts for the configured target author."""

    dataset = load_dataset_dict(dataset_dir)
    split_dataset = ensure_split(dataset, split)

    label2id, _ = read_label_mapping(metadata_dir)
    target_label = label2id.get(target_author)
    if target_label is None:
        raise ValueError(f"Target author '{target_author}' missing from label mapping.")

    return texts_for_label(split_dataset, target_label)

