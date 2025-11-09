"""System-level helpers shared across scripts."""

from __future__ import annotations

import logging
from typing import Optional

import torch


def setup_logging(level: str, *, fmt: Optional[str] = None) -> None:
    """Initialise global logging configuration with a consistent format."""

    if fmt is None:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format=fmt,
        datefmt="%H:%M:%S",
        force=True,
    )


def select_device(choice: str) -> torch.device:
    """Resolve the requested torch device, supporting an ``auto`` sentinel."""

    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(choice)

