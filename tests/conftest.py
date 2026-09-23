"""Shared fixtures: data, mock LM, and program. Everything runs offline."""

from __future__ import annotations

from pathlib import Path

import dspy
import pytest

from src.data import load_examples
from src.mock_lm import WeakMockLM
from src.program import build_program

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def train_examples():
    return load_examples(ROOT / "data" / "train.jsonl")


@pytest.fixture()
def eval_examples():
    return load_examples(ROOT / "data" / "eval.jsonl")


@pytest.fixture()
def mock_lm():
    lm = WeakMockLM()
    dspy.configure(lm=lm)
    return lm


@pytest.fixture()
def program(mock_lm):
    return build_program()
