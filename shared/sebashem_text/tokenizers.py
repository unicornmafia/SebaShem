"""Tokenizers named in a model's manifest.

The trainer writes pre-tokenized (space separated) files for OpenNMT and records
the tokenizer names in ``sebashem.json``; the API looks them up here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sebashem_text.mdc import sign_codes, tokenize_mdc

SPACE_MARK = "▁"


@dataclass(frozen=True)
class Tokenizer:
    name: str
    encode: Callable[[str], list[str]]
    decode: Callable[[list[str]], str]


def _chars_encode(text: str) -> list[str]:
    return [SPACE_MARK if ch == " " else ch for ch in " ".join(text.split())]


def _chars_decode(tokens: list[str]) -> str:
    return " ".join("".join(tokens).replace(SPACE_MARK, " ").split())


TOKENIZERS = {
    t.name: t
    for t in (
        # Word level, as in the original notebooks (OpenNMT's default split).
        Tokenizer("whitespace", str.split, " ".join),
        # MdC sign codes and layout operators, e.g. ``D21 : Q3 - D36``.
        Tokenizer("mdc", tokenize_mdc, "".join),
        # Sign codes only (no layout operators), canonicalized; lets layout-free
        # sources (TLA, dictionaries) and JSesh MdC share one vocabulary.
        Tokenizer("signs", sign_codes, "-".join),
        # Character level with ``▁`` for spaces; used for transliteration output,
        # where the vocabulary is too sparse for word level with little data.
        Tokenizer("chars", _chars_encode, _chars_decode),
    )
}


def get_tokenizer(name: str) -> Tokenizer:
    try:
        return TOKENIZERS[name]
    except KeyError:
        raise ValueError(f"unknown tokenizer {name!r}; expected one of {sorted(TOKENIZERS)}") from None
