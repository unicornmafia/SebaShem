"""Thesaurus Linguae Aegyptiae data, as prepared in SebaSebaDictionary/dictionaries.

* ``train.jsonl`` / ``train_en.jsonl``: TLA "Original Earlier Egyptian sentences"
  (CC BY-SA 4.0). Unicode hieroglyphs and Unicode transliteration, word-aligned.
* ``tla_wordlist.jsonl``: one line per lemma with its attested spellings
  (``"gardiner": "I9 Z4"``), built by ``build_tla_wordlist.py``.

The English in ``train_en.jsonl`` is a MarianMT translation of TLA's German, so it
is only loaded when a source asks for it with ``translation: true``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sebashem_text import (
    hieroglyphs_to_mdc,
    normalize_translation,
    normalize_transliteration,
    scholarly_transliteration,
    unicode_to_mdc_translit,
)
from sebashem_trainer.records import Record

_MORPH_MARKER = re.compile(r"\(?\.?(?:PL|DU)\)?")


def _tla_letters(text: str) -> str:
    """TLA's Unicode transliteration -> MdC-ASCII letters, keeping ``=`` and ``.``."""
    s = _MORPH_MARKER.sub("", text)
    # TLA writes the special phonemes with Unicode letters, so capitals only mark
    # names (Wnis, Ꞽmn); lowercase before MdC ASCII gives case a meaning (h/H, s/S).
    s = s.lower()
    s = s.replace("\u1e96", "X").replace("h\u0331", "X").replace("\u1e71", "t").replace("\u02be", "")
    s = unicode_to_mdc_translit(s)
    s = re.sub("[\u0300-\u036f]", "", s)  # remaining combining marks, e.g. the breve in i̯ / u̯
    return re.sub(r"[!+~/]|[\U00013000-\U0001343F]", "", s)


def tla_translit_to_mdc(text: str) -> str:
    """``ḥm.t =f``, ``(w)di̯``, ``kꜣ(.PL)`` -> MdC-ASCII corpus format (``Hm .t .f``, ``wdi``, ``kA``)."""
    return normalize_transliteration(_tla_letters(text))


def tla_translit_scholarly(text: str, glossing: str = "") -> str:
    """``ḥm.t =f``, ``ḥtp.w`` (glossed stative) -> ``Hm.t=f``, ``Htp=w``.

    TLA writes suffix pronouns as ``=`` tokens but the stative ending with ``.``;
    the glossing (``V\\res-3sg.m``) says which words are statives.
    """
    words, glosses = text.split(), glossing.split()
    if len(glosses) != len(words):
        glosses = [""] * len(words)
    out = []
    for word, gloss in zip(words, glosses):
        word = _tla_letters(word)
        if "\\res" in gloss and "." in word.lstrip("="):
            i = word.rfind(".")
            word = word[:i] + "=" + word[i + 1:]
        out.append(word)
    return scholarly_transliteration(" ".join(out))


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_sentences(path: Path, origin: str | None, with_translation: bool) -> list[Record]:
    records = []
    for row in _read_jsonl(path):
        translation = row.get("translation") if with_translation else None
        records.append(Record(
            origin=origin or path.name,
            mdc=hieroglyphs_to_mdc(row.get("hieroglyphs", "")) or None,
            translit=tla_translit_to_mdc(row.get("transliteration", "")) or None,
            translation=normalize_translation(translation) if translation else None,
            translit_scholarly=tla_translit_scholarly(row.get("transliteration", ""),
                                                      row.get("glossing", "")) or None,
        ))
    return records


def load_wordlist(path: Path, origin: str | None, min_count: int) -> list[Record]:
    """One record per attested spelling of each lemma."""
    records = []
    for row in _read_jsonl(path):
        translit = tla_translit_to_mdc(row.get("lemma", ""))
        scholarly = tla_translit_scholarly(row.get("lemma", ""))
        if not translit:
            continue
        for spelling in row.get("spellings", []):
            if spelling.get("count", 1) < min_count:
                continue
            mdc = "-".join(spelling.get("gardiner", "").split())
            if mdc:
                records.append(Record(origin=origin or path.name, mdc=mdc, translit=translit,
                                      translit_scholarly=scholarly or None))
    return records
