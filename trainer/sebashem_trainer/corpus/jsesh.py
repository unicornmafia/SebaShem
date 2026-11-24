"""JSesh ``.gly`` files with glyph lines followed by ``+t`` / ``+l`` text blocks.

Handles both layouts in use:

* annotated editions (``Sinuhe_tm_sebasesh*.gly``): one or more MdC lines, then
  ``+t<transliteration>+s-!`` and ``+l<translation>+s-!`` lines;
* ``sinuhe_trilingual.gly``: ``+s<mdc>+s-!`` / ``+t...`` / ``+l...`` triples.

A record is every glyph line up to the next glyph line that follows non-empty
text, so glyph lines whose text is attached later (empty ``+l+s-!`` in between)
are kept with that text.
"""

from __future__ import annotations

from pathlib import Path

from sebashem_text import (
    normalize_translation,
    normalize_transliteration,
    scholarly_transliteration,
    split_mdc_segments,
)
from sebashem_trainer.records import Record


def _block_text(line: str) -> str:
    body = line[2:]
    return body[: body.rfind("+s")] if "+s" in body else body


def parse_gly(text: str, origin: str) -> list[Record]:
    """Records from one .gly file; ``translit_scholarly`` only if its ``+t`` lines use ``=``."""
    scholarly = any(line.startswith("+t") and "=" in line for line in text.splitlines())
    records: list[Record] = []
    glyphs: list[str] = []
    translit: list[str] = []
    translation: list[str] = []

    def flush() -> None:
        mdc = "-".join(split_mdc_segments("\n".join(glyphs)))
        tr = normalize_transliteration(" ".join(translit))
        tr_scholarly = scholarly_transliteration(" ".join(translit)) if scholarly else ""
        en = normalize_translation(" ".join(translation))
        if sum(map(bool, (mdc, tr, en))) >= 2:
            records.append(Record(origin=origin, mdc=mdc or None, translit=tr or None, translation=en or None,
                                  translit_scholarly=tr_scholarly or None))
        glyphs.clear()
        translit.clear()
        translation.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("++") or line.startswith("+b"):
            continue
        if line.startswith("+t"):
            if body := _block_text(line).strip():
                translit.append(body)
        elif line.startswith("+l"):
            if body := _block_text(line).strip():
                translation.append(body)
        elif line.startswith("+") and not line.startswith("+s"):
            continue
        else:
            if translit or translation:
                flush()
            glyphs.append(line)
    flush()
    return records


def load(paths: list[Path]) -> list[Record]:
    records = []
    for p in paths:
        records.extend(parse_gly(p.read_text(encoding="utf-8", errors="replace"), p.name))
    return records
