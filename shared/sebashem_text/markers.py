"""Manuscript line markers inside a segment.

A line marker records where a physical line of the source starts (``|R12`` in
JSesh MdC; ``(12)``, ``|(12)``, ``|B276`` or Nederhof's ``<12>`` in transliteration).
It is not a sentence boundary: a sentence can run across it. Markers are kept out
of what the models see and put back into the output afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# In transliteration / translation text.
TEXT_MARKER = re.compile(r"\|\(\d+[a-z]?\)|\|[A-Za-z]*\d+[A-Za-z]*\b|\(\d+[a-z]?\)|<\d+[a-z]?>")


@dataclass
class MarkedText:
    """Text split at line markers: ``pieces[0] labels[0] pieces[1] ... labels[-1] pieces[-1]``."""

    pieces: list[str]
    labels: list[str]
    joiner: str = " "

    def __post_init__(self) -> None:
        if len(self.pieces) != len(self.labels) + 1:
            raise ValueError("MarkedText needs exactly one more piece than labels")

    @property
    def text(self) -> str:
        """The text without markers."""
        return self.joiner.join(p for p in self.pieces if p)

    def prefix(self, i: int) -> str:
        """The text before marker ``i``."""
        return self.joiner.join(p for p in self.pieces[: i + 1] if p)

    def render(self) -> str:
        """The text with each marker back in place."""
        items = [self.pieces[0]]
        for label, piece in zip(self.labels, self.pieces[1:]):
            items += [label, piece]
        return self.joiner.join(item for item in items if item)


def split_text_markers(text: str, normalize=None) -> MarkedText:
    """Pull line markers out of transliteration/translation text, normalizing each piece."""
    pieces, labels, pos = [], [], 0
    for m in TEXT_MARKER.finditer(text):
        pieces.append(text[pos:m.start()])
        labels.append(m.group())
        pos = m.end()
    pieces.append(text[pos:])
    if normalize:
        pieces = [normalize(p) for p in pieces]
    else:
        pieces = [" ".join(p.split()) for p in pieces]
    return MarkedText(pieces, labels, " ")


def strip_text_markers(text: str) -> str:
    return TEXT_MARKER.sub(" ", text)
