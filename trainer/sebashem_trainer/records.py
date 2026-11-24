from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Record:
    """One aligned unit from a corpus. Any of the three text fields may be missing.

    ``origin`` identifies the source file, so splits can hold out whole texts
    (as the notebook did with e.g. ``WestcarTr.txt``).
    """

    origin: str
    mdc: str | None = None
    translit: str | None = None
    translation: str | None = None
    # The transliteration in scholarly layout (``Hm=f``, ``sDm.n=f``), for sources
    # that mark suffix pronouns with ``=``; see ``translit_style`` in the config.
    translit_scholarly: str | None = None

    def get(self, field: str) -> str | None:
        return getattr(self, field)
