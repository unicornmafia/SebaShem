"""Normalization of MdC-ASCII transliteration and English translations.

These mirror the final normalization steps in EgyptianTranslation's
``egyptiandatasetgenerator.ipynb`` so that free-form input (API requests, JSesh
``+t`` / ``+l`` blocks) looks like the compiled training corpora:

    sDm.n=f (r) pr   ->  sDm .n .f r pr
    'Listen to me, count!'  ->  listen to me count
"""

from __future__ import annotations

import re

from sebashem_text.markers import strip_text_markers

# Egyptological Unicode transliteration -> MdC ASCII (notebook ``to_swap``, plus
# the Unicode 5.1 Egyptological alef/ayin/yod that newer sources use).
UNICODE_TO_MDC = {
    "ḫ": "x", "ḥ": "H", "ȝ": "A", "ꜣ": "A", "ḳ": "q", "ỉ": "i", "ꞽ": "i", "i͗": "i",
    "ḏ": "D", "š": "S", "ʿ": "a", "ꜥ": "a", "ẖ": "X", "h̭": "X", "ṯ": "T",
    "Ḫ": "x", "Ḥ": "H", "Ḏ": "D", "Š": "S", "Ṯ": "T",
}

_TAG = re.compile(r"<[^>]*>")
_NOTE = re.compile(r"<note>.+?</note>", re.S)
_STRAY_CLOSE = re.compile(r"([^.\]])\]")
_STRAY_OPEN = re.compile(r"\[([^.\]])")


def unicode_to_mdc_translit(text: str) -> str:
    for src, dst in UNICODE_TO_MDC.items():
        text = text.replace(src, dst)
    return text


def _strip_markup(text: str) -> str:
    text = text.replace("\r\n", " ").replace("\n", " ")
    text = strip_text_markers(text)  # line markers like (12) or |B276 aren't text
    text = _NOTE.sub("", text)
    return _TAG.sub("", text)


def _strip_stray_brackets(text: str) -> str:
    """Drop ``[`` / ``]`` around restored words but keep ``[...]`` lacunae."""
    text = _STRAY_CLOSE.sub(r"\1", text)
    return _STRAY_OPEN.sub(r"\1", text)


def normalize_transliteration(text: str) -> str:
    """Normalize a transliteration to the spacing/punctuation of the training corpus."""
    s = unicode_to_mdc_translit(_strip_markup(text))
    s = s.replace("(?)", "").replace("^", "").replace(".*", "").replace("*", "")
    s = s.replace("=", ".").replace(": ", " ").replace(":", ".")
    s = re.sub(r"\.{4,}", "...", s)
    s = s.replace(".", " .").replace("a .w .s .", "a.w.s.").replace(" . . .", " [...]")
    s = s.replace("[ [...]]", "[...]").replace("[[...]]", "[...]")
    s = re.sub(r"[(){}]", "", s).replace(",", " ,").replace("-", " ")
    s = _strip_stray_brackets(s)
    return " ".join(s.split())


def scholarly_transliteration(text: str) -> str:
    """Normalize a transliteration to the conventional scholarly layout, for display.

    Suffix pronouns and stative endings attach with ``=``, other endings with ``.``
    (``sDm.n=f``, ``iy=kwi``, ``mAA.tw=f``); ``i`` not ``j`` and ``s`` not ``z``.
    Unlike :func:`normalize_transliteration`, which splits these off as tokens for
    the translator, words stay whole. Which ``.`` is a pronoun can't be decided
    here, so the text must already use ``=`` for those.
    """
    s = unicode_to_mdc_translit(_strip_markup(text))
    s = s.replace("(?)", "").replace("^", "").replace("*", "").replace("~", "")
    s = re.sub(r"[(){}]", "", s)
    s = _strip_stray_brackets(s)
    s = re.sub(r"[,;]", " ", s)
    # Endings written as separate tokens (``Hm =f``, ``sDm .n``) attach to their word.
    s = re.sub(r"\s+(?=[=.][A-Za-z])", "", s)
    s = re.sub(r"(?<=\w)\.(?=\s|$)", "", s)  # full stops, not endings
    s = respell(s, {"j": "i", "z": "s"})
    return " ".join(s.split())


def normalize_translation(text: str) -> str:
    """Normalize an English translation the way the training targets were built."""
    s = _strip_markup(text).replace("&quot;", "")
    s = s.replace(".", " .").replace("[ . . .]", "[...]").replace("l .p .h .", "l.p.h.").replace("!", "")
    s = s.replace(",", "").replace("???", "...").replace("?", "").replace("'", "").replace('"', "")
    s = re.sub(r"[(){}]", "", s).replace("-", " ").replace(":", "").replace(";", "")
    s = _strip_stray_brackets(s)
    return " ".join(x for x in s.split() if x != ".").lower()


# Letters that transliteration conventions write differently for the same sound:
# reed leaf i / j (and y / j), and z / s, which merged in Middle Egyptian.
RESPELL_CANDIDATES = (("i", "j"), ("y", "j"), ("z", "s"))


def respell(text: str, mapping: dict[str, str]) -> str:
    """Apply a per-letter respelling like ``{"i": "j", "z": "s"}`` (MdC ASCII has no other use for these letters)."""
    return text.translate(str.maketrans(mapping)) if mapping else text


def infer_respelling(lines, threshold: float = 0.01) -> dict[str, str]:
    """The respelling that maps text into the convention ``lines`` use.

    For each pair (a, b): if words with ``a`` are under ``threshold`` of words with
    ``b``, the corpus writes b, so input a should become b (and the reverse).
    """
    counts: dict[str, int] = {}
    for line in lines:
        for word in line.split():
            for ch in set(word) & {"i", "j", "y", "z", "s"}:
                counts[ch] = counts.get(ch, 0) + 1
    mapping = {}
    for a, b in RESPELL_CANDIDATES:
        na, nb = counts.get(a, 0), counts.get(b, 0)
        if nb and na < threshold * nb:
            mapping[a] = b
        elif na and nb < threshold * na and b not in mapping.values():
            mapping[b] = a
    return mapping
