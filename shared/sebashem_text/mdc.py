"""Manuel de Codage (MdC) hieroglyph encoding: cleaning, tokenizing, Unicode rendering.

Input is MdC as written by JSesh, e.g. ``D21:Q3-D36:X1-F4:D36``. The structural
operators (``-`` separator, ``:`` vertical stacking, ``*`` juxtaposition, ``&``
ligature, ``(...)`` grouping, ``<`` / ``>`` cartouche) are kept. JSesh-specific
layout and critical-apparatus markup is stripped, following the rules used to
build ``sinuhe_glyphs_transliteration.txt`` in EgyptianTranslation.
"""

from __future__ import annotations

import re
import unicodedata

from sebashem_text.markers import MarkedText

_NESTED_CITATION = re.compile(r"\[\[(?:(?!\[\[|\]\]).)*\]\]", re.S)
_TEXT_BLOCK = re.compile(r"\+[a-z].*?\+s", re.S)
# ``|goto B1`` is an editorial jump to another manuscript's line, written with a space.
_LINE_LABEL = re.compile(r"\|goto\s+[^-!\s]+|\|[^-!\s]*")
# A line marker outside citations: ``|R12``, ``|B276``, ``|12``, ``|goto B1``.
_LINE_MARKER = re.compile(r"\|goto\s+[^-!\s\[\]:*&()]+|\|[^-!\s\[\]:*&()]+")
_FINE_POSITION = re.compile(r"\{\{[^}]*\}\}")
_ESCAPE = re.compile(r"\\[A-Za-z]+\d*|\\.")
_FORMAT_COMMAND = re.compile(r"\$[a-z]|#[a-z0-9]*")
_DAMAGE_BRACKET = re.compile(r"\[[{?&(\"']|[}?&)\"']\]")
_CARTOUCHE_START = re.compile(r"<[A-Za-z]\d?(?=-)")
_TOKEN = re.compile(r"//|[A-Za-z]+\d+[A-Za-z]*|[A-Za-z]+|\d+|[-:*&()<>]")
_GARDINER = re.compile(r"^(Aa|AA|aa|NL|NU|[A-IK-Z])(\d{1,3})([A-Za-z]{0,2})$")
_GLYPH_PLACEHOLDER = re.compile(r"<g>(.*?)</g>")
_HIEROGLYPH_NAME = re.compile(r"^EGYPTIAN HIEROGLYPH ([A-Z]{1,2})0*(\d+)([A-Z]*)$")

# Common MdC phonetic codes and their Gardiner sign.
ALIASES = {
    "A": "G1", "i": "M17", "y": "Z4", "a": "D36", "w": "G43", "W": "Z7",
    "b": "D58", "p": "Q3", "f": "I9", "m": "G17", "n": "N35", "r": "D21",
    "h": "O4", "H": "V28", "x": "Aa1", "X": "F32", "z": "O34", "s": "S29",
    "S": "N37", "q": "N29", "k": "V31", "g": "W11", "t": "X1", "T": "V13",
    "d": "D46", "D": "I10",
    "mn": "Y5", "ra": "N5", "zp": "O50", "Htp": "R4", "nTr": "R8",
    "HqA": "S38", "wAs": "S40", "nb": "V30", "nfr": "F35", "anx": "S34",
    "Dd": "R11", "xpr": "L1", "ib": "F34", "pr": "O1", "kA": "D28",
    "bA": "G29", "sA": "G39", "ir": "D4", "Hm": "U36", "sw": "M23",
    "nsw": "M23", "dw": "N26", "tA": "N16", "pt": "N1", "Axt": "N27",
    "niwt": "O49", "wr": "G36", "aA": "O29", "Sms": "T18", "Hr": "D2",
    "tp": "D1",
}

OPERATORS = frozenset("-:*&()<>")


def _unwrap_glyph_block(line: str) -> str:
    """``+s<mdc>+s-!`` (a tagged glyph line, as in sinuhe_trilingual.gly) -> ``<mdc>``."""
    if line.startswith("+s"):
        return line[2:].split("+s", 1)[0]
    return line


def _strip_citations(s: str) -> str:
    """Drop ``[[...]]`` citations of other manuscripts (they can nest, and contain ``|S 1``-style references)."""
    prev = None
    while prev != s:
        prev, s = s, _NESTED_CITATION.sub("", s)
    return s


def clean_mdc(text: str) -> str:
    """Strip JSesh markup from one MdC line, leaving signs and structural operators."""
    s = _strip_citations(_TEXT_BLOCK.sub("", _unwrap_glyph_block(text.strip())))
    s = _LINE_LABEL.sub("", s)
    s = _FINE_POSITION.sub("", s)
    s = _ESCAPE.sub("", s)
    s = _FORMAT_COMMAND.sub("", s)
    s = _DAMAGE_BRACKET.sub("", s)
    s = _CARTOUCHE_START.sub("<", s)
    s = re.sub(r"[\[\]{}_!]|\.\.", "", s)
    s = re.sub(r"\s+", "-", s)
    # Removing markup can orphan operators (``[{*R5\*}]:[&*R5*&]`` -> ``*R5:*R5*``).
    s = re.sub(r"[:*&]{2,}", lambda m: ":" if ":" in m.group() else m.group()[0], s)
    s = re.sub(r"(?<=[-(])[:*&]+|[:*&]+(?=[-)])", "", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-:*&")


def split_mdc_segments_marked(text: str) -> list[MarkedText]:
    """Split raw MdC (or a whole .gly file) into cleaned, non-empty segments, keeping line markers.

    Segments end at newlines and at the MdC end-of-line marker ``!``. JSesh header
    lines (``++...``) and non-glyph text blocks (``+t``, ``+l``, ``+b``...) are dropped.
    Line markers (``|R12``) don't end a segment; each is kept at its position. A
    marker with no signs after it in its segment opens the next segment instead.
    """
    segments: list[MarkedText] = []
    pending: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("++") or (line.startswith("+") and not line.startswith("+s")):
            continue
        for part in _strip_citations(_unwrap_glyph_block(line)).split("!"):
            pieces, labels, pos = [], [], 0
            for m in _LINE_MARKER.finditer(part):
                pieces.append(clean_mdc(part[pos:m.start()]))
                labels.append(m.group())
                pos = m.end()
            pieces.append(clean_mdc(part[pos:]))

            # Markers before the first sign of this part belong with any still pending.
            first = next((i for i, p in enumerate(pieces) if p), None)
            if first is None:
                pending += labels
                continue
            lead, labels, pieces = pending + labels[:first], labels[first:], pieces[first:]
            # Trailing markers with nothing after them open the next segment.
            last = max(i for i, p in enumerate(pieces) if p)
            pending = labels[last:]
            labels, pieces = labels[:last], pieces[:last + 1]
            segments.append(MarkedText([""] * len(lead) + pieces, lead + labels, "-"))
    if pending and segments:
        seg = segments[-1]
        segments[-1] = MarkedText(seg.pieces + [""] * len(pending), seg.labels + pending, "-")
    return segments


def split_mdc_segments(text: str) -> list[str]:
    """Like :func:`split_mdc_segments_marked`, without the line markers."""
    return [seg.text for seg in split_mdc_segments_marked(text)]


def tokenize_mdc(mdc: str) -> list[str]:
    """Split cleaned MdC into sign codes and operators."""
    return _TOKEN.findall(mdc)


def canonical_sign(token: str) -> str:
    """``ra`` -> ``N5``, ``U1a`` / ``u1A`` -> ``U1A``, ``AA1`` -> ``Aa1``; other tokens unchanged."""
    token = ALIASES.get(token, token)
    m = _GARDINER.match(token)
    if not m:
        return token
    category, number, suffix = m.groups()
    category = "Aa" if category.lower() == "aa" else category
    return f"{category}{int(number)}{suffix.upper()}"


def sign_codes(mdc: str) -> list[str]:
    """The signs of an MdC string in reading order, layout operators dropped, codes canonicalized.

    This makes JSesh MdC (``D21:Q3-ra``) comparable with sources that only record
    sign sequences, like the TLA corpus and dictionary spellings.
    """
    return [canonical_sign(t) for t in tokenize_mdc(mdc) if t not in OPERATORS and t != "//"]


def unicode_to_gardiner(ch: str) -> str | None:
    """``𓀀`` -> ``A1``; None for anything that isn't a basic Egyptian hieroglyph."""
    try:
        m = _HIEROGLYPH_NAME.match(unicodedata.name(ch))
    except ValueError:
        return None
    return canonical_sign("".join(m.groups())) if m else None


def hieroglyphs_to_mdc(text: str) -> str:
    """Unicode hieroglyphs (as in the TLA corpus) -> MdC sign codes joined with ``-``.

    ``<g>D207</g>`` placeholders for unencoded signs are kept as their code; joiners,
    variation selectors and other characters are dropped.
    """
    codes = []
    for word in text.split():
        pos = 0
        for m in _GLYPH_PLACEHOLDER.finditer(word):
            codes.extend(filter(None, map(unicode_to_gardiner, word[pos:m.start()])))
            if _GARDINER.match(m.group(1)):
                codes.append(canonical_sign(m.group(1)))
            pos = m.end()
        codes.extend(filter(None, map(unicode_to_gardiner, word[pos:])))
    return "-".join(codes)


def gardiner_to_unicode(code: str) -> str | None:
    """Map a Gardiner code (``A1``, ``Aa13``, ``M1A``) or MdC alias to its Unicode sign."""
    code = ALIASES.get(code, code)
    m = _GARDINER.match(code)
    if not m:
        return None
    category, number, suffix = m.groups()
    name = f"EGYPTIAN HIEROGLYPH {category.upper()}{int(number):03d}{suffix.upper()}"
    try:
        return unicodedata.lookup(name)
    except KeyError:
        return None


def mdc_to_unicode(mdc: str) -> tuple[str, list[str]]:
    """Render MdC as a plain sequence of Unicode hieroglyphs.

    Layout (stacking, grouping) is not reproduced. Returns the rendered text and
    the sign codes that have no Unicode equivalent.
    """
    out = []
    unknown = []
    for token in tokenize_mdc(mdc):
        if token in OPERATORS or token == "//":
            continue
        glyph = gardiner_to_unicode(token)
        if glyph is None:
            unknown.append(token)
        else:
            out.append(glyph)
    return "".join(out), unknown
