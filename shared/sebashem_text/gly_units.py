"""The glyph lines of a JSesh .gly file as a stream of units, for re-dividing it into sentences.

A unit is what sits between two top-level ``-``: a sign or quadrat, a line marker
(``|R12``), a format toggle (``#b``, ``$r``), or a whole bracketed group (a cartouche
``<-...->``, a variant reading ``[[-|B3, 4-G17-]]``). Sentences are cut between
units, so a cartouche or a cited variant is never split, and each unit keeps its
original markup.

A line that opens with ``[[`` (text restored where the manuscript is lost) is an
exception: its brackets are toggles and its content is ordinary units, so the
restored text is read and can be cut. Like shading (``#b``/``#e``) and red
(``$r``/``$b``), a toggle still open at a cut is closed and reopened around it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sebashem_text.mdc import clean_mdc, sign_codes

# Openers and closers of bracketed groups that are kept whole.
_OPEN = re.compile(r"\[\[|\[\{|\[&|\[\?|\[\"|\['")
_CLOSE = re.compile(r"\]\]|\}\]|&\]|\?\]|\"\]|'\]")
_CARTOUCHE_OPEN = re.compile(r"^<[A-Za-z0-9]{0,2}$")
_CARTOUCHE_CLOSE = re.compile(r"^[A-Za-z0-9]{0,2}>$")

# Toggles: (opener, closer). The restored-text bracket is "[[" at a line start.
RESTORED, SHADING, RED = ("[[", "]]"), ("#b", "#e"), ("$r", "$b")
TOGGLES = (RESTORED, SHADING, RED)
_FORMAT_TOGGLE = re.compile(r"#[be](?![0-9])|\$[rb]")


@dataclass
class Unit:
    raw: str
    signs: list[str]
    line: int
    kind: str  # "signs", "marker", "open" (goes with what follows), "close" (with what precedes)
    open_before: tuple = ()  # toggles open just before this unit
    open_after: tuple = ()


@dataclass
class GlyDocument:
    header: list[str] = field(default_factory=list)  # ++JSesh_* lines
    front: list[str] = field(default_factory=list)   # +b / +i text blocks before the glyphs
    units: list[Unit] = field(default_factory=list)
    lines: int = 0


def _pieces(line: str) -> list[str]:
    """Top-level ``-``-separated pieces, keeping bracketed groups and cartouches whole."""
    out, cur, depth = [], [], 0
    for piece in line.split("-"):
        cur.append(piece)
        depth += len(_OPEN.findall(piece)) - len(_CLOSE.findall(piece))
        if _CARTOUCHE_OPEN.match(piece):
            depth += 1
        elif _CARTOUCHE_CLOSE.match(piece) and depth > 0:
            depth -= 1
        if depth <= 0:
            out.append("-".join(cur))
            cur, depth = [], 0
    if cur:
        out.append("-".join(cur))
    return out


def parse_gly(text: str) -> GlyDocument:
    doc = GlyDocument()
    open_toggles: list[tuple] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("++"):
            doc.header.append(line)
            continue
        if line.startswith("+"):
            if line[1:2] in ("b", "i") and not doc.units:
                doc.front.append(line)
            continue  # existing +t / +l blocks are regenerated
        n = doc.lines
        doc.lines += 1
        if line.startswith("[["):
            # Restored text: the opening "[[" (with any "_") becomes a toggle.
            m = re.match(r"\[\[_*", line)
            line = line[m.end():].lstrip("-")
            doc.units.append(_unit(m.group(), n, "open", open_toggles))
            open_toggles.append(RESTORED)
            doc.units[-1].open_after = tuple(open_toggles)
        for piece in _pieces(line):
            if piece in ("", "!"):
                continue
            if RESTORED in open_toggles and re.fullmatch(r"\]\]_*", piece):
                kind = "close"
            elif piece in (SHADING[0], RED[0]):
                kind = "open"
            elif piece in (SHADING[1], RED[1]):
                kind = "close"
            elif piece.startswith("|"):
                kind = "marker"
            else:
                kind = "signs"
            unit = _unit(piece, n, kind, open_toggles)
            if RESTORED in open_toggles and kind == "close" and piece.startswith("]]"):
                open_toggles.remove(RESTORED)
            # Shading and red can also be glued to a sign ("Y1_$b").
            for m in _FORMAT_TOGGLE.finditer(piece):
                toggle = next(t for t in (SHADING, RED) if m.group() in t)
                if m.group() == toggle[0] and toggle not in open_toggles:
                    open_toggles.append(toggle)
                elif m.group() == toggle[1] and toggle in open_toggles:
                    open_toggles.remove(toggle)
            unit.open_after = tuple(open_toggles)
            doc.units.append(unit)
    return doc


def _unit(raw: str, line: int, kind: str, open_toggles: list) -> Unit:
    signs = sign_codes(clean_mdc(raw)) if kind == "signs" else []
    if kind == "signs" and not signs:
        kind = "close"  # a variant reading or a format command: stays with the text before it
    return Unit(raw, signs, line, kind, tuple(open_toggles), tuple(open_toggles))


def render_units(units: list[Unit]) -> str:
    """Glyph markup for consecutive units, closing and reopening toggles cut at either end."""
    if not units:
        return ""
    before = [t[0] for t in units[0].open_before]
    after = [t[1] for t in reversed(units[-1].open_after)]  # innermost first
    return "-".join(before + [u.raw for u in units] + after)


def group_units(units: list[Unit], sign_starts: list[int]) -> list[list[Unit]]:
    """Divide ``units`` into groups starting at the given global sign indices.

    Units without signs go with the group they belong to: markers and openers with
    the next signs, closers and variant readings with the previous ones.
    """
    starts = set(sign_starts)
    groups: list[list[Unit]] = [[]]
    pending: list[Unit] = []  # markers / openers waiting for the next signs
    sign_index = 0
    for unit in units:
        if unit.kind == "signs":
            if sign_index in starts and sign_index and groups[-1]:
                groups.append([])
            groups[-1] += pending + [unit]
            pending = []
            sign_index += len(unit.signs)
        elif unit.kind in ("marker", "open"):
            pending.append(unit)
        else:
            (pending if pending else groups[-1]).append(unit)
    if pending:
        groups[-1] += pending
    return [g for g in groups if g]
