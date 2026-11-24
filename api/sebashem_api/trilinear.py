"""A JSesh .gly file (or plain MdC) -> a trilinear .gly file, one sentence per group.

1. Each manuscript line is transliterated, keeping the decoder's attention.
2. The sentence splitter reads the running transliteration of all lines and picks
   the words that start sentences.
3. Attention maps each of those words back to a gap between JSesh units in its
   line (a sentence starting a line's first word starts at that line), and a
   sign-level model moves it to the best gap within two signs.
4. Each glyph sentence is transliterated again on its own (the model was trained
   on sentences) and translated.

Glyph lines keep their original markup, line markers included; the ``+t`` and
``+l`` lines carry only the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sebashem_api.alignment import best_source_cut
from sebashem_api.gly import HEADER, text_block
from sebashem_text import normalize_transliteration
from sebashem_text.gly_units import GlyDocument, Unit, group_units, parse_gly, render_units
from sebashem_text.sentences import SentenceSplitter, splitter_tokens
from sebashem_text.tokenizers import SPACE_MARK


@dataclass
class Sentence:
    units: list[Unit]
    signs: list[str]
    transliteration: str = ""
    translation: str = ""

    @property
    def glyph_line(self) -> str:
        return render_units(self.units)


@dataclass
class _Line:
    offset: int              # global index of its first sign
    signs: list[str]
    unit_starts: list[int]   # local sign indices where a unit begins (the possible cuts)


def _lines(doc: GlyDocument) -> list[_Line]:
    lines: dict[int, _Line] = {}
    offset = 0
    for unit in doc.units:
        if not unit.signs:
            continue
        line = lines.setdefault(unit.line, _Line(offset, [], []))
        line.unit_starts.append(len(line.signs))
        line.signs += unit.signs
        offset += len(unit.signs)
    return list(lines.values())


def sentence_starts(doc: GlyDocument, mdc_model, splitter: SentenceSplitter | None,
                    beam_size: int | None = None) -> list[int]:
    """Global sign indices where sentences start."""
    lines = _lines(doc)
    if splitter is None:
        return [line.offset for line in lines]
    hyps = mdc_model.translate(["-".join(line.signs) for line in lines], beam_size=beam_size, with_alignment=True)

    tokens: list[str] = []
    allowed: set[int] = set()
    origin: dict[int, tuple[int, int]] = {}  # token index -> (line, output position of its word)
    for n, hyp in enumerate(hyps):
        chars = hyp.tokens
        bounds = [0] + [i + 1 for i, c in enumerate(chars) if c == SPACE_MARK] + [len(chars) + 1]
        for a, b in zip(bounds, bounds[1:]):
            word = "".join(chars[a:b - 1])
            word_tokens = splitter_tokens(word)
            if not word_tokens:
                continue
            if not word_tokens[0].startswith("."):
                allowed.add(len(tokens))
                origin[len(tokens)] = (n, a)
            tokens += word_tokens

    cuts = {0}
    for start in splitter.starts(tokens, allowed):
        if start == 0:
            continue
        n, q = origin[start]
        line, hyp = lines[n], hyps[n]
        if q == 0 or not hyp.alignment:
            cuts.add(line.offset)
            continue
        local = best_source_cut(hyp.alignment, q, line.unit_starts)
        cuts.add(line.offset + (local or 0))
    signs = [s for line in lines for s in line.signs]
    unit_starts = {line.offset + u for line in lines for u in line.unit_starts}
    return splitter.refine(signs, sorted(cuts), unit_starts)


def translate_sentences(sentences: list[Sentence], mdc_model, translit_model, beam_size: int | None = None) -> None:
    hyps = mdc_model.translate(["-".join(s.signs) for s in sentences], beam_size=beam_size)
    for s, hyp in zip(sentences, hyps):
        s.transliteration = hyp.text
    english = translit_model.translate([normalize_transliteration(s.transliteration) for s in sentences],
                                       beam_size=beam_size)
    for s, hyp in zip(sentences, english):
        s.translation = hyp.text


def trilinear(text: str, mdc_model, translit_model, splitter: SentenceSplitter | None,
              beam_size: int | None = None) -> tuple[GlyDocument, list[Sentence]]:
    doc = parse_gly(text)
    starts = sentence_starts(doc, mdc_model, splitter, beam_size)
    sentences = [Sentence(g, [s for u in g for s in u.signs]) for g in group_units(doc.units, starts)]
    sentences = [s for s in sentences if s.signs]
    translate_sentences(sentences, mdc_model, translit_model, beam_size)
    return doc, sentences


def _header(doc: GlyDocument) -> list[str]:
    """The input's JSesh header (or the default one), always left to right."""
    header = doc.header or HEADER.strip().splitlines()
    ltr = "++JSesh_page_direction LEFT_TO_RIGHT +s"
    if any("JSesh_page_direction" in h for h in header):
        return [ltr if "JSesh_page_direction" in h else h for h in header]
    return header + [ltr]


def render_trilinear(doc: GlyDocument, sentences: list[Sentence]) -> str:
    lines = _header(doc) + doc.front
    for s in sentences:
        lines.append(re.sub(r"-*$", "", s.glyph_line) + "-!")
        lines.append(text_block("t", s.transliteration))
        lines.append(text_block("l", s.translation))
    return "\n".join(lines) + "\n"
