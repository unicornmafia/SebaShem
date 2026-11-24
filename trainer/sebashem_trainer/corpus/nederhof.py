"""Nederhof PhilologEg ``*Tr.txt`` files (``texts/nederhof-texts``).

Ported from section 2 of ``egyptiandatasetgenerator.ipynb``. The regexes are kept
as-is so the compiled corpus matches what the original model was trained on.
"""

from __future__ import annotations

import re
from pathlib import Path

from sebashem_trainer.records import Record


def parse_nederhof_file(path: Path) -> list[tuple[str | None, str, str]]:
    """Return ``(signs, translit, translation)`` per paragraph, unnormalized."""
    text = path.read_text(encoding="utf-8")
    # The header (metadata, then bibliography) is terminated by the last "###".
    without_header = text[text.rindex("###") + 4:].strip() if "###" in text else text.strip()

    out = []
    for para in without_header.split("\n\n"):
        if "\n;" not in para:
            continue
        comma_idx = para.index("\n,\n") if "\n,\n" in para else -1
        semicolon_idx = para.index("\n;")
        signs = para[:comma_idx] if comma_idx != -1 else None
        translit = para[comma_idx + 3:semicolon_idx] if comma_idx != -1 else para[:semicolon_idx]
        translate = para[semicolon_idx + 3:]
        if "jw jr.tw n=j" in para:
            translate = "they made for me"
        out.append((signs, translit, translate))
    return out


def preprocess_translation(t: str) -> str:
    if "<" in t or ">" in t:
        t = t.replace("\n", " ")
        t = re.sub(r"(<note>.+?</note>)", "", t)
        t = re.sub(r"^.+</note>", "", t)
        t = re.sub(r"<@?\d+[a-z]?>?\s?", "", t)
        t = re.sub(r"<[A-Za-z]\d+>\s?", "", t)
        t = re.sub(r"<\d+(:|-)\d+>", "", t)
        t = re.sub(r'<"[A-Za-z]+">', "", t)
        t = re.sub(r"(:|-|,|.)\d+>\s?", "", t)
        t = t.replace("<al>", "").replace("</al>", "")
        t = re.sub(r",?\d*\^pre>(.+?),?\d*\^post>", r"\1", t)
        t = re.sub(r"<I{1,2}", "", t)
        t = t.replace("b>", "")
    t = t.replace("&quot;", "")
    t = t.replace("\r\n", " ").replace("\n", " ")
    t = t.replace(".", " .").replace("[ . . .]", "[...]").replace("l .p .h .", "l.p.h.").replace("!", "")
    t = (t.replace(",", "").replace("???", "...").replace("?", "").replace("'", "")
         .replace("(", "").replace(")", "").replace("{", "").replace("}", "")
         .replace("-", " ").replace(":", "").replace(";", ""))
    t = re.sub(r"([^\.\.\.\]])\]", r"\1", t)
    t = re.sub(r"\[([^\.\.\.\]])", r"\1", t)
    t = " ".join(x for x in t.split() if x != ".")
    return t.lower()


def preprocess_translit(t: str) -> str:
    if "<" in t or ">" in t:
        t = t.replace("\n", " ")
        t = re.sub(r"(<note>.+?</note>)", "", t)
        t = re.sub(r"^.+</note>", "", t)
        t = t.replace('<no>""</no>', "").replace("<no> </no>", "")
        t = re.sub(r"<@?\d+[a-z]?>?\s?", "", t)
        t = re.sub(r"<[A-Za-z]\d+>\s?", "", t)
        t = re.sub(r"<\d+(:|-)\d+>", "", t)
        t = re.sub(r'<"([A-Za-z]+|\d+\s)">\s?', "", t)
        t = re.sub(r"(:|-|,|.)\d+>\s?", "", t)
        t = t.replace("<al>", "").replace("</al>", "")
        t = re.sub(r",?\d*\^pre>(.+?),?\d*\^post>", r"\1", t)
        t = re.sub(r"<I{1,2}", "", t)
        t = re.sub(r"[a-z]>\s?", "", t)
    t = t.replace("^", "").replace("=", ".")
    t = t.replace("\n", " ")
    t = t.replace(".", " .").replace("a .w .s .", "a.w.s.").replace("[ . . .]", "[...]")
    t = t.replace("(", "").replace(")", "").replace("{", "").replace("}", "")
    t = t.replace("-", " ")
    t = re.sub(r"([^\.\.\.\]])\]", r"\1", t)
    t = re.sub(r"\[([^\.\.\.\]])", r"\1", t)
    return " ".join(t.split())


def load(path: Path) -> list[Record]:
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    records = []
    for f in files:
        for _signs, translit, translate in parse_nederhof_file(f):
            # Nederhof's sign field is RES ("[vlr]wn:n:n"), not MdC, so it is not used.
            records.append(Record(origin=f.name, translit=preprocess_translit(translit),
                                  translation=preprocess_translation(translate)))
    return records
