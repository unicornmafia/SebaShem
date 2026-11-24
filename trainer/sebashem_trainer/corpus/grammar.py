"""Grammar example files (Allen, Callender, Gardiner, Loprieno, Ockinga).

Ported from section 1 of ``egyptiandatasetgenerator.ipynb``. Files are read from
a local directory; download them from
``https://mjn.host.cs.st-andrews.ac.uk/egyptian/grammars/`` if you don't have them.
"""

from __future__ import annotations

import re
from pathlib import Path

from sebashem_trainer.records import Record


def parse_paragraph(paragraph: str) -> tuple[str, str] | None:
    merged: list[str] = []
    for line in paragraph.split("\n"):
        if not line.startswith("%") and merged:
            merged[-1] += " " + line
        elif line.startswith("%"):
            merged.append(line)
    fields = {item[:3].strip(): item[3:].strip() for item in merged if item.startswith(("%al", "%tr"))}
    if "%al" not in fields or "%tr" not in fields:
        return None
    return fields["%al"], fields["%tr"]


def preprocess_translation(t: str) -> str:
    if "<" in t or ">" in t:
        t = re.sub(r"<em>(.+)</em>", r"\1", t)
        t = re.sub(r"<i>(.+)</i>", r"\1", t)
        t = re.sub(r"<.+>(.+)</.+>", "", t)
    if "lit" in t.lower():
        t = re.sub(r"\(\s*\)", "", t)
        t = re.sub(r"\(lit\.(.+?)\)", "", t)
        t = re.sub(r"(,|\.|\!|\?)\s*(l|L)it\..+$", "", t)
    if "(" in t or ")" in t:
        t = re.sub(r"(it) \(.+?\)", r"\1", t)
        t = re.sub(r"\(=.+?\)", "", t)
        t = t.replace(")", "").replace("(", "")
    if "[" in t or "]" in t:
        t = re.sub(r"\[.+?\]\s*", "", t)
    if "i.e" in t.lower():
        t = re.sub(r"(,|\.) (i|I)\.e\..+$", ".", t)

    t = t.replace("favor", "favour")
    t = " ".join(t.split())
    t = re.sub(r"\.{4,}", r"...", t)
    t = t.replace("[...]", "...").replace(".", " .").replace(" . . .", " [...]").replace(":", "").replace(";", "")
    t = t.replace(",", "").replace("???", "...").replace("?", "").replace("'", "").replace('"', "")
    t = re.sub(r"([^\.\.\.\]])\]", r"\1", t)
    t = re.sub(r"\[([^\.\.\.\]])", r"\1", t)
    t = " ".join(x for x in t.split() if x != "." and len(x))
    return t.lower()


def preprocess_translit(t: str) -> str:
    if "<no>" in t:
        # "<no>written</no>intended<no>..." -> swap in the intended word and drop the tags.
        matches = list(re.finditer(r"</no>(.+?)<no>", t))
        if len(matches) >= 2:
            t = t.replace(matches[0].group(1), matches[1].group(1))
        t = re.sub(r"<no>(.+)</no>", "", t).strip()
    t = t.replace("i", "j").replace("y", "j")
    t = t.replace("^", "").replace("(?)", "")
    t = t.replace(".*", "").replace("*", "")
    t = t.replace(": ", " ").replace(":", ".")
    t = t.replace("=", ".").replace("-", " ")
    t = re.sub(r"\.{4,}", r"...", t)
    t = t.replace(".", " .").replace(" . . .", "[...]")
    t = t.replace("(", "").replace(")", "").replace(",", " ,")
    t = re.sub(r"([^\.\.\.\]])\]", r"\1", t)
    t = re.sub(r"\[([^\.\.\.\]])", r"\1", t)
    return " ".join(t.split())


def load(path: Path) -> list[Record]:
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    records = []
    for f in files:
        for para in f.read_text(encoding="utf-8").split("\n\n"):
            if "%al" not in para or "%tr" not in para:
                continue
            parsed = parse_paragraph(para)
            if parsed is None:
                continue
            translit, translate = parsed
            records.append(Record(origin=f.name, translit=preprocess_translit(translit),
                                  translation=preprocess_translation(translate)))
    return records
