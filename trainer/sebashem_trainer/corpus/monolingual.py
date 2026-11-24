"""Monolingual (transliteration-only) corpus, used for word2vec embeddings.

Ported from the "Web-scrape single-language corpus" section of
``egyptiandatasetgenerator.ipynb``, but reading the already-downloaded HTML pages
(``texts/monolingual-texts``) instead of fetching them.
"""

from __future__ import annotations

import csv
import re
import string
import unicodedata
from pathlib import Path

from sebashem_text import unicode_to_mdc_translit


def _html_lines(path: Path) -> list[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise SystemExit("reading HTML needs beautifulsoup4: pip install -e 'trainer[word2vec]'") from None

    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    lines = []
    for para in soup.select("p"):
        spans = para.select("span")
        if spans and all("Times" in s.attrs.get("style", "") and not s.text.isspace() for s in spans):
            text = unicodedata.normalize("NFKD", para.text).replace("\r\n", " ").replace("\n", " ").strip()
            if text and not text.isnumeric() and not text.startswith("Warning"):
                lines.append(text)
    return lines


def _clean(line: str) -> str:
    s = unicode_to_mdc_translit(unicodedata.normalize("NFC", line))
    s = re.sub(r"^(\d+|1A4)\)\s?", "", s)
    s = s.replace("{", "").replace("}", "").replace("(...)", "[...]").replace("...", "[...]")
    s = re.sub(r"\(\s?\?\)", "", s)
    s = s.replace(".", " .").replace("a .w .s .", "a.w.s.").replace("[ . . .]", "[...]")
    s = s.replace("(", "").replace(")", "").replace("-", " ")
    return " ".join(s.split())


def _is_junk(line: str) -> bool:
    return not line or all(ch.isspace() or ch.isnumeric() or ch in string.punctuation for ch in line)


def load_sentences(path: Path) -> list[str]:
    """Read a monolingual corpus from an HTML directory, a CSV with a ``Sentences`` column, or plain lines."""
    if path.is_dir():
        raw = [line for f in sorted(path.glob("*.htm*")) for line in _html_lines(f)]
        raw = [line for line in raw if "Chapter" not in line and "Sethe" not in line]
        lines = [_clean(line) for line in raw]
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            col = "Sentences" if "Sentences" in (reader.fieldnames or []) else reader.fieldnames[0]
            lines = [row[col] for row in reader]
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if not _is_junk(line.strip())]
