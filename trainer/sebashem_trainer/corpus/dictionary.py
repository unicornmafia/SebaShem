"""SebaSebaDictionary ``entries.json``: dictionary headwords with their Gardiner spellings.

Each entry is ``{"Transliteration": "wsir", "GardinerSigns": "D4 M23 A45", ...}``,
merged from Faulkner, Vygus, Dickson and others. Transliterations are already MdC
ASCII. Entries with uncertain readings (``?``) or numbered homographs are skipped, as are
the few entries whose transliteration is English text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sebashem_text import normalize_transliteration, scholarly_transliteration, sign_codes
from sebashem_trainer.records import Record

_UNCERTAIN = re.compile(r"[?0-9]")
# MdC ASCII only uses A D H S T X as capitals; others mean English text crept in.
_NOT_MDC = re.compile(r"[BCEFGIJKLMNOPQRUVWYZ]")
# Headwords for suffix pronouns and stative endings (".f", ".kwi") are written "=f", "=kwi".
_SUFFIX_ENTRY = re.compile(r"^[.=](?=(?:i|j|k|T|t|f|s|sy|si|n|Tn|tn|sn|w|wi|kwi|kw|ti|tj|tiwny|wy|ny)$)")


def load(path: Path, origin: str | None) -> list[Record]:
    records = []
    for entry in json.loads(path.read_text(encoding="utf-8")):
        signs, translit = entry.get("GardinerSigns"), entry.get("Transliteration")
        if not signs or not translit or _UNCERTAIN.search(translit) or _NOT_MDC.search(translit) or "?" in signs:
            continue
        # Spellings separate signs with spaces or MdC operators; keep the codes only.
        codes = sign_codes(re.sub(r"[\s,]+", "-", signs))
        scholarly = scholarly_transliteration(_SUFFIX_ENTRY.sub("=", translit))
        translit = normalize_transliteration(translit.replace("~", ""))
        if codes and translit:
            records.append(Record(origin=origin or path.name, mdc="-".join(codes), translit=translit,
                                  translit_scholarly=scholarly or None))
    return records
