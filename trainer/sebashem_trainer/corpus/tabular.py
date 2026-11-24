"""Generic line/column based sources: CSV/TSV, parallel line files, alternating lines."""

from __future__ import annotations

import csv
from pathlib import Path

from sebashem_text import normalize_translation, normalize_transliteration, split_mdc_segments
from sebashem_trainer.records import Record

FIELDS = ("mdc", "translit", "translation")


def normalize_field(field: str, value: str | None) -> str | None:
    if value is None:
        return None
    if field == "mdc":
        value = "-".join(split_mdc_segments(value))
    elif field == "translit":
        value = normalize_transliteration(value)
    else:
        value = normalize_translation(value)
    return value or None


def _check_fields(fields) -> None:
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown field(s) {sorted(unknown)}; expected {FIELDS}")


def load_csv(path: Path, columns: dict[str, str], origin: str | None, normalize: bool,
             delimiter: str = ",", origin_column: str | None = None) -> list[Record]:
    """``columns`` maps record fields to CSV header names, e.g. ``{translit: Transliteration}``."""
    _check_fields(columns)
    records = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            values = {}
            for field, col in columns.items():
                v = (row.get(col) or "").strip()
                values[field] = normalize_field(field, v) if normalize else (v or None)
            row_origin = row.get(origin_column) if origin_column else None
            records.append(Record(origin=row_origin or origin or path.name, **values))
    return records


def _read_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if path.suffix == ".csv":
        # Single-column CSVs written by pandas quote lines containing commas.
        lines = [next(csv.reader([ln]))[0] if ln.startswith('"') else ln for ln in lines]
    return lines


def load_parallel(files: dict[str, Path], origin: str | None, normalize: bool) -> list[Record]:
    """Line-aligned files, one per field (e.g. ``aligned_train.egy.csv`` + ``.eng.csv``)."""
    _check_fields(files)
    columns = {field: _read_lines(p) for field, p in files.items()}
    lengths = {field: len(lines) for field, lines in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"parallel files have different line counts: {lengths}")
    name = origin or next(iter(files.values())).name
    records = []
    for i in range(next(iter(lengths.values()))):
        values = {}
        for field, lines in columns.items():
            v = lines[i].strip()
            values[field] = normalize_field(field, v) if normalize else (v or None)
        records.append(Record(origin=name, **values))
    return records


def load_alternating(path: Path, fields: list[str], origin: str | None, normalize: bool,
                     skip_marker: str = "[not preserved") -> list[Record]:
    """Groups of consecutive lines, one per field (``sinuhe_glyphs_transliteration.txt``)."""
    _check_fields(fields)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) % len(fields):
        raise ValueError(f"{path}: {len(lines)} lines is not a multiple of {len(fields)} fields")
    records = []
    for i in range(0, len(lines), len(fields)):
        values = {}
        for field, line in zip(fields, lines[i:i + len(fields)]):
            line = line.strip()
            if skip_marker and skip_marker in line:
                line = ""
            values[field] = normalize_field(field, line) if normalize else (line or None)
        records.append(Record(origin=origin or path.name, **values))
    return records
