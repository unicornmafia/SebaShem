"""Corpus sources named in a training config's ``sources:`` list."""

from __future__ import annotations

import glob
import os
from pathlib import Path

from sebashem_trainer.corpus import dictionary, grammar, jsesh, nederhof, tabular, tla
from sebashem_trainer.records import Record


def resolve_path(value: str, base_dir: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(value)))
    return p if p.is_absolute() else (base_dir / p).resolve()


def resolve_glob(value: str, base_dir: Path, default_pattern: str) -> list[Path]:
    p = resolve_path(value, base_dir)
    if p.is_dir():
        return sorted(p.glob(default_pattern))
    if any(ch in value for ch in "*?["):
        return sorted(Path(m) for m in glob.glob(str(p)))
    return [p]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"corpus path does not exist: {path}")
    return path


def load_source(spec: dict, base_dir: Path) -> list[Record]:
    kind = spec.get("type")
    normalize = spec.get("normalize", True)
    origin = spec.get("origin")

    if kind == "nederhof":
        return nederhof.load(_require(resolve_path(spec["path"], base_dir)))
    if kind == "grammar":
        return grammar.load(_require(resolve_path(spec["path"], base_dir)))
    if kind == "jsesh":
        paths = resolve_glob(spec["path"], base_dir, "*.gly")
        if not paths:
            raise FileNotFoundError(f"no .gly files match {spec['path']!r}")
        return jsesh.load([_require(p) for p in paths])
    if kind == "csv":
        return tabular.load_csv(_require(resolve_path(spec["path"], base_dir)), spec["columns"], origin, normalize,
                                delimiter=spec.get("delimiter", ","), origin_column=spec.get("origin_column"))
    if kind == "parallel":
        files = {field: _require(resolve_path(p, base_dir)) for field, p in spec["files"].items()}
        return tabular.load_parallel(files, origin, normalize)
    if kind == "alternating":
        return tabular.load_alternating(_require(resolve_path(spec["path"], base_dir)), spec["fields"], origin,
                                        normalize)
    if kind == "tla":
        return tla.load_sentences(_require(resolve_path(spec["path"], base_dir)), origin,
                                  with_translation=spec.get("translation", False))
    if kind == "tla_wordlist":
        return tla.load_wordlist(_require(resolve_path(spec["path"], base_dir)), origin,
                                 min_count=spec.get("min_count", 1))
    if kind == "dictionary":
        return dictionary.load(_require(resolve_path(spec["path"], base_dir)), origin)
    raise ValueError(f"unknown source type {kind!r}; expected nederhof, grammar, jsesh, csv, parallel, "
                     "alternating, tla, tla_wordlist or dictionary")
