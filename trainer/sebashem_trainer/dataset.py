"""Build tokenized train/valid/test files for OpenNMT from the configured sources.

Each source belongs to a corpus (``corpus:``, default ``main``). Training data is
written per corpus so OpenNMT can sample them with different weights
(``corpus_weights:``). Sources marked ``train_only`` (e.g. dictionary word pairs)
never go into valid/test.
"""

from __future__ import annotations

import json
import random
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sebashem_text import get_tokenizer
from sebashem_trainer.config import TrainConfig
from sebashem_trainer.corpus import load_source, resolve_path

SPLITS = ("train", "valid", "test")
DEFAULT_CORPUS = "main"


@dataclass
class Example:
    src: str
    tgt: str
    origin: str
    corpus: str = DEFAULT_CORPUS
    train_only: bool = False


def _is_junk(text: str) -> bool:
    return all(ch.isspace() or ch.isnumeric() or ch in string.punctuation for ch in text)


def _usable(src: str | None, tgt: str | None) -> bool:
    # Same filter the notebook applied before writing aligned.{egy,eng}.csv.
    return bool(src and tgt) and not _is_junk(src) and not _is_junk(tgt) and "[...]" not in src and "[...]" not in tgt


def _holdout_keys(cfg: TrainConfig, encode) -> set[tuple]:
    keys = set()
    for p in cfg.split.get("holdout", []):
        path = resolve_path(p, cfg.base_dir)
        keys.update(tuple(encode(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return keys


def _field(cfg: TrainConfig, field: str) -> str:
    return "translit_scholarly" if field == "translit" and cfg.translit_style == "scholarly" else field


def collect_examples(cfg: TrainConfig) -> tuple[list[Example], dict]:
    encode = get_tokenizer(cfg.src_tokenizer).encode
    holdout = _holdout_keys(cfg, encode)
    seen_src: set[tuple] = set()
    seen_pair: set[tuple] = set()
    examples: list[Example] = []
    per_source = []

    for spec in cfg.sources:
        if spec.get("corpus") == "valid":
            raise ValueError("'valid' is reserved by OpenNMT; give the source another corpus name")
        dedupe = spec.get("dedupe", cfg.dedupe)
        if dedupe not in ("pair", "src", "none"):
            raise ValueError(f"dedupe must be 'pair', 'src' or 'none', got {dedupe!r}")
        records = load_source(spec, cfg.base_dir)
        counts = Counter()
        for r in records:
            src, tgt = r.get(_field(cfg, cfg.task.src_field)), r.get(_field(cfg, cfg.task.tgt_field))
            if not _usable(src, tgt):
                counts["unusable"] += 1
                continue
            src, tgt = src.strip(), tgt.strip()
            key = tuple(encode(src))
            if key in holdout:
                counts["held_out"] += 1
                continue
            if (dedupe == "src" and key in seen_src) or (dedupe == "pair" and (key, tgt) in seen_pair):
                counts["duplicate"] += 1
                continue
            seen_src.add(key)
            seen_pair.add((key, tgt))
            examples.append(Example(src, tgt, r.origin, spec.get("corpus", DEFAULT_CORPUS),
                                    spec.get("train_only", False)))
            counts["kept"] += 1
        per_source.append({"type": spec.get("type"), "path": spec.get("path") or spec.get("files"),
                           "corpus": spec.get("corpus", DEFAULT_CORPUS), "records": len(records), **counts})

    return examples, {"sources": per_source, "examples": len(examples), "holdout_lines": len(holdout)}


def split_examples(examples: list[Example], split: dict) -> dict[str, list[Example]]:
    train_only = [ex for ex in examples if ex.train_only]
    examples = [ex for ex in examples if not ex.train_only]
    strategy = split.get("strategy", "random")
    if strategy == "by_origin":
        val, test = set(split.get("val", [])), set(split.get("test", []))
        origins = {ex.origin for ex in examples}
        missing = (val | test) - origins
        if missing:
            raise ValueError(f"split lists origins with no examples: {sorted(missing)}")
        return {
            "train": [ex for ex in examples if ex.origin not in val and ex.origin not in test] + train_only,
            "valid": [ex for ex in examples if ex.origin in val],
            "test": [ex for ex in examples if ex.origin in test],
        }
    if strategy == "random":
        shuffled = examples[:]
        random.Random(split.get("seed", 1234)).shuffle(shuffled)
        n_val = round(len(shuffled) * split.get("val_ratio", 0.1))
        n_test = round(len(shuffled) * split.get("test_ratio", 0.1))
        return {
            "valid": shuffled[:n_val],
            "test": shuffled[n_val:n_val + n_test],
            "train": shuffled[n_val + n_test:] + train_only,
        }
    raise ValueError(f"split strategy must be 'by_origin' or 'random', got {strategy!r}")


def train_files(data_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Training corpora written by :func:`prepare`: ``{corpus: (src, tgt)}``."""
    return {p.name[len("train."):-len(".src")]: (p, p.with_suffix(".tgt")) for p in sorted(data_dir.glob("train.*.src"))}


def prepare(cfg: TrainConfig) -> dict:
    """Write ``train.<corpus>.{src,tgt}``, ``{valid,test}.{src,tgt}`` (tokenized) and
    ``test.{input,ref,corpus}`` (plain, for evaluation) into the work dir."""
    examples, stats = collect_examples(cfg)
    if not examples:
        raise SystemExit("no usable examples found in the configured sources")
    splits = split_examples(examples, cfg.split)
    if not splits["train"] or not splits["valid"]:
        raise SystemExit(f"train and valid splits must be non-empty: { {k: len(v) for k, v in splits.items()} }")

    src_tok, tgt_tok = get_tokenizer(cfg.src_tokenizer), get_tokenizer(cfg.tgt_tokenizer)

    def write(path: Path, rows: list[Example]) -> None:
        path.with_suffix(".src").write_text("".join(" ".join(src_tok.encode(ex.src)) + "\n" for ex in rows))
        path.with_suffix(".tgt").write_text("".join(" ".join(tgt_tok.encode(ex.tgt)) + "\n" for ex in rows))

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    for stale in cfg.data_dir.glob("train*"):
        stale.unlink()
    by_corpus: dict[str, list[Example]] = {}
    for ex in splits["train"]:
        by_corpus.setdefault(ex.corpus, []).append(ex)
    for corpus, rows in by_corpus.items():
        write(cfg.data_dir / f"train.{corpus}.x", rows)
    write(cfg.data_dir / "valid.x", splits["valid"])
    write(cfg.data_dir / "test.x", splits["test"])

    # Untokenized test set, so evaluation goes through the same tokenizers as the API.
    test = splits["test"]
    (cfg.data_dir / "test.input").write_text("".join(ex.src + "\n" for ex in test))
    (cfg.data_dir / "test.ref").write_text("".join(ex.tgt + "\n" for ex in test))
    (cfg.data_dir / "test.corpus").write_text("".join(ex.corpus + "\n" for ex in test))

    stats["splits"] = {name: len(splits[name]) for name in SPLITS}
    stats["train_corpora"] = {corpus: len(rows) for corpus, rows in by_corpus.items()}
    stats["top_origins"] = Counter(ex.origin for ex in examples).most_common(10)
    (cfg.data_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
