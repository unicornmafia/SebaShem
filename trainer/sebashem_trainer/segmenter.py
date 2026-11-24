"""Train the sentence splitter (``sebashem_text.sentences``) on sentence-divided transliteration.

Sources, each read in text order so the gaps between consecutive sentences are real:

* Nederhof and grammar texts (``EgyptianTranslation/texts``), one sentence per
  aligned line;
* sentence-divided JSesh editions, whose ``+t`` blocks are sentences.

The end of the first JSesh edition is held out: ``dev_ratio`` to choose the
threshold, ``test_ratio`` to report precision / recall. Scores are reported twice:
with all texts, and without texts whose name matches ``eval_exclude`` (e.g. the
other manuscripts of the held-out text, which share its sentences).
"""

from __future__ import annotations

import collections
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sebashem_text import sign_codes
from sebashem_text.sentences import HASH_BITS, SentenceSplitter, gap_features, splitter_tokens, word_starts
from sebashem_trainer.corpus import grammar, jsesh, nederhof, tla

RUN_SENTENCES = 6  # sentences per training run


def _runs(sentences: list[list[str]], size: int = RUN_SENTENCES) -> list[tuple[list[str], set[int]]]:
    """Concatenate consecutive sentences into runs: (tokens, indices where sentences start)."""
    runs = []
    for i in range(0, len(sentences), size):
        tokens, starts = [], set()
        for s in sentences[i:i + size]:
            starts.add(len(tokens))
            tokens += s
        runs.append((tokens, starts))
    return runs


def _examples(runs, weight: float, every_gap: bool = False):
    feats, labels, weights = [], [], []
    for tokens, starts in runs:
        for g in (range(1, len(tokens)) if every_gap else sorted(word_starts(tokens))):
            feats.append(gap_features(tokens, g))
            labels.append(1.0 if g in starts else 0.0)
            weights.append(weight)
    return feats, labels, weights


def _fit(feats, labels, weights, epochs: int, lr: float = 0.2, l2: float = 1e-6, seed: int = 1234):
    """Logistic regression on hashed binary features, AdaGrad, minibatches."""
    X = np.array(feats, dtype=np.int64)
    y = np.array(labels, dtype=np.float32)
    sw = np.array(weights, dtype=np.float32)
    w = np.zeros(1 << HASH_BITS, dtype=np.float32)
    g2 = np.full_like(w, 1e-8)
    b, gb2 = 0.0, 1e-8
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(y)), max(1, len(y) // 256)):
            xb = X[idx]
            p = 1 / (1 + np.exp(-(w[xb].sum(1) + b)))
            err = (p - y[idx]) * sw[idx]
            grad = np.zeros_like(w)
            np.add.at(grad, xb.ravel(), np.repeat(err, xb.shape[1]))
            touched = np.unique(xb)
            grad[touched] += l2 * w[touched]
            g2[touched] += grad[touched] ** 2
            w[touched] -= lr * grad[touched] / np.sqrt(g2[touched])
            gb = float(err.sum())
            gb2 += gb * gb
            b -= lr * gb / np.sqrt(gb2)
    return w, b


def score(splitter: SentenceSplitter, runs) -> dict:
    """Precision / recall of predicted sentence starts against ``runs``' gold starts."""
    tp = n_pred = n_gold = 0
    for tokens, gold in runs:
        pred = set(splitter.starts(tokens)) - {0}
        gold = gold - {0}
        tp += len(pred & gold)
        n_pred += len(pred)
        n_gold += len(gold)
    precision, recall = tp / max(n_pred, 1), tp / max(n_gold, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def sign_sentences(gly_path: Path, n: int | None = None) -> list[list[str]]:
    """The sign codes of each sentence of a sentence-divided JSesh edition (the first ``n``)."""
    records = jsesh.load([gly_path])[:n]
    return [s for r in records if r.mdc and (s := sign_codes(r.mdc))]


def fit_sign_refiner(sign_runs: list, tla_path: Path | None, seed: int = 1234) -> tuple[np.ndarray, float]:
    """A sign-level classifier of sentence starts, used to place a start exactly.

    On its own it finds sentences poorly (the signs say little), but among a few
    neighbouring gaps it tells a word's last determinative from the next word's
    first sign. Trained on the editions' sentences plus TLA sentences in shuffled runs.
    """
    import random

    f, l, w = _examples(sign_runs, 3.0, every_gap=True)
    if tla_path:
        sentences = [s for r in tla.load_sentences(tla_path, None, False) if r.mdc and (s := sign_codes(r.mdc))]
        random.Random(seed).shuffle(sentences)
        f2, l2, w2 = _examples(_runs(sentences), 1.0, every_gap=True)
        f, l, w = f + f2, l + l2, w + w2
    return _fit(f, l, w, epochs=6, seed=seed)


def _ordered_texts(text_dirs: list[Path]) -> dict[str, list[list[str]]]:
    by_text: dict[str, list[list[str]]] = collections.OrderedDict()
    for d in text_dirs:
        loader = grammar if "grammar" in d.name else nederhof
        for r in loader.load(d):
            if tokens := splitter_tokens(r.translit or ""):
                by_text.setdefault(r.origin, []).append(tokens)
    return by_text


def _gly_sentences(path: Path) -> list[list[str]]:
    return [t for r in jsesh.load([path]) if (t := splitter_tokens(r.translit or ""))]


def train_segmenter(text_dirs: list[Path], gly_paths: list[Path], out_dir: Path, *, dev_ratio: float = 0.1,
                    test_ratio: float = 0.2, gly_weight: float = 1.0, epochs: int = 6,
                    eval_exclude: str | None = None, tla_path: Path | None = None) -> dict:
    texts = _ordered_texts(text_dirs)
    glys = [_gly_sentences(p) for p in gly_paths]

    held = glys[0]
    cut_dev = len(held) - int(len(held) * (dev_ratio + test_ratio))
    cut_test = len(held) - int(len(held) * test_ratio)
    dev, test = _runs(held[cut_dev:cut_test]), _runs(held[cut_test:])

    def fit(include_text, gly_parts):
        f, l, w = [], [], []
        for name, sentences in texts.items():
            if include_text(name):
                f2, l2, w2 = _examples(_runs(sentences), 1.0)
                f, l, w = f + f2, l + l2, w + w2
        for sentences in gly_parts:
            f2, l2, w2 = _examples(_runs(sentences), gly_weight)
            f, l, w = f + f2, l + l2, w + w2
        weights, bias = _fit(f, l, w, epochs=epochs)
        threshold = max((t / 20 for t in range(3, 18)),
                        key=lambda t: score(SentenceSplitter(weights, bias, t), dev)["f1"])
        return SentenceSplitter(weights, bias, threshold)

    held_out_parts = [held[:cut_dev]] + glys[1:]
    metrics = {}
    splitter = fit(lambda name: True, held_out_parts)
    metrics["all_texts"] = {"threshold": splitter.threshold, "dev": score(splitter, dev), "test": score(splitter, test)}
    if eval_exclude:
        honest = fit(lambda name: eval_exclude.lower() not in name.lower(), held_out_parts)
        metrics[f"without_{eval_exclude}_texts"] = {"threshold": honest.threshold, "dev": score(honest, dev),
                                                     "test": score(honest, test)}
    metrics["test_sentences"] = sum(len(g) for _, g in test)
    print(json.dumps(metrics), flush=True)

    # The shipped model also learns from the held-out part; keep the tuned threshold.
    final = fit(lambda name: True, glys)
    final.threshold = splitter.threshold

    refiner_weights, refiner_bias = fit_sign_refiner(
        [run for p in gly_paths for run in _runs(sign_sentences(p))], tla_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "segmenter.npz", weights=final.weights.astype(np.float16),
                        bias=np.float32(final.bias), refiner_weights=refiner_weights.astype(np.float16),
                        refiner_bias=np.float32(refiner_bias))
    (out_dir / "sebashem.json").write_text(json.dumps({
        "name": out_dir.name,
        "task": "sentences",
        "format": "hashed-logistic",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": final.threshold,
        "min_tokens": final.min_tokens,
        # Placing a start in the glyphs: the best sign-level gap within this many signs of
        # where attention puts it, at this cost per sign moved. Measured end to end on the
        # held-out sentences, exact starts went from F1 0.40 to 0.55.
        "refine_radius": 2,
        "refine_penalty": 0.5,
        "sources": [str(p) for p in text_dirs + gly_paths],
        "metrics": metrics,
        "note": "metrics are from models trained without the held-out part; this model includes it",
    }, indent=2))
    return metrics
