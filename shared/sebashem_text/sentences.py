"""Sentence boundaries in running transliteration.

Manuscripts run sentences together and their lines don't follow sentences. A
linear classifier decides for each gap between two words whether a sentence
starts there, from hashed n-grams of the words around it: sentence-initial words
(iw, ist, aHa.n, Dd.in), verb forms (sDm.n=f), what the sentence before ends with.
Transliteration carries these cues far better than the sign sequence.

Words are compared as tokens in the corpus layout (``sDm .n .f``, i not j, s not
z), so text in any layout can be split; a sentence never starts at a suffix
(a token beginning with ``.``). Training lives in ``sebashem_trainer.segmenter``.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sebashem_text.translit import normalize_transliteration, respell

HASH_BITS = 20
WINDOW = (-4, 4)  # tokens before / after the gap
PAD = "<s>"


def splitter_tokens(text: str) -> list[str]:
    """Transliteration (any layout) -> the tokens the splitter compares."""
    return [t for t in respell(normalize_transliteration(text), {"j": "i", "z": "s"}).split() if t != ","]


def word_starts(tokens: list[str]) -> set[int]:
    """Gaps where a sentence may start: before a word, not before a suffix like ``.f``."""
    return {i for i, t in enumerate(tokens) if i and not t.startswith(".")}


def gap_features(tokens: list[str], gap: int) -> list[int]:
    """Hashed features for the gap before ``tokens[gap]``."""
    lo, hi = WINDOW
    ctx = {k: (tokens[gap + k] if 0 <= gap + k < len(tokens) else PAD) for k in range(lo, hi)}
    feats = [f"u{k}={ctx[k]}" for k in ctx]
    feats += [f"b{k}={ctx[k]}|{ctx[k + 1]}" for k in range(lo, hi - 1)]
    feats += [f"t{k}={ctx[k]}|{ctx[k + 1]}|{ctx[k + 2]}" for k in range(-3, 2)]
    feats += [f"q-4={ctx[-4]}|{ctx[-3]}|{ctx[-2]}|{ctx[-1]}", f"q0={ctx[0]}|{ctx[1]}|{ctx[2]}|{ctx[3]}"]
    mask = (1 << HASH_BITS) - 1
    return [zlib.crc32(f.encode()) & mask for f in feats]


@dataclass
class SentenceSplitter:
    weights: np.ndarray
    bias: float
    threshold: float = 0.5
    min_tokens: int = 2
    # Optional sign-level model for placing starts exactly in the glyphs (see ``refine``).
    refiner: np.ndarray | None = None
    refine_radius: int = 2
    refine_penalty: float = 0.5

    @classmethod
    def load(cls, model_dir: Path) -> "SentenceSplitter":
        manifest = json.loads((model_dir / "sebashem.json").read_text())
        data = np.load(model_dir / "segmenter.npz")
        refiner = data["refiner_weights"].astype(np.float32) if "refiner_weights" in data else None
        return cls(data["weights"].astype(np.float32), float(data["bias"]), manifest.get("threshold", 0.5),
                   manifest.get("min_tokens", 2), refiner, manifest.get("refine_radius", 2),
                   manifest.get("refine_penalty", 0.5))

    def refine(self, signs: list[str], starts: list[int], allowed: set[int]) -> list[int]:
        """Move each sign-index start to the best gap (in ``allowed``) within ``refine_radius`` signs.

        Attention places a start only roughly; the sign model tells, say, a word's last
        determinative from the next word's first sign.
        """
        if self.refiner is None:
            return starts
        out = set()
        for p in starts:
            lo, hi = max(1, p - self.refine_radius), min(len(signs) - 1, p + self.refine_radius)
            cands = [g for g in range(lo, hi + 1) if g in allowed] if p else []
            out.add(max(cands, key=lambda g: self.refiner[gap_features(signs, g)].sum()
                        - self.refine_penalty * abs(g - p)) if cands else p)
        return sorted(out)

    def probabilities(self, tokens: list[str]) -> np.ndarray:
        """P(a sentence starts before ``tokens[i]``) for i in 1..len-1 (index 0 is always a start)."""
        scores = np.array([self.weights[gap_features(tokens, g)].sum() + self.bias for g in range(1, len(tokens))])
        return 1.0 / (1.0 + np.exp(-scores)) if len(scores) else scores

    def starts(self, tokens: list[str], allowed: set[int] | None = None) -> list[int]:
        """Token indices where sentences start (always including 0).

        Only gaps in ``allowed`` are considered (by default, word starts). Confident
        gaps are taken first; one closer than ``min_tokens`` tokens to an earlier pick
        is skipped.
        """
        allowed = word_starts(tokens) if allowed is None else allowed
        probs = self.probabilities(tokens)
        order = sorted(range(1, len(tokens)), key=lambda g: -probs[g - 1])
        chosen: list[int] = []
        for g in order:
            if probs[g - 1] < self.threshold:
                break
            if g not in allowed:
                continue
            if g < self.min_tokens or len(tokens) - g < self.min_tokens:
                continue
            if all(abs(g - c) >= self.min_tokens for c in chosen):
                chosen.append(g)
        return [0] + sorted(chosen)
