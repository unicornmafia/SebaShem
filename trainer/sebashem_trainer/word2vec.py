"""Word2vec embeddings for the semi-supervised setup (Generate_Word2Vec.ipynb + SemiSupervised.ipynb).

Writes text-format vectors, which OpenNMT reads via ``src_embeddings`` /
``tgt_embeddings`` with ``embeddings_type: word2vec`` in a config's ``onmt:`` block.
"""

from __future__ import annotations

from pathlib import Path

from sebashem_trainer.corpus.monolingual import load_sentences


def _read_tokenized(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_embeddings(src: Path, tgt: Path, mono: Path | None, out_dir: Path, vector_size: int = 512,
                     window: int = 5, min_count: int = 1, workers: int = 4) -> dict[str, Path]:
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise SystemExit("word2vec needs gensim: pip install -e 'trainer[word2vec]'") from None

    corpora = {"src": _read_tokenized(src), "tgt": _read_tokenized(tgt)}
    if mono:
        corpora["mono"] = [s.split() for s in load_sentences(mono)]
        corpora["extended"] = corpora["src"] + corpora["mono"]

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, sentences in corpora.items():
        print(f"training {name} embeddings on {len(sentences)} sentences", flush=True)
        model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count,
                         workers=workers)
        path = out_dir / f"{name}.w2v.txt"
        model.wv.save_word2vec_format(str(path), binary=False)
        written[name] = path
    return written
