"""CTranslate2 models exported by the trainer (a model dir + ``sebashem.json``)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sebashem_api.alignment import place_markers
from sebashem_text import MarkedText, get_tokenizer, respell
from sebashem_text.tokenizers import SPACE_MARK

log = logging.getLogger(__name__)

MANIFEST = "sebashem.json"


@dataclass
class Hypothesis:
    text: str
    score: float | None
    tokens: list[str] = field(default_factory=list)
    # For each output token, the source token it attends to most (when requested).
    alignment: list[int] | None = None


class Seq2SeqModel:
    def __init__(self, path: Path, *, device: str = "cpu", compute_type: str = "default", inter_threads: int = 1,
                 intra_threads: int = 0, batch_size: int = 32):
        import ctranslate2

        self.path = path
        self.manifest = json.loads((path / MANIFEST).read_text())
        self.src_tokenizer = get_tokenizer(self.manifest["tokenizers"]["src"])
        self.tgt_tokenizer = get_tokenizer(self.manifest["tokenizers"]["tgt"])
        self.decode = self.manifest.get("decode", {})
        # Respelling into the training data's convention (e.g. i -> j), recorded by the trainer.
        self.input_respell = self.manifest.get("input_respell") or {}
        self.batch_size = batch_size
        self.translator = ctranslate2.Translator(str(path), device=device, compute_type=compute_type,
                                                 inter_threads=inter_threads, intra_threads=intra_threads)

    @property
    def task(self) -> str:
        return self.manifest["task"]

    def info(self) -> dict:
        keys = ("name", "task", "created", "quantization", "tokenizers", "decode", "metrics")
        return {k: self.manifest[k] for k in keys if k in self.manifest}

    def translate(self, texts: list[str], beam_size: int | None = None,
                  with_alignment: bool = False) -> list[Hypothesis]:
        """Translate each text; empty inputs give empty outputs. ``score`` is the mean token log-probability."""
        results: list[Hypothesis] = [Hypothesis("", None)] * len(texts)
        todo = [(i, toks) for i, t in enumerate(texts)
                if (toks := self.src_tokenizer.encode(respell(t, self.input_respell)))]
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            outputs = self.translator.translate_batch(
                [toks for _, toks in chunk],
                beam_size=beam_size or self.decode.get("beam_size", 5),
                max_decoding_length=self.decode.get("max_decoding_length", 256),
                # With the default length_penalty=1 the score is the length-normalized log-probability.
                return_scores=True,
                return_attention=with_alignment,
            )
            for (i, _), out in zip(chunk, outputs):
                tokens = out.hypotheses[0]
                alignment = None
                if with_alignment:
                    alignment = [max(range(len(row)), key=row.__getitem__) for row in out.attention[0]]
                results[i] = Hypothesis(self.tgt_tokenizer.decode(tokens), round(out.scores[0], 4), tokens,
                                        alignment)
        return results


def translate_marked(model, inputs: list[MarkedText], beam_size: int | None = None,
                     ) -> list[tuple[Hypothesis, MarkedText]]:
    """Translate texts that contain line markers; returns each hypothesis and its output with markers placed.

    The model sees ``text`` (markers removed). Each marker's source position is the
    token count of the text before it, and attention decides where it goes in the output.
    """
    need_alignment = any(m.labels for m in inputs)
    hyps = model.translate([m.text for m in inputs], beam_size=beam_size, with_alignment=need_alignment)
    out = []
    for marked, hyp in zip(inputs, hyps):
        if not marked.labels:
            out.append((hyp, MarkedText([hyp.text], [], " ")))
            continue
        tokens = hyp.tokens
        positions = [len(model.src_tokenizer.encode(marked.prefix(i))) for i in range(len(marked.labels))]
        if model.tgt_tokenizer.name == "chars":
            # Only cut between words.
            allowed = [p for p in range(len(tokens) + 1) if p in (0, len(tokens)) or tokens[p - 1] == SPACE_MARK]
        else:
            allowed = list(range(len(tokens) + 1))
        if hyp.alignment is None or not tokens:
            cuts = [0 if k == 0 else len(tokens) for k in positions]
        else:
            cuts = place_markers(hyp.alignment, positions, allowed)
        bounds = [0, *cuts, len(tokens)]
        pieces = [model.tgt_tokenizer.decode(tokens[a:b]) for a, b in zip(bounds, bounds[1:])]
        out.append((hyp, MarkedText(pieces, marked.labels, " ")))
    return out


def load_model(settings, name: str, expected_task: str) -> Seq2SeqModel | None:
    path = settings.model_dir / name
    if not (path / MANIFEST).exists():
        log.warning("model %r not found at %s; its endpoints will return 503", name, path)
        return None
    model = Seq2SeqModel(path, device=settings.device, compute_type=settings.compute_type,
                         inter_threads=settings.inter_threads, intra_threads=settings.intra_threads,
                         batch_size=settings.batch_size)
    if model.task != expected_task:
        raise RuntimeError(f"{path} is a {model.task!r} model, expected {expected_task!r}")
    log.info("loaded %s (%s) from %s", name, model.task, path)
    return model
