"""Score an exported model on a test set with sacrebleu (BLEU and chrF), as the notebooks did."""

from __future__ import annotations

import json
from pathlib import Path

from sebashem_text import get_tokenizer, respell
from sebashem_trainer.export import MANIFEST


def translate_lines(model_dir: Path, src_lines: list[str], beam_size: int | None = None,
                    batch_size: int = 32) -> list[str]:
    import ctranslate2

    manifest = json.loads((model_dir / MANIFEST).read_text())
    src_tok = get_tokenizer(manifest["tokenizers"]["src"])
    tgt_tok = get_tokenizer(manifest["tokenizers"]["tgt"])
    decode = manifest.get("decode", {})
    mapping = manifest.get("input_respell") or {}
    translator = ctranslate2.Translator(str(model_dir), device="cpu")

    out = []
    for i in range(0, len(src_lines), batch_size):
        batch = [src_tok.encode(respell(line, mapping)) or ["<unk>"] for line in src_lines[i:i + batch_size]]
        results = translator.translate_batch(
            batch,
            beam_size=beam_size or decode.get("beam_size", 5),
            max_decoding_length=decode.get("max_decoding_length", 256),
        )
        out.extend(tgt_tok.decode(r.hypotheses[0]) for r in results)
    return out


def evaluate(model_dir: Path, src_lines: list[str], refs: list[str], pred_path: Path | None = None,
             beam_size: int | None = None) -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    if len(src_lines) != len(refs):
        raise ValueError(f"{len(src_lines)} source lines but {len(refs)} references")
    preds = translate_lines(model_dir, src_lines, beam_size=beam_size)
    if pred_path:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text("".join(p + "\n" for p in preds))
    return {
        "test_sentences": len(refs),
        "bleu": round(BLEU().corpus_score(preds, [refs]).score, 2),
        "chrf": round(CHRF().corpus_score(preds, [refs]).score, 2),
        "exact_match": round(sum(p == r for p, r in zip(preds, refs)) / max(len(refs), 1), 4),
    }
