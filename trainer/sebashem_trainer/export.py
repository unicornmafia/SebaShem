"""Convert an OpenNMT-py checkpoint into a CTranslate2 model directory the API can load.

The output directory holds the CTranslate2 files plus ``sebashem.json``, which
tells the API the task, tokenizers and decoding defaults.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = "sebashem.json"


def _convert(checkpoint: Path, out_dir: Path, quantization: str | None,
             alignment_layer: int | None = None, alignment_heads: int | None = None) -> None:
    import torch
    from ctranslate2.converters import OpenNMTPyConverter

    # Checkpoints pickle an argparse.Namespace, which torch.load(weights_only=True)
    # (what the converter uses) refuses. Only convert checkpoints you trust.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    opt = ckpt["opt"]
    if getattr(opt, "self_attn_type", "scaled-dot") == "scaled-dot-flash":
        # Same computation; CTranslate2 only accepts the plain name.
        opt.self_attn_type = "scaled-dot"
    if alignment_layer is not None:
        # Which decoder cross-attention the API uses to place line markers. The converter
        # only reads these when lambda_align is set; otherwise it takes layer -1, head 1.
        opt.lambda_align = getattr(opt, "lambda_align", 0) or 1.0
        opt.alignment_layer = alignment_layer
        opt.alignment_heads = alignment_heads if alignment_heads is not None else 0

    with tempfile.TemporaryDirectory() as tmp:
        patched = Path(tmp) / "model.pt"
        torch.save(ckpt, patched)
        del ckpt
        original_load = torch.load
        torch.load = lambda f, *a, **k: original_load(f, *a, **{**k, "weights_only": False})
        try:
            OpenNMTPyConverter(str(patched)).convert(str(out_dir), quantization=quantization, force=True)
        finally:
            torch.load = original_load


def export_checkpoint(checkpoint: Path, model_dir: Path, *, task: str, src_tokenizer: str, tgt_tokenizer: str,
                      decode: dict, quantization: str | None = "int8", extra: dict | None = None,
                      alignment_layer: int | None = None, alignment_heads: int | None = None) -> Path:
    model_dir = model_dir.resolve()
    staging = model_dir.with_name(model_dir.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    print(f"converting {checkpoint} -> {model_dir} (quantization={quantization})", flush=True)
    _convert(checkpoint, staging, quantization, alignment_layer, alignment_heads)

    manifest = {
        "name": model_dir.name,
        "task": task,
        "format": "ctranslate2",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_checkpoint": str(checkpoint.resolve()),
        "quantization": quantization,
        "alignment": {"layer": alignment_layer, "heads": alignment_heads},
        "tokenizers": {"src": src_tokenizer, "tgt": tgt_tokenizer},
        "decode": decode,
        **(extra or {}),
    }
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2))

    if model_dir.exists():
        shutil.rmtree(model_dir)
    staging.rename(model_dir)
    return model_dir


def update_manifest(model_dir: Path, **fields) -> dict:
    path = model_dir / MANIFEST
    manifest = json.loads(path.read_text())
    manifest.update(fields)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest
