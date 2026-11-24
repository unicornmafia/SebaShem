"""Training config: YAML file plus per-task defaults."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

from sebashem_trainer.corpus import resolve_path


@dataclass(frozen=True)
class Task:
    name: str
    src_field: str
    tgt_field: str
    src_tokenizer: str
    tgt_tokenizer: str
    onmt_defaults: dict


# Hyperparameters from EgyptianTranslation.ipynb (supervised run, BLEU 42.22).
_NOTEBOOK_TRANSFORMER = {
    "save_checkpoint_steps": 1000,
    "train_steps": 3000,
    "valid_steps": 1500,
    "model_dtype": "fp32",
    "optim": "adam",
    "learning_rate": 2,
    "warmup_steps": 8000,
    "decay_method": "noam",
    "adam_beta2": 0.998,
    "max_grad_norm": 0,
    "param_init": 0,
    "param_init_glorot": True,
    "label_smoothing": 0.1,
    "normalization": "tokens",
    "encoder_type": "transformer",
    "decoder_type": "transformer",
    "position_encoding": True,
    "heads": 8,
    "hidden_size": 512,
    "word_vec_size": 512,
    "max_generator_batches": 2,
    "transformer_ff": 2048,
    "dropout_steps": [0],
    "dropout": [0.1],
    "accum_count": 4,
    "attention_dropout": [0.1],
    "enc_layers": 6,
    "dec_layers": 6,
    # CTranslate2 can't convert "scaled-dot-flash" (OpenNMT-py's default on newer torch).
    "self_attn_type": "scaled-dot",
}

# A much smaller model: MdC -> transliteration data is under 1k lines.
_SMALL_TRANSFORMER = {
    **_NOTEBOOK_TRANSFORMER,
    "train_steps": 3000,
    "valid_steps": 500,
    "save_checkpoint_steps": 1000,
    "warmup_steps": 1000,
    "heads": 4,
    "hidden_size": 256,
    "word_vec_size": 256,
    "transformer_ff": 1024,
    "enc_layers": 3,
    "dec_layers": 3,
    "accum_count": 1,
    "batch_size": 32,
    "dropout": [0.3],
    "attention_dropout": [0.2],
}

TASKS = {
    "translit2eng": Task("translit2eng", "translit", "translation", "whitespace", "whitespace", _NOTEBOOK_TRANSFORMER),
    "mdc2translit": Task("mdc2translit", "mdc", "translit", "signs", "chars", _SMALL_TRANSFORMER),
}


@dataclass
class TrainConfig:
    path: Path
    task: Task
    name: str
    work_dir: Path
    output_dir: Path
    sources: list[dict]
    split: dict
    dedupe: str
    corpus_weights: dict
    src_tokenizer: str
    tgt_tokenizer: str
    device: str
    onmt: dict
    export: dict
    decode: dict
    translit_style: str = "corpus"

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    @property
    def data_dir(self) -> Path:
        return self.work_dir / "data"

    @property
    def model_dir(self) -> Path:
        return self.output_dir / self.name


def _translit_style(raw: dict, path: Path) -> str:
    """``corpus``: tokens split as in the notebook corpora (``Hm .f``), what the translator reads.
    ``scholarly``: whole words with ``=`` for pronouns (``Hm=f``); only sources that
    distinguish ``=`` from ``.`` provide it, the others are skipped."""
    style = raw.get("translit_style", "corpus")
    if style not in ("corpus", "scholarly"):
        raise ValueError(f"{path}: translit_style must be 'corpus' or 'scholarly', got {style!r}")
    return style


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    base = path.parent

    task_name = raw.get("task")
    if task_name not in TASKS:
        raise ValueError(f"{path}: 'task' must be one of {sorted(TASKS)}, got {task_name!r}")
    task = TASKS[task_name]
    if not raw.get("sources"):
        raise ValueError(f"{path}: no 'sources' listed")

    tokenizers = raw.get("tokenizers", {})
    name = raw.get("name", task.name)
    return TrainConfig(
        path=path,
        task=task,
        name=name,
        work_dir=resolve_path(raw.get("work_dir", f"../work/{name}"), base),
        output_dir=resolve_path(raw.get("output_dir", "../../models"), base),
        sources=raw["sources"],
        split=raw.get("split", {"strategy": "random"}),
        dedupe=raw.get("dedupe", "pair"),
        corpus_weights=raw.get("corpus_weights") or {},
        src_tokenizer=tokenizers.get("src", task.src_tokenizer),
        tgt_tokenizer=tokenizers.get("tgt", task.tgt_tokenizer),
        device=raw.get("device", "auto"),
        translit_style=_translit_style(raw, path),
        onmt={**copy.deepcopy(task.onmt_defaults), **(raw.get("onmt") or {})},
        export={"quantization": "int8", **(raw.get("export") or {})},
        decode={"beam_size": 5, "max_decoding_length": 256, **(raw.get("decode") or {})},
    )
