"""Drive OpenNMT-py (build_vocab + train) the same way the notebooks did, via a generated YAML."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

from sebashem_trainer.config import TrainConfig
from sebashem_trainer.dataset import train_files


def _device_opts(device: str) -> dict:
    if device == "cpu":
        return {"world_size": 1}
    if device == "auto":
        import torch

        device = "gpu" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            return {"world_size": 1}
    if device == "gpu":
        return {"world_size": 1, "gpu_ranks": [0]}
    raise ValueError(f"device must be auto, cpu or gpu, got {device!r}")


def write_onmt_config(cfg: TrainConfig) -> Path:
    data, run = cfg.data_dir, cfg.work_dir / "run"
    run.mkdir(parents=True, exist_ok=True)
    corpora = {}
    for name, (src, tgt) in train_files(data).items():
        corpora[name] = {"path_src": str(src), "path_tgt": str(tgt)}
        if name in cfg.corpus_weights:
            corpora[name]["weight"] = cfg.corpus_weights[name]
    if not corpora:
        raise FileNotFoundError(f"no training files in {data}; run prepare first")
    unknown = set(cfg.corpus_weights) - set(corpora)
    if unknown:
        raise ValueError(f"corpus_weights names corpora with no training data: {sorted(unknown)}")
    onmt_cfg = {
        "save_data": str(run / "samples"),
        "src_vocab": str(run / "vocab.src"),
        "tgt_vocab": str(run / "vocab.tgt"),
        "overwrite": True,
        "data": {
            **corpora,
            "valid": {"path_src": str(data / "valid.src"), "path_tgt": str(data / "valid.tgt")},
        },
        "save_model": str(run / "model"),
        "log_file": str(run / "train.log"),
        **_device_opts(cfg.device),
        **cfg.onmt,
    }
    path = cfg.work_dir / "onmt.yaml"
    path.write_text(yaml.safe_dump(onmt_cfg, sort_keys=False))
    return path


def _run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_vocab(onmt_yaml: Path) -> None:
    _run("onmt.bin.build_vocab", "-config", str(onmt_yaml), "-n_sample", "-1")


def train(onmt_yaml: Path, train_from: Path | None = None) -> None:
    args = ["-config", str(onmt_yaml)]
    if train_from:
        args += ["-train_from", str(train_from)]
    _run("onmt.bin.train", *args)


def latest_checkpoint(cfg: TrainConfig) -> Path:
    run = cfg.work_dir / "run"
    checkpoints = sorted(run.glob("model_step_*.pt"), key=lambda p: int(re.search(r"_(\d+)\.pt$", p.name).group(1)))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return checkpoints[-1]


_VALID_ACC = re.compile(r"Validation accuracy: ([\d.]+)")
_SAVED = re.compile(r"Saving checkpoint (\S+\.pt)")


def best_checkpoint(cfg: TrainConfig) -> Path:
    """The saved checkpoint with the highest validation accuracy, from OpenNMT's log.

    OpenNMT validates, then saves, so each "Saving checkpoint" line takes the most
    recent validation accuracy. Falls back to the latest checkpoint.
    """
    log = cfg.work_dir / "run" / "train.log"
    scores, last_acc = {}, None
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            if m := _VALID_ACC.search(line):
                last_acc = float(m.group(1))
            elif (m := _SAVED.search(line)) and last_acc is not None:
                scores[Path(m.group(1))] = last_acc
    existing = {p: acc for p, acc in scores.items() if p.exists()}
    if not existing:
        return latest_checkpoint(cfg)
    best = max(existing, key=existing.get)
    print(f"best checkpoint {best.name} (validation accuracy {existing[best]})", flush=True)
    return best
