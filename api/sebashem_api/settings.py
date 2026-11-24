from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    mdc_model: str
    translit_model: str
    sentence_model: str
    device: str
    compute_type: str
    inter_threads: int
    intra_threads: int
    max_input_chars: int
    max_segments: int
    batch_size: int
    max_gly_chars: int

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            model_dir=Path(env("MODEL_DIR", "models")),
            mdc_model=env("MDC_MODEL", "mdc2translit"),
            translit_model=env("TRANSLIT_MODEL", "translit2eng"),
            sentence_model=env("SENTENCE_MODEL", "sentences"),
            device=env("DEVICE", "cpu"),
            compute_type=env("COMPUTE_TYPE", "default"),
            inter_threads=int(env("INTER_THREADS", "1")),
            intra_threads=int(env("INTRA_THREADS", "0")),
            max_input_chars=int(env("MAX_INPUT_CHARS", "20000")),
            max_segments=int(env("MAX_SEGMENTS", "200")),
            batch_size=int(env("BATCH_SIZE", "32")),
            # Whole .gly files for /v1/pipeline/gly (Sinuhe is ~50k characters).
            max_gly_chars=int(env("MAX_GLY_CHARS", "1000000")),
        )
