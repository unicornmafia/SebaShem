"""Runs against the exported models in ../models (or $MODEL_DIR); skipped when they are absent."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sebashem_api import main

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))


def _require(name):
    if not (MODEL_DIR / name / "sebashem.json").exists():
        pytest.skip(f"no {name} model in {MODEL_DIR}")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(MODEL_DIR))
    with TestClient(main.app) as c:
        yield c


def test_translate(client):
    _require("translit2eng")
    r = client.post("/v1/translate", json={"text": "jrj pat HAtj-a"})
    assert r.status_code == 200
    seg = r.json()["segments"][0]
    assert seg["translation"]
    assert seg["translation_score"] < 0


def test_pipeline(client):
    _require("translit2eng")
    _require("mdc2translit")
    r = client.post("/v1/pipeline", json={"text": "D21:Q3-D36:X1-F4:D36"})
    assert r.status_code == 200
    seg = r.json()["segments"][0]
    assert seg["glyphs"] and seg["transliteration"] and seg["translation"]


def test_translate_places_marker(client):
    _require("translit2eng")
    r = client.post("/v1/translate", json={"text": "Dd.in Hm=f |(12) n wa m nn n smrw"})
    seg = r.json()["segments"][0]
    marked = seg["marked"]["translation"]
    assert marked.count("|(12)") == 1
    assert marked.replace(" |(12)", "").replace("|(12) ", "") == seg["translation"]
