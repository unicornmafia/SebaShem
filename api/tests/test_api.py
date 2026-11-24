import pytest
from fastapi.testclient import TestClient

from sebashem_api import main
from sebashem_api.alignment import best_cut, place_markers
from sebashem_api.engine import Hypothesis
from sebashem_text import get_tokenizer


class FakeModel:
    """Maps each input through ``fn``; output token i attends to source token i * S / T (monotonic)."""

    def __init__(self, task, fn, src_tokenizer, tgt_tokenizer):
        self.task = task
        self.fn = fn
        self.src_tokenizer = get_tokenizer(src_tokenizer)
        self.tgt_tokenizer = get_tokenizer(tgt_tokenizer)
        self.calls = []

    def translate(self, texts, beam_size=None, with_alignment=False):
        self.calls.append(texts)
        out = []
        for t in texts:
            tokens = self.tgt_tokenizer.encode(self.fn(t))
            n_src = len(self.src_tokenizer.encode(t))
            alignment = [i * n_src // len(tokens) for i in range(len(tokens))] if with_alignment else None
            out.append(Hypothesis(self.tgt_tokenizer.decode(tokens), -0.1, tokens, alignment))
        return out

    def info(self):
        return {"name": self.task, "task": self.task}


@pytest.fixture
def models(monkeypatch):
    fakes = {
        # One output word per input sign, so marker positions are easy to predict.
        "mdc2translit": FakeModel("mdc2translit", lambda t: " ".join(f"w{i}" for i, _ in
                                                                     enumerate(get_tokenizer("signs").encode(t))),
                                  "signs", "chars"),
        "translit2eng": FakeModel("translit2eng", lambda t: " ".join(w.upper() for w in t.split()),
                                  "whitespace", "whitespace"),
    }
    monkeypatch.setattr(main, "load_model", lambda settings, name, task: fakes.get(task))
    return fakes


@pytest.fixture
def client(models, monkeypatch):
    monkeypatch.setattr(main, "_load_splitter", lambda settings: None)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "models": {"mdc2translit": True, "translit2eng": True}}


def test_pipeline_returns_all_three(client):
    r = client.post("/v1/pipeline", json={"text": "|R1-D21_:Q3_-!\nra-A363"})
    assert r.status_code == 200
    body = r.json()
    assert [s["mdc"] for s in body["segments"]] == ["D21:Q3", "ra-A363"]
    seg = body["segments"][1]
    assert seg["glyphs"] == "\U000131F3"
    assert seg["unknown_signs"] == ["A363"]
    assert seg["transliteration"] == "w0 w1"
    assert seg["translation"] == "W0 W1"
    assert seg["line_markers"] == [] and seg["marked"] is None
    assert body["translation"] == "W0 W1 W0 W1"


def test_line_marker_mid_sentence_is_echoed_in_all_three(client):
    # |R12 sits after the third sign; it must not split the sentence.
    r = client.post("/v1/pipeline", json={"text": "A1_:3-G17-|R12-D21_-N16-!"})
    (seg,) = r.json()["segments"]
    assert seg["mdc"] == "A1:3-G17-D21-N16"
    assert seg["line_markers"] == ["|R12"]
    assert seg["marked"] == {
        "mdc": "A1:3-G17-|R12-D21-N16",
        "transliteration": "w0 w1 w2 |R12 w3 w4",
        "translation": "W0 W1 W2 |R12 W3 W4",
    }


def test_marker_on_its_own_line_opens_next_segment(client):
    r = client.post("/v1/pipeline", json={"text": "|B276-!\nD21-N16-!"})
    (seg,) = r.json()["segments"]
    assert seg["marked"]["mdc"] == "|B276-D21-N16"
    assert seg["marked"]["translation"] == "|B276 W0 W1"


class FakeSplitter:
    """Starts a sentence at every word "w1"."""

    def starts(self, tokens, allowed):
        return [0] + [i for i in sorted(allowed) if tokens[i] == "w1"]

    def refine(self, signs, starts, allowed):
        return starts


def test_pipeline_gly_lines(client):
    gly = "++JSesh_page_direction RIGHT_TO_LEFT +s\n+bTitle+s-!\n|R1-D21_:Q3_-!\nX1-|R2-N35-!"
    r = client.post("/v1/pipeline/gly", json={"text": gly, "sentences": False})
    assert r.status_code == 200
    assert r.text.splitlines() == [
        "++JSesh_page_direction LEFT_TO_RIGHT +s", "+bTitle+s-!",
        "|R1-D21_:Q3_-!", "+tw0 w1+s-!", "+lW0 W1+s-!",
        "X1-|R2-N35-!", "+tw0 w1+s-!", "+lW0 W1+s-!",
    ]


def test_pipeline_gly_sentences(client, monkeypatch):
    monkeypatch.setattr(client.app.state, "splitter", FakeSplitter())
    r = client.post("/v1/pipeline/gly", json={"text": "|R1-D21-Q3-X1-!\nN35-|R2-G17-!"})
    lines = r.text.splitlines()
    assert lines[0] == "++JSesh_Info 1.0 +s"
    assert "++JSesh_page_direction LEFT_TO_RIGHT +s" in lines
    # Sentences start at each "w1": the second sign of each manuscript line.
    assert [l for l in lines if not l.startswith("++")] == [
        "|R1-D21-!", "+tw0+s-!", "+lW0+s-!",
        "Q3-X1-N35-!", "+tw0 w1 w2+s-!", "+lW0 W1 W2+s-!",
        "|R2-G17-!", "+tw0+s-!", "+lW0+s-!",
    ]


def test_translate_normalizes_input(client):
    r = client.post("/v1/translate", json={"text": "ḏd.n=f\n\nnsw-bjtj"})
    assert [(s["transliteration"], s["translation"]) for s in r.json()["segments"]] == [
        ("Dd .n .f", "DD .N .F"), ("nsw bjtj", "NSW BJTJ")]


def test_translate_echoes_text_markers(client):
    r = client.post("/v1/translate", json={"text": "ist r=f mSa |(12) r tA timHw"})
    (seg,) = r.json()["segments"]
    assert seg["transliteration"] == "ist r .f mSa r tA timHw"
    assert seg["line_markers"] == ["|(12)"]
    assert seg["marked"]["translation"] == "IST R .F MSA |(12) R TA TIMHW"


def test_empty_input_is_rejected(client):
    assert client.post("/v1/transliterate", json={"text": "++JSesh_Info 1.0 +s\n-!"}).status_code == 422


def test_missing_model_gives_503(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda settings, name, task: None)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/v1/pipeline", json={"text": "D21"}).status_code == 503


def test_best_cut_handles_reordering():
    # "his majesty said" for "Dd Hm .f" attends [2, 1, 0]. With the marker before
    # source token 1 every cut misplaces at least one word; the cheapest is the start.
    assert best_cut([2, 1, 0], 1, [0, 1, 2, 3]) == 0
    assert best_cut([0, 0, 1, 1, 2], 1, [0, 1, 2, 3, 4, 5]) == 2
    assert place_markers([0, 1, 2, 3], [1, 1, 3], [0, 1, 2, 3, 4]) == [1, 1, 3]
