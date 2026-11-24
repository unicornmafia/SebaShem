import json
from pathlib import Path

import yaml

from sebashem_trainer.config import load_config
from sebashem_trainer.corpus.jsesh import parse_gly
from sebashem_trainer.corpus.nederhof import parse_nederhof_file, preprocess_translation, preprocess_translit
from sebashem_trainer.corpus.tla import load_wordlist, tla_translit_to_mdc
from sebashem_trainer.dataset import Example, prepare, split_examples, train_files
from sebashem_trainer.onmt_runner import write_onmt_config

GLY = """++JSesh_Info 1.0 +s
+bSinuhe+s-!
|R1-D21_:Q3_-D36_:X1_-!
+tiri-pat HAti-a+s-!
+lMember of the Elite, Governor+s-!
|R3-T18_-S29_-!
+l+s-!
D21_:X1_-G36_:D21_-!
+tSms nb=f+s-!
+lwho followed his lord+s-!
"""

NEDERHOF = """creator = someone
###
bibliography
###

<1> Dd.jn Smsw jqr
;
<1> An excellent follower says:

wDA-jb=k
;
'May it please you,
"""


def test_parse_gly_groups_glyph_lines_with_their_text():
    records = parse_gly(GLY, "x.gly")
    assert [(r.mdc, r.translit, r.translation) for r in records] == [
        ("D21:Q3-D36:X1", "iri pat HAti a", "member of the elite governor"),
        ("T18-S29-D21:X1-G36:D21", "Sms nb .f", "who followed his lord"),
    ]


def test_parse_nederhof(tmp_path):
    f = tmp_path / "ShipwreckedTr.txt"
    f.write_text(NEDERHOF)
    rows = [(preprocess_translit(t), preprocess_translation(e)) for _, t, e in parse_nederhof_file(f)]
    assert rows == [("Dd .jn Smsw jqr", "an excellent follower says"), ("wDA jb .k", "may it please you")]


def test_split_by_origin():
    exs = [Example("a", "b", o) for o in ("x", "x", "val", "test")]
    splits = split_examples(exs, {"strategy": "by_origin", "val": ["val"], "test": ["test"]})
    assert [len(splits[k]) for k in ("train", "valid", "test")] == [2, 1, 1]


def test_prepare_writes_tokenized_files(tmp_path: Path):
    (tmp_path / "a.gly").write_text(GLY * 5 + GLY.replace("D21_:X1_", "N35_:X1_"))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "task": "mdc2translit", "work_dir": "work", "output_dir": "models", "tokenizers": {"src": "mdc"},
        "sources": [{"type": "jsesh", "path": "*.gly"}],
        "dedupe": "src", "split": {"strategy": "random", "val_ratio": 0.34, "test_ratio": 0.0},
    }))
    cfg = load_config(cfg_path)
    stats = prepare(cfg)
    assert stats["examples"] == 3
    src = (cfg.data_dir / "train.main.src").read_text().splitlines() + (cfg.data_dir / "valid.src").read_text().splitlines()
    assert "D21 : Q3 - D36 : X1" in src
    tgt = (cfg.data_dir / "train.main.tgt").read_text() + (cfg.data_dir / "valid.tgt").read_text()
    assert "S m s ▁ n b ▁ . f" in tgt


def test_tla_translit_to_mdc():
    assert tla_translit_to_mdc("ḥm.t =f") == "Hm .t .f"
    assert tla_translit_to_mdc("nḏ (w)di̯ r =s") == "nD wdi r .s"
    assert tla_translit_to_mdc("kꜣ(.PL) s.tPL Ꞽmn-m-ḥꜣ.t ẖr") == "kA s .t imn m HA .t Xr"


def test_tla_wordlist(tmp_path):
    f = tmp_path / "w.jsonl"
    f.write_text(json.dumps({"lemma": "=f", "spellings": [{"gardiner": "I9", "count": 10},
                                                          {"gardiner": "I9 Z4", "count": 1}]}) + "\n")
    assert [(r.mdc, r.translit) for r in load_wordlist(f, None, min_count=2)] == [("I9", ".f")]


def test_corpora_holdout_and_train_only(tmp_path: Path):
    (tmp_path / "a.gly").write_text(GLY + GLY.replace("D21_:Q3_", "N35_:Q3_"))
    (tmp_path / "pairs.txt").write_text("".join(f"A{i}\nw{i}\n" for i in range(1, 21)))
    (tmp_path / "holdout.txt").write_text("T18:S29-(D21*X1)-G36:D21\n")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "task": "mdc2translit", "work_dir": "work", "output_dir": "models",
        "sources": [
            {"type": "jsesh", "path": "a.gly", "corpus": "sinuhe"},
            {"type": "alternating", "path": "pairs.txt", "fields": ["mdc", "translit"], "corpus": "lexicon",
             "train_only": True},
        ],
        "corpus_weights": {"sinuhe": 3},
        "split": {"strategy": "random", "val_ratio": 0.5, "test_ratio": 0.0, "holdout": ["holdout.txt"]},
    }))
    cfg = load_config(cfg_path)
    stats = prepare(cfg)
    # The T18 record matches the holdout once layout is ignored ("signs" tokenizer).
    assert stats["sources"][0]["held_out"] == 2
    # Train-only lexicon pairs never reach valid.
    assert stats["splits"] == {"train": 21, "valid": 1, "test": 0}
    assert stats["train_corpora"] == {"sinuhe": 1, "lexicon": 20}
    assert set(train_files(cfg.data_dir)) == {"sinuhe", "lexicon"}

    onmt = yaml.safe_load(write_onmt_config(cfg).read_text())
    assert onmt["data"]["sinuhe"]["weight"] == 3
    assert "weight" not in onmt["data"]["lexicon"]
