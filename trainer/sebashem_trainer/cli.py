"""``sebashem-train`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sebashem_text import infer_respelling
from sebashem_trainer import dataset, onmt_runner
from sebashem_trainer.config import TASKS, load_config
from sebashem_trainer.evaluate import evaluate
from sebashem_trainer.export import export_checkpoint, update_manifest


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_prepare(args) -> None:
    cfg = load_config(args.config)
    _print(dataset.prepare(cfg))
    print(f"wrote {cfg.data_dir}")


def _input_respelling(cfg) -> dict:
    """For transliteration input, the respelling (e.g. i -> j) into the training data's convention."""
    if cfg.task.src_field != "translit":
        return {}
    lines = []
    for src, _ in dataset.train_files(cfg.data_dir).values():
        lines += src.read_text().splitlines()
    return infer_respelling(lines)


def _export_and_eval(cfg, checkpoint: Path, stats: dict | None, run_eval: bool) -> None:
    respelling = _input_respelling(cfg)
    if respelling:
        print(f"input respelling into the training convention: {respelling}", flush=True)
    model_dir = export_checkpoint(
        checkpoint, cfg.model_dir, task=cfg.task.name, src_tokenizer=cfg.src_tokenizer,
        tgt_tokenizer=cfg.tgt_tokenizer, decode=cfg.decode, quantization=cfg.export.get("quantization"),
        extra={"data": stats, "config": str(cfg.path), "input_respell": respelling,
               "translit_style": cfg.translit_style},
        alignment_layer=cfg.export.get("alignment_layer"), alignment_heads=cfg.export.get("alignment_heads"),
    )
    if run_eval:
        metrics = _evaluate_test_split(cfg, model_dir)
        if metrics:
            update_manifest(model_dir, metrics=metrics)
            _print({"metrics": metrics})
    print(f"model ready: {model_dir}")


def _evaluate_test_split(cfg, model_dir: Path, beam_size: int | None = None) -> dict | None:
    """Score the prepared test split, overall and per corpus when there are several."""
    test_ref = cfg.data_dir / "test.ref"
    if not test_ref.exists() or not test_ref.stat().st_size:
        return None
    inputs = (cfg.data_dir / "test.input").read_text().splitlines()
    refs = test_ref.read_text().splitlines()
    corpus_file = cfg.data_dir / "test.corpus"
    corpora = corpus_file.read_text().splitlines() if corpus_file.exists() else ["main"] * len(refs)

    metrics = evaluate(model_dir, inputs, refs, pred_path=cfg.work_dir / "eval" / "test.pred", beam_size=beam_size)
    if len(set(corpora)) > 1:
        metrics["by_corpus"] = {}
        for name in sorted(set(corpora)):
            idx = [i for i, c in enumerate(corpora) if c == name]
            metrics["by_corpus"][name] = evaluate(model_dir, [inputs[i] for i in idx], [refs[i] for i in idx],
                                                  beam_size=beam_size)
    return metrics


def _checkpoint_to_export(cfg) -> Path:
    if cfg.export.get("checkpoint", "last") == "best":
        return onmt_runner.best_checkpoint(cfg)
    return onmt_runner.latest_checkpoint(cfg)


def cmd_train(args) -> None:
    cfg = load_config(args.config)
    stats_path = cfg.data_dir / "stats.json"
    if args.skip_prepare and stats_path.exists():
        stats = json.loads(stats_path.read_text())
    else:
        stats = dataset.prepare(cfg)
        _print(stats)

    onmt_yaml = onmt_runner.write_onmt_config(cfg)
    train_from = None
    if args.resume:
        train_from = onmt_runner.latest_checkpoint(cfg)
    else:
        onmt_runner.build_vocab(onmt_yaml)
    onmt_runner.train(onmt_yaml, train_from=train_from)

    if not args.no_export:
        _export_and_eval(cfg, _checkpoint_to_export(cfg), stats, run_eval=not args.no_eval)


def cmd_export(args) -> None:
    if args.config:
        cfg = load_config(args.config)
        checkpoint = Path(args.checkpoint) if args.checkpoint else _checkpoint_to_export(cfg)
        stats_path = cfg.data_dir / "stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
        _export_and_eval(cfg, checkpoint, stats, run_eval=not args.no_eval)
        return

    if not (args.checkpoint and args.task):
        sys.exit("export without --config needs --checkpoint and --task")
    task = TASKS[args.task]
    name = args.name or task.name
    model_dir = export_checkpoint(
        Path(args.checkpoint), Path(args.output_dir) / name, task=task.name,
        src_tokenizer=args.src_tokenizer or task.src_tokenizer, tgt_tokenizer=args.tgt_tokenizer or task.tgt_tokenizer,
        decode={"beam_size": 5, "max_decoding_length": 256},
        quantization=None if args.quantization == "none" else args.quantization,
        alignment_layer=args.alignment_layer, alignment_heads=args.alignment_heads,
    )
    print(f"model ready: {model_dir}")


def cmd_evaluate(args) -> None:
    model_dir = Path(args.model)
    if args.config:
        metrics = _evaluate_test_split(load_config(args.config), model_dir, beam_size=args.beam_size)
        if metrics is None:
            sys.exit("the config's test split is empty; run prepare first")
    elif args.src and args.ref:
        metrics = evaluate(model_dir, Path(args.src).read_text().splitlines(), Path(args.ref).read_text().splitlines(),
                           pred_path=Path(args.pred) if args.pred else None, beam_size=args.beam_size)
    else:
        sys.exit("evaluate needs --config or both --src and --ref")
    if args.save:
        update_manifest(model_dir, metrics=metrics)
    _print(metrics)


def cmd_word2vec(args) -> None:
    from sebashem_trainer.word2vec import train_embeddings

    written = train_embeddings(Path(args.src), Path(args.tgt), Path(args.mono) if args.mono else None,
                               Path(args.out_dir), vector_size=args.vector_size)
    _print({k: str(v) for k, v in written.items()})


def cmd_sentences(args) -> None:
    from sebashem_trainer.segmenter import train_segmenter

    metrics = train_segmenter([Path(d).expanduser() for d in args.texts], [Path(p).expanduser() for p in args.gly],
                              Path(args.out), eval_exclude=args.eval_exclude,
                              tla_path=Path(args.tla).expanduser() if args.tla else None)
    _print({"model": args.out, "metrics": metrics})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sebashem-train", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="build tokenized train/valid/test files from the config's sources")
    p.add_argument("config")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="prepare, build vocab, train, export to the models dir and evaluate")
    p.add_argument("config")
    p.add_argument("--skip-prepare", action="store_true", help="reuse the existing work_dir/data")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint (raise train_steps)")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export", help="convert an OpenNMT-py checkpoint to a CTranslate2 model for the API")
    p.add_argument("--config", help="export the config's latest (or --checkpoint) checkpoint")
    p.add_argument("--checkpoint", help="OpenNMT-py .pt file")
    p.add_argument("--task", choices=sorted(TASKS), help="required without --config")
    p.add_argument("--name", help="model directory name (default: the task name)")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--src-tokenizer")
    p.add_argument("--tgt-tokenizer")
    p.add_argument("--quantization", default="int8", help="int8, int8_float32, float16, ... or none")
    p.add_argument("--alignment-layer", type=int, help="decoder layer whose attention places line markers, e.g. -4")
    p.add_argument("--alignment-heads", type=int, help="heads to average for it (0 = all)")
    p.add_argument("--no-eval", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("evaluate", help="BLEU/chrF of an exported model")
    p.add_argument("--model", required=True, help="exported model directory")
    p.add_argument("--config", help="use the config's prepared test set")
    p.add_argument("--src", help="plain-text source lines")
    p.add_argument("--ref", help="plain-text reference lines")
    p.add_argument("--pred", help="write predictions here")
    p.add_argument("--beam-size", type=int)
    p.add_argument("--save", action="store_true", help="store the metrics in the model's sebashem.json")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("word2vec", help="train word2vec embeddings for semi-supervised training")
    p.add_argument("--src", required=True, help="tokenized source training file")
    p.add_argument("--tgt", required=True, help="tokenized target training file")
    p.add_argument("--mono", help="monolingual corpus: .csv, plain lines, or a directory of HTML pages")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--vector-size", type=int, default=512)
    p.set_defaults(func=cmd_word2vec)

    p = sub.add_parser("sentences", help="train the sentence splitter on sentence-divided transliteration")
    p.add_argument("--texts", nargs="*", default=[], help="Nederhof / grammar text directories, read in order")
    p.add_argument("--gly", nargs="+", required=True,
                   help="sentence-divided JSesh editions; the end of the first is held out for scoring")
    p.add_argument("--out", default="models/sentences")
    p.add_argument("--eval-exclude", help="also score a model trained without texts whose name contains this")
    p.add_argument("--tla", help="TLA sentences (train.jsonl) for the sign-level refiner")
    p.set_defaults(func=cmd_sentences)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
