# SebaShem

Middle Egyptian machine translation, converted from the
[EgyptianTranslation](../EgyptianTranslation) notebooks into:

1. **`api/`**: a Dockerized FastAPI service that turns Manuel de Codage (MdC)
   hieroglyphs into Unicode glyphs, a transliteration, and an English translation.
2. **`trainer/`**: a local (non-containerized) trainer that builds datasets from
   local files, trains OpenNMT-py models, and exports them for the API.

```
MdC "D21:Q3-D36:X1-F4:D36"
   │  mdc2translit model (sign sequence -> characters)
   ▼
transliteration "iri pat HAti a"
   │  translit2eng model (the notebook's transformer)
   ▼
English "prince regent and count"
```

| Directory | What it is |
|---|---|
| `shared/` | `sebashem_text`: MdC cleaning/tokenizing/Unicode rendering, transliteration normalization, tokenizers. Used by both the trainer and the API, so models see the same input at train and serve time. |
| `trainer/` | `sebashem-train` CLI, corpus parsers ported from `egyptiandatasetgenerator.ipynb`, configs. |
| `api/` | FastAPI app and Dockerfile. Serves CTranslate2 models only, so the image doesn't need torch. |
| `models/` | Exported models (`models/<name>/` = CTranslate2 files + `sebashem.json`). The trainer writes here; the API reads from here. The current models are committed with Git LFS (`git lfs install` before cloning, or `git lfs pull` after); retraining overwrites them. |

## Quick start

```bash
# 1. Trainer environment (OpenNMT-py 3.5 needs Python 3.10/3.11)
python3.11 -m venv trainer/.venv
trainer/.venv/bin/pip install -e shared -e trainer

# 2a. Reuse the already trained notebook model (BLEU 46.98 on its test set)
trainer/.venv/bin/sebashem-train export \
    --checkpoint ../EgyptianTranslation/onmt-files/run/model_step_3000.pt \
    --task translit2eng --output-dir models
#  ...or 2b. retrain it from the raw texts
trainer/.venv/bin/sebashem-train train trainer/configs/translit2eng.yaml

# 3. Train the hieroglyph -> transliteration model
trainer/.venv/bin/sebashem-train train trainer/configs/mdc2translit.yaml

# 4. Train the sentence splitter (for trilinear .gly output)
trainer/.venv/bin/sebashem-train sentences \
    --texts ../EgyptianTranslation/texts/nederhof-texts ../EgyptianTranslation/texts/grammar-texts \
    --gly ~/Downloads/Sinuhe_tm_sebasesh3_sebasesh.gly \
    --tla ../SebaSebaDictionary/dictionaries/train.jsonl --eval-exclude sinuhe

# 5. Serve
docker compose up --build
curl -s localhost:8000/v1/pipeline -H 'content-type: application/json' \
     -d '{"text": "D21:Q3-D36:X1-F4:D36"}'
```

Interactive API docs: http://localhost:8000/docs

## API

Every endpoint takes `{"text": "...", "beam_size": 5}` (`beam_size` is optional). Multi-line
input is processed one segment per line.

| Endpoint | Input | Output per segment |
|---|---|---|
| `POST /v1/pipeline` | MdC | `mdc`, `glyphs`, `unknown_signs`, `transliteration`, `translation` (+ scores) |
| `POST /v1/pipeline/gly` | JSesh `.gly` or MdC | A trilinear JSesh `.gly` file, one sentence per group (see below) |
| `POST /v1/transliterate` | MdC | `mdc`, `glyphs`, `unknown_signs`, `transliteration` |
| `POST /v1/translate` | transliteration | normalized `transliteration`, `translation` |
| `GET /health` | | `ok` / `degraded` and which models loaded |
| `GET /v1/models` | | each model's manifest (task, tokenizers, test metrics) |

`/v1/pipeline` is the endpoint that produces all three. Each response also has the
segments joined into top-level `glyphs`, `transliteration` and `translation` strings.

### Trilinear `.gly` output

`/v1/pipeline/gly` takes `{"text": "<.gly contents>", "sentences": true}` and returns a
JSesh file in the layout of the annotated Sinuhe editions: for each sentence a glyph
line, a `+t` transliteration line and a `+l` translation line.

```bash
jq -Rs '{text: .}' Sinuhe.gly | curl -s localhost:8000/v1/pipeline/gly \
    -H 'content-type: application/json' --data @- -o Sinuhe.trilinear.gly
```

- **Sentences, not manuscript lines.** Each line is transliterated, the sentence
  splitter (`models/sentences`) picks the words that start sentences in the running
  text, attention maps them back into the glyphs, and a sign-level model moves each
  cut to the best gap within two signs. Each glyph sentence is then transliterated
  again on its own and translated. `"sentences": false` keeps one group per
  manuscript line.
- **Glyph lines keep their markup**: line markers (`|R12`, `|goto B1`), cartouches and
  `[[...]]` variants stay whole. Restored text (`[[...]]` lines), shading (`#b`/`#e`)
  and red (`$r`/`$b`) still open at a cut are closed and reopened around it.
- **`+t` / `+l` carry only text**, no line markers.
- **Always left to right** (`++JSesh_page_direction LEFT_TO_RIGHT`), whatever the source's direction.
- **Transliteration style** is the scholarly one (see below).
- Existing `+t` / `+l` blocks in the input are replaced; `+b` / `+i` blocks before the
  glyphs are kept.

Sentence-division accuracy, against the user's own division of the held-out last 20%
of `Sinuhe_tm_sebasesh3_sebasesh.gly` (re-cut into ~27-sign lines): about half the
sentence starts are exact (F1 ≈ 0.52–0.55) and about 0.62–0.65 are within two signs,
versus 0.37 for line starts. The splitter itself, on gold transliteration, scores F1
0.92 with Nederhof's Sinuhe manuscripts in training and 0.74 without them.

MdC input can be plain MdC or the contents of a JSesh `.gly` file. Segments split on
newlines and `!`. JSesh headers (`++...`), text blocks (`+t`, `+l`, `+b`), line labels
(`|R1`), layout commands (`$r`, `#b`, `\shading`, `{{x,y,z}}`), critical-apparatus
citations (`[[...]]`) and damage brackets are stripped. Signs and the structural
operators `- : * & ( ) < >` are kept.

### Line markers (`/v1/pipeline`)

Manuscript line markers are read in the input and echoed in the response:
- `|R12` or `|B276` in MdC;
- `(12)`, `|(12)`, `|B276` or `<12>` in transliteration.

A marker is not a sentence break. A sentence that runs across `|R12` stays one
segment and is translated as a whole, and the marker is put back into all three
lines. A marker with no signs after it in its segment (`|R13-!` on a line of its
own) opens the next segment instead.

```
|R11-M17-S29-A1:3-|R12-D21-N16-!
+t|R11 ... |R12 ...+s-!
+l|R11 ... |R12 ...+s-!
```

In `/v1/pipeline/gly` markers stay only in the glyph lines. The models never see the markers. Each segment's response has `line_markers` and a
`marked` object holding the `mdc`, `transliteration` and `translation` lines with the
markers in place (`null` when there are none). In the MdC line the marker's position
is exact. In the transliteration and translation it is placed using the model's
attention: the cut point that leaves the fewest output words on the wrong side of
the marker, only between words.

Placement accuracy for `translit2eng`, measured on Nederhof's own interior markers in
61 held-out sentences: 66% exactly right, 90% within one word. For `mdc2translit`,
measured on 34 pairs of held-out Sinuhe lines joined at a known boundary: 85% exact,
94% within one word (decoder layer −2, all heads). The attention comes from
decoder layer −4, all heads averaged (`export: {alignment_layer, alignment_heads}` in
the config, or `--alignment-layer` / `--alignment-heads` on `export`). With the
converter's default (last layer, first head) the figures were 46% and 72%. Placement
can't be right when the model leaves out or makes up words around the marker.

Bare digits in MdC (`3`, `4\R90`) are numeral and stroke signs, not markers. Markers
inside `[[...]]` citations (`|S 1`, `|B3, 4`) refer to other manuscripts and are
dropped with the citation.

### Transliteration input

Transliteration input for `/v1/translate` may be MdC ASCII (`Dd.n=f`) or Egyptological
Unicode (`ḏd.n=f`). It is normalized to the training corpus format (`Dd .n .f`).

Example:

```bash
curl -s localhost:8000/v1/translate -H 'content-type: application/json' \
     -d '{"text": "nsw-bjtj aA-xpr-kA-ra\nḏd.n=f n ḥm=f"}'
# -> "the king of upper and lower egypt aakheperkare", "and he said to his majesty"
```

Configuration (environment variables):

| Variable | Default | |
|---|---|---|
| `MODEL_DIR` | `/models` in Docker, `models` otherwise | directory holding the model dirs |
| `MDC_MODEL` / `TRANSLIT_MODEL` | `mdc2translit` / `translit2eng` | model dir names (e.g. `MDC_MODEL=mdc2translit-sinuhe` for the baseline) |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `COMPUTE_TYPE` | `default` | CTranslate2 compute type |
| `INTER_THREADS` / `INTRA_THREADS` | `1` / `0` | CTranslate2 parallelism |
| `SENTENCE_MODEL` | `sentences` | sentence splitter dir for `/v1/pipeline/gly` (without it, groups are manuscript lines) |
| `MAX_INPUT_CHARS` / `MAX_SEGMENTS` | `20000` / `200` | request limits |
| `MAX_GLY_CHARS` | `1000000` | input limit for `/v1/pipeline/gly` |

A missing model doesn't stop the service. Its endpoints return 503 and `/health`
reports `degraded`. So `/v1/translate` works with only `translit2eng` present.

`docker-compose.yml` mounts `./models` read-only. To bake models into the image,
uncomment the `COPY models/` line in `api/Dockerfile`.

Run the API without Docker:

```bash
python3.11 -m venv api/.venv && api/.venv/bin/pip install -e shared -e 'api[test]'
MODEL_DIR=models api/.venv/bin/uvicorn sebashem_api.main:app --reload
```

## Trainer

```
sebashem-train prepare  CONFIG               # build work_dir/data/{train,valid,test}.{src,tgt}
sebashem-train train    CONFIG [--skip-prepare] [--resume] [--no-export] [--no-eval]
sebashem-train export   --config CONFIG [--checkpoint X.pt]
sebashem-train export   --checkpoint X.pt --task translit2eng|mdc2translit [--name N] [--output-dir models]
sebashem-train evaluate --model models/NAME (--config CONFIG | --src F --ref F) [--save]
sebashem-train word2vec --src train.src --tgt train.tgt [--mono corpus] --out-dir DIR
sebashem-train sentences --gly F.gly [...] [--texts DIR ...] [--tla train.jsonl] [--eval-exclude NAME] [--out models/sentences]
```

`sentences` trains the sentence splitter: a hashed n-gram logistic regression over
transliteration words, from Nederhof / grammar texts (one sentence per aligned line,
read in order) and sentence-divided JSesh editions, plus a sign-level refiner trained
on the editions and TLA. The end of the first `--gly` file is held out to tune the
threshold and report precision / recall (also without texts matching
`--eval-exclude`); the shipped model then trains on everything.

`train` runs the whole pipeline: it prepares the data, runs `onmt_build_vocab` and
`onmt_train` with a generated YAML (`work_dir/onmt.yaml`), converts the last checkpoint
to CTranslate2 (int8 by default) in `models/<name>/`, then scores the test split with
sacrebleu (BLEU, chrF, exact match). The scores go into `models/<name>/sebashem.json`.
Training uses CUDA when available, otherwise CPU.

### Configs

| Config | Task | Data |
|---|---|---|
| `configs/translit2eng.yaml` | transliteration -> English | Rebuilt from the raw texts like `egyptiandatasetgenerator.ipynb` (Nederhof texts, grammars, Pyramid Texts), with the notebook's file-level val/test holdout. Notebook hyperparameters. |
| `configs/translit2eng-compiled.yaml` | transliteration -> English | The committed `compiled_corpora/aligned_{train,val,test}` files, which are exactly what `model_step_3000.pt` was trained on. |
| `configs/mdc2translit.yaml` | MdC -> transliteration | The TLA sentence corpus and the sentence-divided Sinuhe JSesh editions, plus dictionary spellings as word pairs (all from `../SebaSebaDictionary` and `~/Downloads`), in the scholarly style. This is the model the API uses. About 45 min on CPU. |
| `configs/mdc2translit-sinuhe.yaml` | MdC -> transliteration | The first baseline: only the Sinuhe JSesh editions (`~/Downloads/*sebasesh*.gly`, `Sinuhe_november_2026.gly`) and `sinuhe/sinuhe_glyphs_transliteration.txt`. Exported as `models/mdc2translit-sinuhe`. |

Relative paths in a config are resolved against the config file. `~`, `$VARS` and
globs work. The sections are:

```yaml
task: mdc2translit            # or translit2eng
name: mdc2translit            # -> models/mdc2translit
work_dir: ../work/mdc2translit
output_dir: ../../models
sources:                      # any mix of the types below
  - {type: jsesh, path: ~/data/*.gly, corpus: sinuhe, dedupe: src}
  - {type: dictionary, path: entries.json, corpus: lexicon, train_only: true}
dedupe: pair                  # pair | src | none; a source can override it
corpus_weights: {sinuhe: 1, lexicon: 2}   # OpenNMT sampling weight per corpus
split: {strategy: random, val_ratio: 0.1, test_ratio: 0.1, seed: 1234,
        holdout: [../eval/sinuhe-holdout.input]}   # src lines never used at all
#  or {strategy: by_origin, val: [WestcarTr.txt], test: [DisputeTr.txt]}
tokenizers: {src: signs, tgt: chars}   # optional; task defaults shown
translit_style: corpus        # mdc2translit target: corpus (Hm .f) | scholarly (Hm=f)
device: auto                  # auto | cpu | gpu
onmt: {train_steps: 3000}     # any OpenNMT-py option, merged over task defaults
export: {quantization: int8}
decode: {beam_size: 5, max_decoding_length: 512}
```

### Transliteration style

`translit_style: scholarly` trains `mdc2translit` to write whole words: suffix
pronouns and stative endings with `=` (`sDm.n=f`, `Hm=k`, `iy.kw`), other endings
with `.` (`.n`, `.tw`, `.t`, `.w`), always `i` (not `j`) and `s` (not `z`). The TLA
source turns its `=x` pronoun tokens and stative glosses (`\res`) into this; JSesh
editions are used only if their `+t` lines already use `=`. The translator still
reads the corpus layout: the API converts with `normalize_transliteration` and the
translator's `input_respell`.

### Source types

| `type` | Fields it provides | Options |
|---|---|---|
| `nederhof` | translit, translation | `path`: dir of `*Tr.txt` or one file |
| `grammar` | translit, translation | `path`: dir of grammar `.txt` (`%al` / `%tr` paragraphs) |
| `jsesh` | mdc, translit, translation | `path`: `.gly` file, dir or glob. Glyph lines are grouped with the `+t` / `+l` lines that follow them. `+s<mdc>+s-!` glyph blocks also work. |
| `csv` | any | `path`, `columns: {translit: ColName, translation: ColName, mdc: ColName}`, `origin`, `origin_column`, `delimiter` |
| `parallel` | any | `files: {translit: a.txt, translation: b.txt}`, line-aligned |
| `alternating` | any | `path`, `fields: [mdc, translit]`: one line per field, repeating |
| `tla` | mdc, translit (translation if `translation: true`) | `path`: TLA sentences JSONL (`SebaSebaDictionary/dictionaries/train.jsonl`). Unicode hieroglyphs and transliteration are converted to MdC / MdC ASCII. |
| `tla_wordlist` | mdc, translit | `path`: `tla_wordlist.jsonl`, one pair per attested spelling of each lemma. `min_count` (default 1) |
| `dictionary` | mdc, translit | `path`: SebaSebaDictionary `entries.json` (Gardiner spelling + transliteration per headword) |

Any source can also set:
- `corpus`: the group it trains in (default `main`). Each corpus is a separate
  OpenNMT training corpus, sampled by `corpus_weights`, and test scores are
  reported per corpus.
- `train_only: true`: never put it in valid/test. Used for dictionary word pairs.
- `dedupe`: override the config-level setting.

Duplicates and `holdout` lines are matched on the tokenized source, so with the
`signs` tokenizer two MdC lines with the same signs but different layout count as
the same.

The `nederhof` and `grammar` sources use the notebook's own preprocessing regexes.
The `tla*` sources lowercase TLA's name capitals and drop its `.PL`/`.DU` markers and
`ʾ`/`i̯` modifiers before converting to MdC ASCII. Every
other source is normalized with `sebashem_text` unless it sets `normalize: false`. That
is the same normalization the API applies to its input. Each task uses the two fields
it needs (`translit2eng`: translit -> translation, `mdc2translit`: mdc -> translit) and
skips records that lack either one.

### Semi-supervised (word2vec) variant

This is a port of `Generate_Word2Vec.ipynb` + `SemiSupervised.ipynb`. In that work it
scored slightly lower than the supervised model (41.78 vs 42.22 BLEU).

```bash
trainer/.venv/bin/pip install -e 'trainer[word2vec]'
trainer/.venv/bin/sebashem-train prepare trainer/configs/translit2eng.yaml
trainer/.venv/bin/sebashem-train word2vec \
    --src trainer/work/translit2eng/data/train.src --tgt trainer/work/translit2eng/data/train.tgt \
    --mono ../EgyptianTranslation/compiled_corpora/egyptian_monolingual.csv \
    --out-dir trainer/work/w2v
```

Then add `src_embeddings`, `tgt_embeddings` and `embeddings_type: word2vec` to the
config's `onmt:` block (see the comment in `translit2eng.yaml`). `--mono` also accepts
the downloaded HTML pages directory (`texts/monolingual-texts`).

## Notes and limitations

- **MdC -> transliteration results.** Scores on 68 Sinuhe lines no model trains on
  (`trainer/eval/sinuhe-holdout.*`) and on 498 TLA test sentences:

  | Model | Sinuhe holdout BLEU / chrF | TLA test BLEU / chrF |
  |---|---|---|
  | `mdc2translit-sinuhe` (Sinuhe only) | 22.8 / 42.1 | 7.7 / 20.3 |
  | `mdc2translit-corpus` (TLA + Sinuhe + dictionaries) | 35.4 / 52.6 | 53.9 / 61.8 |
  | `mdc2translit` (same, scholarly style) | 35.4 / 52.0 ¹ | 52.3 / 63.2 |

  ¹ on 39 of the holdout lines with scholarly references
  (`trainer/eval/sinuhe-holdout-scholarly.*`). The model still sometimes repeats
  words (`Hswt Hswt Hswt`), and the English translation is the weakest stage.
- **Training stability.** The first combined run (learning rate 2, warmup 800, no
  gradient clipping) diverged after step 500. The config now uses learning rate 1,
  warmup 2000 and `max_grad_norm: 1`, and exports the checkpoint with the best
  validation accuracy (`export: {checkpoint: best}`).
- **MdC -> transliteration training data.** About 9k TLA sentences (Old and Middle
  Kingdom, many texts), about 540 Sinuhe lines, and 60k dictionary word pairs. The
  model reads only the sign sequence (`signs` tokenizer). JSesh layout (`:` `*` `( )`)
  is ignored because TLA and the dictionaries don't record it. Phonetic codes such as
  `ra` are mapped to their Gardiner sign. `trainer/eval/sinuhe-holdout.*` holds 68
  Sinuhe lines that no model trains on, for comparing models.
- **TLA data licensing and quality.** The TLA corpus is CC BY-SA 4.0, so credit the
  Thesaurus Linguae Aegyptiae when you publish models trained on it. The English in
  `train_en.jsonl` is MarianMT output from TLA's German, so the `tla` source leaves it
  out unless `translation: true` is set.
- **Transliteration conventions differ between the two models.** The translator's
  training data writes `j` for the reed leaf and never uses `z`. The hieroglyph model
  writes TLA/JSesh-style `i` and `z`. So a translator export records an
  `input_respell` mapping in its manifest, inferred from its training data (here
  `i→j, z→s`). The API and the evaluator apply it before translating. Without it,
  27.5% of the hieroglyph model's output words were unknown to the translator. On
  unseen Sinuhe sentences, translation went from BLEU 6.5 / chrF 24.4 to 15.3 / 31.6
  with it.
- `glyphs` is the plain sign sequence. It doesn't reproduce quadrat layout (`:` / `*`
  stacking). MdC phonetic codes (`ra`, `nTr`, `Htp`...) are mapped for common signs.
  Numerals and codes outside Unicode 14's Egyptian Hieroglyphs block are listed in
  `unknown_signs`. They are still passed to the model.
- Exporting an OpenNMT-py checkpoint needs `torch.load(weights_only=False)`, because the
  checkpoint pickles its options. Only export checkpoints you trust. The exporter also
  rewrites `self_attn_type: scaled-dot-flash` to `scaled-dot` (same computation), which
  CTranslate2 requires. New configs train with `scaled-dot` directly.
- `pyonmttok` is pinned to 1.37.1. The 1.38 macOS wheel fails to load
  `libOpenNMTTokenizer.dylib`.

## Tests

```bash
api/.venv/bin/python -m pytest shared/tests api/tests   # API tests use fake models; test_real_models uses ./models
trainer/.venv/bin/python -m pytest trainer/tests
```
