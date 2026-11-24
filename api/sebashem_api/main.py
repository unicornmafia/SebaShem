"""SebaShem API: MdC hieroglyphs -> transliteration -> English."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from sebashem_api.engine import Seq2SeqModel, load_model, translate_marked
from sebashem_api.trilinear import render_trilinear, trilinear
from sebashem_api.schemas import (
    GlyRequest,
    MarkedLines,
    PipelineResponse,
    PipelineSegment,
    TextRequest,
    TranslationResponse,
    TranslationSegment,
    TransliterationResponse,
    TransliterationSegment,
)
from sebashem_api.settings import Settings
from sebashem_text.sentences import SentenceSplitter
from sebashem_text import (
    MarkedText,
    mdc_to_unicode,
    normalize_transliteration,
    split_mdc_segments_marked,
    split_text_markers,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    app.state.settings = settings
    app.state.mdc_model = load_model(settings, settings.mdc_model, "mdc2translit")
    app.state.translit_model = load_model(settings, settings.translit_model, "translit2eng")
    app.state.splitter = _load_splitter(settings)
    yield


def _load_splitter(settings: Settings) -> SentenceSplitter | None:
    path = settings.model_dir / settings.sentence_model
    if not (path / "sebashem.json").exists():
        logging.getLogger(__name__).warning("no sentence splitter at %s; .gly output keeps manuscript lines", path)
        return None
    return SentenceSplitter.load(path)


app = FastAPI(
    title="SebaShem",
    description="Transliterate Manuel de Codage (MdC) hieroglyphs and translate Middle Egyptian "
                "transliteration into English. Line markers (|R12 in MdC; (12), |(12), |B276 or <12> in "
                "transliteration) are not sentence breaks; they are echoed in the output where the "
                "models' attention places them.",
    version="0.2.0",
    lifespan=lifespan,
)


def _model(request: Request, attr: str) -> Seq2SeqModel:
    model = getattr(request.app.state, attr)
    if model is None:
        name = getattr(request.app.state.settings, attr.replace("_model", "") + "_model")
        raise HTTPException(503, f"model {name!r} is not available in {request.app.state.settings.model_dir}")
    return model


def _check_size(request: Request, text: str, segments: list) -> None:
    settings = request.app.state.settings
    if len(text) > settings.max_input_chars:
        raise HTTPException(413, f"input exceeds {settings.max_input_chars} characters")
    if not segments:
        raise HTTPException(422, "input contains no text to process")
    if len(segments) > settings.max_segments:
        raise HTTPException(413, f"input has {len(segments)} segments; the limit is {settings.max_segments}")


def _normalized(marked: MarkedText) -> MarkedText:
    return MarkedText([normalize_transliteration(p) for p in marked.pieces], marked.labels, " ")


def _transliterate(request: Request, body: TextRequest) -> list[tuple[TransliterationSegment, MarkedText]]:
    segments = split_mdc_segments_marked(body.text)
    _check_size(request, body.text, segments)
    results = translate_marked(_model(request, "mdc_model"), segments, beam_size=body.beam_size)
    out = []
    for seg, (hyp, translit) in zip(segments, results):
        glyphs, unknown = mdc_to_unicode(seg.text)
        marked = MarkedLines(mdc=seg.render(), transliteration=translit.render()) if seg.labels else None
        out.append((TransliterationSegment(mdc=seg.text, glyphs=glyphs, unknown_signs=unknown,
                                           line_markers=seg.labels, marked=marked,
                                           transliteration=hyp.text, transliteration_score=hyp.score),
                    translit))
    return out


def _translit_lines(text: str) -> list[MarkedText]:
    """One segment per non-empty line; a line holding only markers opens the next line."""
    lines, pending = [], []
    for raw in text.splitlines():
        marked = split_text_markers(raw, normalize_transliteration)
        if not marked.text:
            pending += marked.labels
            continue
        if pending:
            marked = MarkedText([""] * len(pending) + marked.pieces, pending + marked.labels, " ")
            pending = []
        lines.append(marked)
    if pending and lines:
        last = lines[-1]
        lines[-1] = MarkedText(last.pieces + [""] * len(pending), last.labels + pending, " ")
    return lines


def _pipeline(request: Request, body: TextRequest) -> list[PipelineSegment]:
    translit_model = _model(request, "translit_model")
    glyph_segments = _transliterate(request, body)
    results = translate_marked(translit_model, [_normalized(t) for _, t in glyph_segments], beam_size=body.beam_size)
    segments = []
    for (seg, _), (hyp, english) in zip(glyph_segments, results):
        data = seg.model_dump()
        if seg.marked:
            data["marked"]["translation"] = english.render()
        segments.append(PipelineSegment(**data, translation=hyp.text, translation_score=hyp.score))
    return segments


@app.get("/health")
def health(request: Request) -> dict:
    loaded = {"mdc2translit": request.app.state.mdc_model is not None,
              "translit2eng": request.app.state.translit_model is not None}
    return {"status": "ok" if all(loaded.values()) else "degraded", "models": loaded}


@app.get("/v1/models")
def models(request: Request) -> dict:
    return {key: (m.info() if (m := getattr(request.app.state, attr)) else None)
            for key, attr in (("mdc2translit", "mdc_model"), ("translit2eng", "translit_model"))}


@app.post("/v1/transliterate", response_model=TransliterationResponse)
def transliterate(body: TextRequest, request: Request) -> TransliterationResponse:
    """MdC hieroglyphs -> transliteration (plus a Unicode rendering of the signs)."""
    segments = [seg for seg, _ in _transliterate(request, body)]
    return TransliterationResponse(segments=segments, transliteration=" ".join(s.transliteration for s in segments))


@app.post("/v1/translate", response_model=TranslationResponse)
def translate(body: TextRequest, request: Request) -> TranslationResponse:
    """Transliteration (MdC ASCII or Egyptological Unicode, one sentence per line) -> English."""
    lines = _translit_lines(body.text)
    _check_size(request, body.text, lines)
    results = translate_marked(_model(request, "translit_model"), lines, beam_size=body.beam_size)
    segments = []
    for line, (hyp, english) in zip(lines, results):
        marked = MarkedLines(transliteration=line.render(), translation=english.render()) if line.labels else None
        segments.append(TranslationSegment(transliteration=line.text, translation=hyp.text,
                                           translation_score=hyp.score, line_markers=line.labels, marked=marked))
    return TranslationResponse(segments=segments, translation=" ".join(s.translation for s in segments))


@app.post("/v1/pipeline", response_model=PipelineResponse)
def pipeline(body: TextRequest, request: Request) -> PipelineResponse:
    """MdC hieroglyphs -> Unicode glyphs, transliteration and English translation, per segment."""
    segments = _pipeline(request, body)
    return PipelineResponse(
        segments=segments,
        glyphs=" ".join(s.glyphs for s in segments),
        transliteration=" ".join(s.transliteration for s in segments),
        translation=" ".join(s.translation for s in segments),
    )


@app.post("/v1/pipeline/gly", response_class=PlainTextResponse,
          responses={200: {"content": {"text/plain": {}}, "description": "A JSesh .gly file"}})
def pipeline_gly(body: GlyRequest, request: Request) -> PlainTextResponse:
    """A JSesh .gly file (or plain MdC) -> a trilinear .gly file, left to right.

    The glyphs are divided into sentences (which rarely start where manuscript lines
    do); each sentence keeps its original markup and line markers and is followed by a
    +t transliteration line and a +l translation line.
    """
    settings = request.app.state.settings
    if len(body.text) > settings.max_gly_chars:
        raise HTTPException(413, f"input exceeds {settings.max_gly_chars} characters")
    splitter = request.app.state.splitter if body.sentences else None
    doc, sentences = trilinear(body.text, _model(request, "mdc_model"), _model(request, "translit_model"),
                               splitter, beam_size=body.beam_size)
    if not sentences:
        raise HTTPException(422, "input contains no hieroglyphs")
    return PlainTextResponse(render_trilinear(doc, sentences), media_type="text/plain; charset=utf-8",
                             headers={"Content-Disposition": 'attachment; filename="sebashem.gly"'})
