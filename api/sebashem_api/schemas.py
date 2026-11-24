from __future__ import annotations

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="One segment per line. MdC input may also use '!' line ends, "
                                                     "and may be the contents of a JSesh .gly file.")
    beam_size: int | None = Field(None, ge=1, le=16)


class GlyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The contents of a JSesh .gly file, or plain MdC lines.")
    beam_size: int | None = Field(None, ge=1, le=16)
    sentences: bool = Field(True, description="Divide into sentences (false: keep the manuscript lines).")


class MarkedLines(BaseModel):
    """The segment's lines with its line markers echoed in place (only when it has markers)."""
    mdc: str | None = None
    transliteration: str | None = None
    translation: str | None = None


class GlyphSegment(BaseModel):
    mdc: str = Field(..., description="Cleaned Manuel de Codage, without line markers")
    glyphs: str = Field(..., description="Unicode hieroglyphs (sign sequence only, no quadrat layout)")
    unknown_signs: list[str] = Field(..., description="MdC codes with no Unicode equivalent")
    line_markers: list[str] = Field(default_factory=list, description="Line markers in the segment, e.g. |B276")
    marked: MarkedLines | None = None


class TransliterationSegment(GlyphSegment):
    transliteration: str
    transliteration_score: float | None


class TranslationSegment(BaseModel):
    transliteration: str = Field(..., description="Input after normalization to the training corpus format")
    translation: str
    translation_score: float | None
    line_markers: list[str] = Field(default_factory=list, description="Line markers in the input, e.g. (12)")
    marked: MarkedLines | None = None


class PipelineSegment(TransliterationSegment):
    translation: str
    translation_score: float | None


class TransliterationResponse(BaseModel):
    segments: list[TransliterationSegment]
    transliteration: str


class TranslationResponse(BaseModel):
    segments: list[TranslationSegment]
    translation: str


class PipelineResponse(BaseModel):
    segments: list[PipelineSegment]
    glyphs: str
    transliteration: str
    translation: str
