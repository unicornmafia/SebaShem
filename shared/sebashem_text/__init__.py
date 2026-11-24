"""Text handling shared by the SebaShem trainer and API.

Anything that affects what a model sees at train time (cleaning, normalization,
tokenization) lives here so that the API feeds models exactly what they were
trained on.
"""

from sebashem_text.mdc import (
    canonical_sign,
    clean_mdc,
    hieroglyphs_to_mdc,
    mdc_to_unicode,
    sign_codes,
    split_mdc_segments,
    split_mdc_segments_marked,
    tokenize_mdc,
)
from sebashem_text.markers import MarkedText, split_text_markers
from sebashem_text.translit import (
    infer_respelling,
    normalize_translation,
    normalize_transliteration,
    respell,
    scholarly_transliteration,
    unicode_to_mdc_translit,
)
from sebashem_text.tokenizers import get_tokenizer, TOKENIZERS

__all__ = [
    "canonical_sign",
    "clean_mdc",
    "hieroglyphs_to_mdc",
    "sign_codes",
    "split_mdc_segments",
    "split_mdc_segments_marked",
    "MarkedText",
    "split_text_markers",
    "tokenize_mdc",
    "mdc_to_unicode",
    "normalize_transliteration",
    "normalize_translation",
    "unicode_to_mdc_translit",
    "infer_respelling",
    "respell",
    "scholarly_transliteration",
    "get_tokenizer",
    "TOKENIZERS",
]
