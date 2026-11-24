from sebashem_text import (
    clean_mdc,
    get_tokenizer,
    hieroglyphs_to_mdc,
    mdc_to_unicode,
    normalize_translation,
    normalize_transliteration,
    sign_codes,
    split_mdc_segments,
    tokenize_mdc,
)
from sebashem_text import split_mdc_segments_marked, split_text_markers
from sebashem_text.mdc import gardiner_to_unicode


def test_clean_mdc_strips_jsesh_markup():
    raw = r"|R1-$r-D21_:Q3_-D36_:X1_-U7\shading1234:D36-[[-|G 1-Aa8_:(X1_:O49_)*Z3A-]]-I3:I3-$b-!"
    assert clean_mdc(raw) == "D21:Q3-D36:X1-U7:D36-I3:I3"


def test_clean_mdc_drops_orphaned_operators():
    assert clean_mdc(r"D21_-[{*R5\*}]:[&*R5*&]:N25_-V28") == "D21-R5:R5:N25-V28"


def test_split_segments_reads_gly_files():
    gly = "++JSesh_Info 1.0 +s\n+bTitle+s-!\nD21:Q3-!X1\n+tiri pat+s-!\n+lprince+s-!\n+sF4:D36+s-!\n"
    assert split_mdc_segments(gly) == ["D21:Q3", "X1", "F4:D36"]


def test_tokenize_and_render():
    assert tokenize_mdc("Aa1:X1-<-N5->") == ["Aa1", ":", "X1", "-", "<", "-", "N5", "-", ">"]
    glyphs, unknown = mdc_to_unicode("ra-D21:Q3-A363")
    assert glyphs == "\U000131F3\U0001308B\U000132AA"
    assert unknown == ["A363"]


def test_normalize_transliteration_matches_corpus_format():
    assert normalize_transliteration("sDm.n=f (r) pr-aA") == "sDm .n .f r pr aA"
    assert normalize_transliteration("ḏd.n=f n ḥm=f") == "Dd .n .f n Hm .f"
    assert normalize_transliteration("Sms[w sA-n]ht") == "Smsw sA nht"


def test_normalize_translation():
    assert normalize_translation("'Listen to me, count!'") == "listen to me count"


def test_tokenizer_round_trips():
    chars = get_tokenizer("chars")
    assert chars.encode("sDm .f") == ["s", "D", "m", "▁", ".", "f"]
    assert chars.decode(chars.encode("sDm  .f")) == "sDm .f"
    mdc = get_tokenizer("mdc")
    assert mdc.decode(mdc.encode("D21:Q3-X1")) == "D21:Q3-X1"


def test_sign_codes_drop_layout_and_canonicalize():
    assert sign_codes("D21:Q3-ra-(U1a*AA1)-<-N5->") == ["D21", "Q3", "N5", "U1A", "Aa1", "N5"]
    assert get_tokenizer("signs").encode("ra*X1") == ["N5", "X1"]


def test_hieroglyphs_to_mdc():
    f, a1, m = (gardiner_to_unicode(c) for c in ("I9", "A1", "G17"))
    text = f"{f}{a1} <g>M12B</g>{m}\uFE02 <g>(unidentified)</g>"
    assert hieroglyphs_to_mdc(text) == "I9-A1-M12B-G17"


def test_line_markers_stay_inside_segments():
    gly = "|R11-M17_-S29_-[[-|S 1-D1-]]-A1_:3-|R12-D21_-N16-!\n|R13-!\nG39-!\nX1-|R14-!"
    segs = split_mdc_segments_marked(gly)
    # Citation references (|S 1) are dropped; |R13 opens the next segment; a trailing
    # marker at the very end stays on the last segment.
    assert [(s.text, s.labels, s.render()) for s in segs] == [
        ("M17-S29-A1:3-D21-N16", ["|R11", "|R12"], "|R11-M17-S29-A1:3-|R12-D21-N16"),
        ("G39", ["|R13"], "|R13-G39"),
        ("X1", ["|R14"], "X1-|R14"),
    ]
    assert segs[0].prefix(1) == "M17-S29-A1:3"


def test_text_markers():
    m = split_text_markers("ist r=f mSa |(12) r tA <13> timHw (14)", normalize_transliteration)
    assert m.labels == ["|(12)", "<13>", "(14)"]
    assert m.text == "ist r .f mSa r tA timHw"
    assert m.render() == "ist r .f mSa |(12) r tA <13> timHw (14)"
    assert normalize_transliteration("mSa |(12) r |B276 tA") == "mSa r tA"


def test_goto_marker_is_one_label():
    (seg,) = split_mdc_segments_marked("A12-I9-|goto B1-D21:Q3-!")
    assert seg.text == "A12-I9-D21:Q3"
    assert seg.labels == ["|goto B1"]


def test_respelling():
    from sebashem_text import infer_respelling, respell

    corpus = ["jrj pat HAtj a", "sA .f smsw", "Dd .jn Hm .f"] * 50
    mapping = infer_respelling(corpus)
    assert mapping == {"i": "j", "y": "j", "z": "s"}
    assert respell("zA .f iri .i", mapping) == "sA .f jrj .j"
    assert infer_respelling(["iri .i", "sA .f"] * 50) == {"j": "i", "z": "s"}


def test_gly_units_keep_groups_and_toggles():
    from sebashem_text.gly_units import group_units, parse_gly, render_units

    doc = parse_gly("|R1-<-F12-S29->-[[-|B3, 4-G17-]]-$r-D21-!\n[[_-|R2-D21-Q3-]]-!")
    assert [u.kind for u in doc.units] == ["marker", "signs", "close", "open", "signs",
                                           "open", "marker", "signs", "signs", "close"]
    assert doc.units[1].signs == ["F12", "S29"]  # the cartouche is one unit
    groups = group_units(doc.units, [0, 4])      # cut before Q3, inside restored text and red
    assert [render_units(g) for g in groups] == [
        "|R1-<-F12-S29->-[[-|B3, 4-G17-]]-$r-D21-[[_-|R2-D21-]]-$b",
        "$r-[[-Q3-]]-$b",
    ]


def test_splitter_refine_moves_start_to_best_nearby_gap():
    import numpy as np

    from sebashem_text.sentences import SentenceSplitter, gap_features

    signs = ["A", "B", "C", "D", "E"]
    refiner = np.zeros(1 << 20, dtype=np.float32)
    refiner[gap_features(signs, 3)] = 1.0  # the sign model prefers a start before "D"
    s = SentenceSplitter(np.zeros(1, dtype=np.float32), 0.0, refiner=refiner, refine_radius=2, refine_penalty=0.5)
    assert s.refine(signs, [0, 2], allowed={1, 2, 3, 4}) == [0, 3]
    assert s.refine(signs, [0, 2], allowed={1, 2, 4}) == [0, 2]  # 3 isn't a unit boundary
