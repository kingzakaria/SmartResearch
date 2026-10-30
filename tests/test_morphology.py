"""Tests of the morphology API. Run: pytest tests/test_morphology.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.morphology.analyzer import ArabicMorphologyAnalyzer, analyzer_from_config  # noqa: E402
from src.morphology.normalizer import normalize_ar  # noqa: E402
from src.morphology.stemmer import Stemmer, light_stem  # noqa: E402
from src.morphology.stopwords import is_stopword, load_stopwords  # noqa: E402
from src.morphology.tokenizer import tokenize  # noqa: E402


# ---------- normalizer ----------
def test_diacritics_and_tatweel_removed():
    assert normalize_ar("كَتَبَ") == "كتب"
    assert normalize_ar("كـــتاب") == "كتاب"


def test_letter_variants_unified():
    assert normalize_ar("أحمد إبراهيم آمن") == "احمد ابراهيم امن"
    assert normalize_ar("على") == "علي"
    assert normalize_ar("محكمة") == "محكمه"
    assert normalize_ar("محكمة", ta_marbuta=False) == "محكمة"


def test_digits_and_persian_letters():
    assert normalize_ar("الفصل ٣٧١") == "الفصل 371"
    assert normalize_ar("تحفیظ") == "تحفيظ"                  # Persian yeh


# ---------- tokenizer ----------
def test_punctuation_removed_including_arabic_marks():
    assert tokenize("المحكمة، الطاعن؛ والمطلوب؟ (نعم).") == ["المحكمة", "الطاعن", "والمطلوب", "نعم"]


def test_short_numbers_kept_long_numbers_dropped():
    assert tokenize("الفصل 371 من القانون 2020/1/1/4066 سنة 2023") == ["الفصل", "371", "من", "القانون", "سنة"]


def test_one_letter_words_dropped():
    assert tokenize("و ب كتاب") == ["كتاب"]


# ---------- stopwords ----------
def test_stopword_files_are_normalized_and_matched():
    sw = load_stopwords(config.STOPWORDS_GENERIC)
    assert normalize_ar("إلى") in sw                      # the file has إلى, the text has الي
    assert is_stopword(normalize_ar("إلى"), sw)
    assert is_stopword("وان", sw)                         # و + ان
    assert is_stopword("فان", sw)
    assert not is_stopword("وثيقه", sw)                   # a real word starting with و


def test_meaningful_negation_words_are_not_stopwords():
    sw = load_stopwords(config.STOPWORDS_GENERIC)
    assert "غير" not in sw and "عدم" not in sw


# ---------- light stemmer ----------
def test_light_stem_cases_seen_in_the_corpus():
    expected = {
        "المحكمه": "محكم", "للمحكمه": "محكم", "والمحكمه": "محكم",
        "القانون": "قانون", "قانون": "قانون",                    # was cut to قان by the first analyzer
        "الحق": "حق", "الطاعنين": "طاعن", "الورثه": "ورث", "ورثه": "ورث",
        "المطعون": "مطعون",                                       # participle, not a plural
        "بتقييده": "تقييد", "التقييد": "تقييد",
        "قرارها": "قرار", "الشفعه": "شفع",
        "وثيقه": "وثيق", "بيع": "بيع", "بطاقه": "بطاق",           # short words are protected
    }
    for word, stem in expected.items():
        assert light_stem(word) == stem, word


def test_numbers_are_never_stemmed():
    assert Stemmer("light").stem("371") == "371"


def test_modes():
    assert Stemmer("word").stem("المحكمه") == "المحكمه"
    assert Stemmer("root").stem("المحكمه") == "حكم"
    try:
        Stemmer("lemma")
    except ValueError:
        pass
    else:
        raise AssertionError("lemma mode must raise until it is implemented")


# ---------- full analyzer ----------
def make(mode="light"):
    return analyzer_from_config(config, mode)


def test_analyze_end_to_end():
    text = "وان المحكمة الطاعنين تمسكت بالشفعة بمقتضى الفصل 371 من قانون الالتزامات والعقود."
    assert make().analyze(text) == ["تمسكت", "شفع", "مقتض", "فصل", "371", "قانون", "التزام", "عقود"]


def test_boilerplate_removed_in_every_form():
    a = make()
    assert a.analyze("الطاعن الطاعنة الطاعنين المطعون المحكمة النقض") == []


def test_topic_words_survive():
    terms = make().analyze("دعوى التشطيب على إراثة بالرسم العقاري وتقييد الحق")
    for t in ("دعو", "تشطيب", "رسم", "عقار", "حق"):
        assert t in terms


def test_query_and_document_give_the_same_terms():
    a = make()
    assert a.analyze("المحكمة تقضي بالشفعة") == a.analyze("المحكمة تقضي بالشفعة")
    assert "شفع" in a.analyze("الشفعة") and a.analyze("الشفعة") == a.analyze("شفعة") == a.analyze("بالشفعة")


def test_explain_reports_every_token():
    rows = make().explain("وان الطاعنين تمسكوا بالشفعة")
    status = {tok: st for tok, st, _ in rows}
    assert status["وان"] == "stopword" and status["الطاعنين"] == "boilerplate" and status["بالشفعه"] == "kept"


def test_analyzer_without_files_still_works():
    a = ArabicMorphologyAnalyzer(mode="light")
    assert a.analyze("الشفعة") == ["شفع"]
