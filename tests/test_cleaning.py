"""Cleaning of letterhead lines and the closing ritual, using the REAL endings of 8.txt and 448.txt.
Run: pytest tests/test_cleaning.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.extraction.cleaner import clean_text, split_closing  # noqa: E402
from src.extraction.metadata import parse_document  # noqa: E402

HEAD8 = """قرار محكمة النقض
رقم 1/15
الصادر بتاريخ 21 مارس 2023
في الملف العقاري رقم 2020/1/1/4066
دعوى التشطيب على إراثة - وجوب التمييز بين المطالبة بالحق وبين المطالبة بتقييده بالرسم العقاري.
لكن حق دعوى تحميه وانه بمقتضى الفصل 371 من قانون الالتزامات والعقود
"""

# letterhead BEFORE the operative part (8.txt)
DOC8 = HEAD8 + """المملكة المغربية
المجلس الأعلى للسلطة القضائية
محكم هذال الأسباب
قضت محكمة النقض بنقض القرار المطعون فيه وإحالة القضية وطرفيها على نفس
المحكمة للبت فيها طبقا للقانون وعلى المطلوبين المصاريف.
بطرته .
كما قررت إثبات قرارها هذا بسجلات المحكمة المصدرة له إثر الحكم المطعون فيه أو
وبهذا صدر القرار وتلي بالجلسة العلنية المنعقدة بالتاريخ المذكور أعلاه بقاعة الجلسات
العادية بمحكمة النقض بالرباط. وكانت الهيئة الحاكمة متركبة من رئيس الغرفة السيد حسن
منصف رئيسا والمستشارين السادة محمد أسراج مقررا، ومحمد شافي، وعبد الوهاب عافلاني
وسمير رضوان أعضاء وبمحضر المحامي العام السيد رشيد صدوق وبمساعدة كاتبة الضبط
السيدة ابتسام الزواغي."""

# letterhead AFTER the signature block (448.txt)
DOC448 = HEAD8 + """قضت محكمة النقض بنقض القرار المطعون فيه وإحالة القضية وطرفيها على نفس
المحكمة للبت فيها طبقا للقانون وعلى المطلوبين المصاريف.
بطرته .
كما قررت إثبات قرارها هذا بسجلات المحكمة المصدرة له إثر الحكم المطعون فيه أو
وبهذا صدر القرار وتلي بالجلسة العلنية المنعقدة بالتاريخ المذكور أعلاه بقاعة الجلسات
العادية بمحكمة النقض بالرباط. وكانت الهيئة الحاكمة متركبة من رئيسة الهيئة السيدة نادية
الكاعم رئيسة والمستشارين السادة: محمد رضوان ،مقررا، والمصطفى جرايف وعبد اللطيف
معادي والمهدي شباب أعضاء وبمحضر المحامي العام السيد نور الدين الشطبي وبمساعدة
كاتبة الضبط السيدة ابتسام الزواغي.
المملكة المغربية
المجلس الأعلى للسلطة القضائية
محكمة النقض"""

LETTERHEAD = ("المملكة المغربية", "المجلس الأعلى للسلطة القضائية")


def test_letterhead_removed_wherever_it_is():
    for raw in (DOC8, DOC448):
        text = clean_text(raw)
        assert not any(h in text for h in LETTERHEAD)
        assert not any(line.strip() == "محكمة النقض" for line in text.split("\n"))


def test_title_line_and_longer_lines_with_the_court_name_are_kept():
    text = clean_text(DOC8)
    assert text.startswith("قرار محكمة النقض")
    assert "قضت محكمة النقض بنقض القرار المطعون فيه" in text


def test_closing_is_cut_from_the_body_and_kept_apart_8():
    d = parse_document(DOC8)
    assert d["closing"].startswith("وبهذا صدر القرار وتلي")
    assert "المحامي العام" in d["closing"] and d["closing"].endswith("الزواغي.")
    for name in ("ابتسام", "المحامي العام", "الهيئة الحاكمة", "رشيد"):
        assert name not in d["body"]
    assert "المصاريف." in d["body"]                      # the operative part stays in the body
    assert "المحامي العام" in d["text"]                   # the full text is untouched


def test_closing_is_cut_when_letterhead_comes_after_it_448():
    d = parse_document(DOC448)
    assert d["closing"].endswith("الزواغي.")             # letterhead no longer pollutes the end
    assert "المملكة" not in d["closing"]
    assert "نور الدين" not in d["body"]


def test_metadata_still_parsed_after_cleaning():
    d = parse_document(DOC8)
    assert d["decision_no"] == "1/15" and d["date"] == "2023-03-21" and d["case_no"] == "2020/1/1/4066"
    assert d["subject"].startswith("دعوى التشطيب")


def test_no_closing_means_nothing_is_cut():
    body, closing = split_closing("حيث إن الطاعن يعيب على القرار\nوالمحكمة ردت دفعه")
    assert closing == "" and body.startswith("حيث")


def test_phrase_far_from_the_end_is_a_quotation_not_a_closing():
    lines = ["جاء في القرار المنتقد وبهذا صدر القرار وتلي بالجلسة العلنية"] + [f"سطر {i} من التعليل" for i in range(40)]
    body, closing = split_closing("\n".join(lines))
    assert closing == "" and "سطر 39" in body


def test_invisible_characters_are_removed():
    assert clean_text("قرار" + chr(0x200F) + chr(0xFEFF) + " محكمة") == "قرار محكمة"
