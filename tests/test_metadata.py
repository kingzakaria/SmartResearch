"""Tests built from the REAL headers seen in the corpus. Run: pytest tests/test_metadata.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.extraction.metadata import parse_document  # noqa: E402

F8 = """قرار محكمة النقض
رقم 1/15
الصادر بتاريخ 21 مارس 2023
في الملف العقاري رقم 2020/1/1/4066
دعوى التشطيب على إراثة - وجوب التمييز بين المطالبة بالحق وبين المطالبة بتقييده بالرسم العقاري.
لكن حق دعوى تحميه وانه بمقتضى الفصل 371 من قانون الالتزامات والعقود"""

F75 = """قرار محكمة النقض
رقم 168
الصادر بتاريخ 11 ابريل 2023
في الملف العقاري رقم 2656|1|4|2020
دعوى الإبطال - عقد
بيع
-
عقار محفظ.
حيث إن الطاعن يعيب على القرار"""

F986 = """قرار محكمة النقض
رقم 4124
الصادر بتاريخ 07 فبراير 2023
في الملف العقاري رقم 4973\\7\\4\\2021
مرض الموت – شروطه.
إن مرض الموت هو الذي يجمع فيه علماء الطب"""

F117 = """قرار محكمة النقض
رقم 130
الصادر بتاريخ 28مارس 2023
في الملف العقاري رقم 2020115399
تعرض على مطلب تحفيظ - سبقية البت - شروطها – خلف عام – أثره.
إن المحكمة غير ملزمة بمجاراة الأطراف"""

F31 = """قرار محكمة النقض
رقم 1/111
الصاور بتاريخ 02 مايو 2023
علف عقاري رقم 2019/1/1/8324
نزاع تحفيظ - رفض ملتمس إجراء تحقيق بالوقوف على عين المكان – أثره.
الثابت أن الطاعنين نازعوا في الخبرة"""

F62 = """وبعد المداولة طبقا للقانون
حيث يستفاد من مستندات الملف أنه بمقتضى مطلب تحفيظ قيد بالمحافظة العقارية
بتاونات بتاريخ 2005/11/25 تحت عدد 37/7741 طلب الطاعن (ح) محمد بن لحسن تحفیظ
الملك المسمى "ملك (ح) " الكائن بحي أولاد سعيد مركز تاونات حددت مساحته في آرين"""


def test_standard_header():
    d = parse_document(F8)
    assert d["decision_no"] == "1/15"
    assert d["date"] == "2023-03-21" and d["year"] == 2023
    assert d["case_no"] == "2020/1/1/4066"
    assert d["subject"].startswith("دعوى التشطيب") and d["subject"].endswith("العقاري.")
    assert d["body"].startswith("دعوى التشطيب")


def test_decision_number_without_slash():
    assert parse_document(F75)["decision_no"] == "168"


def test_pipe_separators_and_wrapped_subject():
    d = parse_document(F75)
    assert d["case_no"] == "2656/1/4/2020"
    assert d["date"] == "2023-04-11"                      # "ابريل" without hamza
    assert d["subject"] == "دعوى الإبطال - عقد بيع - عقار محفظ."


def test_backslash_separators():
    d = parse_document(F986)
    assert d["case_no"] == "4973/7/4/2021"
    assert d["date"] == "2023-02-07"
    assert d["subject"] == "مرض الموت – شروطه."


def test_no_space_in_date_and_no_separator_in_case():
    d = parse_document(F117)
    assert d["date"] == "2023-03-28"                      # "28مارس"
    assert d["case_no"] == "2020115399"


def test_typos_in_labels():
    d = parse_document(F31)                               # "الصاور" and "علف عقاري"
    assert d["date"] == "2023-05-02"
    assert d["case_no"] == "2019/1/1/8324"
    assert d["decision_no"] == "1/111"


def test_file_without_header():
    d = parse_document(F62)
    assert d["decision_no"] is None and d["date"] is None and d["case_no"] is None
    assert d["subject"] is None
    assert d["body"].startswith("وبعد المداولة")          # whole text kept as body
    assert "تحفيظ" in d["text"] and "ی" not in d["text"]   # Persian yeh repaired


# ---------- real headers from the second diagnostic run (448, 978, 642, 639, 465, 833, 43, 113) ----------
F448 = """قرار محكمة النقض
رقم 2/94
الصادر بتاريخ 28 فبراير 2023
في الملف العقاري 2019/4/1/7471
دفوع جوهرية – عدم الجواب عنها – أثره
إن عدم الجواب على الدفوع الجوهرية وهي تلك التي إن صحت يكون لها تأثير
على مسار القضية يشكل حالة من حالات انعدام التعليل الموجب للنقض"""

F978 = """قرار محكمة النقض
رتم 3/41
الصاور بتاريخ 07 فبراير 2023
ملف عقاري - الهيئة الثالثة - رقم 2021/8/1/750
نزاع تحفيظ – إقرار بالقسمة – أثره.
إن المحكمة لما قضت بما جرى به منطوق قرارها بعدما تأكد لها من خلال البحث الذي تم"""

# lines before/after the title as printed by the diagnostic; the line after the case number is "حجيته ."
F642 = """قسمة
-
رسم إحصاء
-
قرار محكمة النقض
رقم 2/78
الصادر بتاريخ 21 فبراير 2023
علف عقاري رقم 2019/4/1/3938
حجيته .
إن القسمة التي أجريت بين الورثة تعتبر حجة على جميع الأطراف في هذا النزاع"""

F639 = """قسمة عينية - شروطها .
قرار محكمة النقض
رقم 2/85
الصاور بتاريخ 21 فبراير 2023
ملف عقاري رقم 2020/4/1/378
تحكم المحكمة بقسمة العقار المشاع قسمة عينية بفرز الحصص وتكوين الأنصبة على أساس
أصغر حصة عن طريق التقويم والتعديل ثم توزع الأنصبة المفرزة بين الشركاء بالقرعة وتصدر حكمها"""

F465 = """قرار محكمة النقض
رقم 4/67
الصادر بتاريخ 28 فبراير 2023
في الملف العقاري رقم 2021/4/7/8871
دعوى الحيازة - شروطها
لا يمكن رفع دعوى الحيازة إلا ممن كانت له شخصيا أو بواسطة الغير منذ سنة
على الأقل حيازة عقار أو حق عيني حيازة هادئة علنية متصلة غير متقطعة، وغير مجردة"""

F833 = """قرار محكمة النقض
رقم 2/74
الصادر بتاريخ 14 فبراير 2023
في الملف العقاري 2021/4/1/7319
-
شفعة – تزاحم الشفعاء – أثره.
المقرر قانونا أنه متى بتت محكمة النقض في نقطة معينة وجب على محكمة الإحالة"""

F113 = """قرار محكمة النقض
رقم 1/41
الصادر بتاريخ 28 مارس 2023
في الملف العقاري رقم 2020/1/1/1095
تشكيل المحاكم - تعلقه بالنظام العام – أثره
إن تشكيل المحاكم من النظام العام، وهو يثار تلقائيا ويؤدي إلى النقض متى تحقق
سببه. وأنه يمتنع على القاضي أن ينظر قضية في طور الاستئناف أو النقض بعد ما سبق"""

F43 = """قرار محكمة النقض
رقم 1/103
الصادر بتاريخ 18 ابريل 2023
في الملف العقاري رقم 2022/1/1/2254
-
نزاع تحفيظ - عدم انطباق رسم شراء المتعرض على العقار محل النزاع – أثره
رسم
إن رسم الشراء المدلى به لا ينطبق على العقار موضوع مطلب التحفيظ وهو ما يجعل"""


def test_case_number_without_the_word_raqm():
    d = parse_document(F448)
    assert d["case_no"] == "2019/4/1/7471"
    assert d["subject"] == "دفوع جوهرية – عدم الجواب عنها – أثره"      # stops before the reasoning
    assert d["body"].startswith("دفوع جوهرية")


def test_typo_label_and_extra_words_in_case_line():
    d = parse_document(F978)
    assert d["decision_no"] == "3/41"                  # "رتم"
    assert d["case_no"] == "2021/8/1/750"
    assert d["date"] == "2023-02-07"


def test_subject_before_the_title_line():
    d = parse_document(F642)
    assert d["decision_no"] == "2/78" and d["date"] == "2023-02-21"
    assert d["case_no"] == "2019/4/1/3938"
    assert d["subject"].startswith("قسمة - رسم إحصاء")
    d2 = parse_document(F639)
    assert d2["subject"] == "قسمة عينية - شروطها ."
    assert d2["case_no"] == "2020/4/1/378"


def test_subject_without_full_stop_is_not_merged_with_reasoning():
    assert parse_document(F465)["subject"] == "دعوى الحيازة - شروطها"
    assert parse_document(F113)["subject"] == "تشكيل المحاكم - تعلقه بالنظام العام – أثره"


def test_leading_dash_line_is_skipped():
    d = parse_document(F833)
    assert d["subject"] == "شفعة – تزاحم الشفعاء – أثره."
    assert d["case_no"] == "2021/4/1/7319"
    d43 = parse_document(F43)
    assert d43["subject"].startswith("نزاع تحفيظ")
    assert d43["subject"] != "-"
