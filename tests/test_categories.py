"""Subject-based categories. Run: pytest tests/test_categories.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.analysis.categories import UNLABELLED, assign_category, load_rules, subject_head  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402

ANALYZER = analyzer_from_config(config, "light")
RULES = load_rules(config.RESOURCES_DIR / "category_rules.json", ANALYZER)


def category(subject):
    return assign_category(subject, RULES, ANALYZER)


def test_subject_head():
    assert subject_head("نزاع تحفيظ - عبء الإثبات") == "نزاع تحفيظ"
    assert subject_head("شفعة – تزاحم الشفعاء – أثره.") == "شفعة"
    assert subject_head("- أثره . نزاع تحفيظ") == "أثره . نزاع تحفيظ"      # a leading dash (PDF artefact) is skipped
    assert subject_head(None) == "" and subject_head("") == ""


def test_main_categories():
    assert category("شفعة – تزاحم الشفعاء – أثره.") == ("preemption_partition", "head")
    assert category("دعوى الإفراغ - ادعاء العلاقة الكرائية") == ("eviction", "head")
    assert category("نزاع تحفيظ – نظام الإثبات.") == ("registration", "head")
    assert category("دعوى القسمة وتحديد نصيب في الاستغلال") == ("preemption_partition", "head")
    assert category("طعن بإعادة النظر - حالاته.") == ("procedure", "head")


def test_head_wins_over_the_rest_of_the_subject():
    # the issue is a registration dispute even though the cassation appeal is mentioned later
    assert category("نزاع تحفيظ - طعن بالنقض - مقال") == ("registration", "head")


def test_whole_subject_is_the_second_chance_and_unknown_is_other():
    assert category("مسألة غامضة - كلام عام") == (UNLABELLED, "none")
    assert category("مسألة - بيع العقار") == ("property_contracts", "subject")
    assert category(None) == (UNLABELLED, "none")


def test_spelling_variants_match_through_stems():
    assert category("دعوى الافراغ")[0] == "eviction"          # no hamza
    assert category("الشفعاء وتزاحمهم")[0] == "preemption_partition"


def test_subjects_that_were_unlabelled_now_have_a_category():
    assert category("استحقاق - تعرض الغير")[0] in ("property_contracts", "registration")
    assert category("دعوى الاستحقاق")[0] == "property_contracts"
    assert category("تشكيلة المحاكم")[0] == "procedure"
    assert category("وصية - إبطالها")[0] == "property_contracts"
    assert category("محكمة الإحالة – شفعة – تبعيضها - أثره")[0] == "preemption_partition"
