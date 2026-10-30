"""Morphology step 4: stemming / root extraction. Three levels, to compare in the report:

    "word"  : no reduction (the token itself)
    "light" : strip prefixes and suffixes only (hand-written, Larkey-style "light stemming")
    "root"  : ISRI stemmer (src/morphology/isri.py), goes down to the 3-letter root (racine)

The light stemmer is written by hand on purpose: every rule is visible and its thresholds are
constants, so over-stemming (قانون -> قان) can be controlled. Input must already be normalized
(ة -> ه, ى -> ي, hamza forms of alef -> ا), so the rules below are written for that spelling.
"""

# Definite-article family. Order matters: longest first. (prefix, minimum length of what remains)
# 2 remaining letters are accepted so that الحق -> حق (the article is unambiguous).
ARTICLE_PREFIXES = (("وال", 2), ("بال", 2), ("كال", 2), ("فال", 2), ("لل", 3), ("ال", 2))

# Bare one-letter prefixes are ambiguous, so they are only cut when a long stem remains:
#   و (and): the word must have 6+ letters. ورثة, وثيقة, واقعة, وكالة, ولاية (5 letters) stay whole,
#            while ومحكمة, وتقييد are cut.
#   ب / ل (by, for): 5+ letters must remain. بتقييده -> تقييد, but بطاقة, بلدية, بيع stay whole.
CONJUNCTION_MIN_LEN = 6
PREPOSITION_MIN_REST = 5

# Pronoun / plural / feminine endings. After normalization ة became ه, so "ية" is "يه".
SUFFIXES = (("ها", 3), ("هم", 3), ("ان", 3), ("ات", 3), ("ون", 3), ("ين", 3), ("يه", 3),
            ("ه", 3), ("ي", 3))

# Words that END like a plural/dual suffix but are not: they must stay whole.
PROTECTED = {"قانون", "عنوان", "شؤون", "ديون", "فنون", "عيون", "متون", "مضمون", "مجنون"}


def _strip_article(word: str) -> str:
    for prefix, min_rest in ARTICLE_PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= min_rest:
            return word[len(prefix):]
    return word


def _strip_prefix(word: str) -> str:
    stripped = _strip_article(word)                  # وال بال كال فال لل ال
    if stripped != word:
        return stripped
    if word.startswith("و") and len(word) >= CONJUNCTION_MIN_LEN:
        word = word[1:]                              # و + (ال... | ب... | stem)
        stripped = _strip_article(word)
        if stripped != word:
            return stripped
    for letter in ("ب", "ل"):
        if word.startswith(letter) and len(word) - 1 >= PREPOSITION_MIN_REST:
            return word[1:]
    return word


def light_stem(word: str) -> str:
    word = _strip_prefix(word)
    if word in PROTECTED:
        return word
    for suffix, min_rest in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_rest:
            rest = word[:-len(suffix)]
            # passive participles of the pattern م+3 letters+ون (مطعون, مرهون, مديون) are not plurals
            if suffix in ("ون", "ين") and len(rest) == 3 and rest[0] == "م":
                continue
            return rest                             # one suffix only
    return word


class Stemmer:
    def __init__(self, mode: str = "light"):
        if mode not in ("word", "light", "root"):
            raise ValueError(f"mode must be 'word', 'light' or 'root' (got {mode!r}); "
                             "'lemma' needs CAMeL Tools / Qalsadi and is not implemented yet")
        self.mode = mode
        self._isri = None
        self._cache: dict[str, str] = {}

    def stem(self, word: str) -> str:
        if self.mode == "word":
            return word
        if word.isdigit():                          # article numbers are never stemmed
            return word
        if word not in self._cache:
            self._cache[word] = light_stem(word) if self.mode == "light" else self._root(word)
        return self._cache[word]

    def _root(self, word: str) -> str:
        if self._isri is None:
            from .isri import ISRIStemmer            # vendored copy: no nltk needed
            self._isri = ISRIStemmer()
        return self._isri.stem(word)