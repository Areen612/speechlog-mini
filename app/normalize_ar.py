import re

DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")


def normalize_ar(text: str) -> str:
    text = text.replace("ـ", "")  # tatweel
    text = DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text