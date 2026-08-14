from __future__ import annotations

KEYWORD_RULE_VERSION = "m3-text-keywords-v1"

KEYWORD_RULES: dict[str, dict[str, str]] = {
    "new_build": {
        "keyword_fa": "نوساز",
        "pattern": r"(?:نوساز|نو\s*ساز)",
        "definition": "The listing describes the property/unit as newly built.",
    },
    "unused": {
        "keyword_fa": "کلیدنخورده",
        "pattern": r"(?:کلیدنخورده|کلید\s*نخورده)",
        "definition": "The listing describes the unit as unused/not previously occupied.",
    },
    "urgent": {
        "keyword_fa": "فوری",
        "pattern": r"فوری",
        "definition": "The listing explicitly expresses urgency.",
    },
    "exchange": {
        "keyword_fa": "معاوضه",
        "pattern": r"معاوضه",
        "definition": "The listing explicitly mentions exchange/trade.",
    },
    "below_market": {
        "keyword_fa": "زیر قیمت",
        "pattern": r"(?:زیر\s*قیمت|زیرقیمت)",
        "definition": "The listing explicitly claims a below-market/below-normal asking price.",
    },
    "migration_sale": {
        "keyword_fa": "فروش به دلیل مهاجرت",
        "pattern": r"(?:به\s*(?:دلیل|علت)\s*مهاجرت|بخاطر\s*مهاجرت|به\s*خاطر\s*مهاجرت|فروش(?:\s*فوری)?\s*به\s*(?:دلیل|علت)\s*مهاجرت|فروش.{0,24}مهاجرت|مهاجرت.{0,24}فروش)",
        "definition": "The listing links the sale/urgency to migration; broad variants require manual precision review.",
    },
}

MANDATORY_ASSIGNMENT_KEYWORDS = {"urgent", "below_market"}
