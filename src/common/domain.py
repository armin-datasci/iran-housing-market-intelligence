from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from .paths import CONFIG_DIR


@lru_cache(maxsize=1)
def load_domain_mappings() -> dict[str, Any]:
    path = CONFIG_DIR / "domain_mappings.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("domain_mappings.yaml must contain a mapping at the root.")
    return data


def property_family_categories() -> dict[str, set[str]]:
    families = load_domain_mappings().get("property_families", {})
    return {
        str(family): {str(value).strip() for value in spec.get("categories", [])}
        for family, spec in families.items()
        if isinstance(spec, dict)
    }


def category_to_property_family() -> dict[str, str]:
    result: dict[str, str] = {}
    for family, categories in property_family_categories().items():
        for category in categories:
            result[category] = family
    return result
