from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import polars as pl

VALID_STATUSES = {"PASS", "REVIEW", "FAIL"}


@dataclass(frozen=True)
class Check:
    check_id: str
    category: str
    actual: str
    expected: str
    status: str
    critical: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Unsupported check status: {self.status}")


def make_check(
    check_id: str,
    category: str,
    actual: object,
    expected: object,
    passed: bool,
    *,
    critical: bool = True,
    review_on_fail: bool = False,
    notes: str = "",
) -> Check:
    status = "PASS" if passed else ("REVIEW" if review_on_fail else "FAIL")
    return Check(check_id, category, str(actual), str(expected), status, critical, notes)


def checks_frame(checks: Iterable[Check]) -> pl.DataFrame:
    return pl.DataFrame([asdict(check) for check in checks])


def summarize_checks(checks: Iterable[Check]) -> dict[str, object]:
    items = list(checks)
    critical_failures = sum(c.critical and c.status == "FAIL" for c in items)
    reviews = sum(c.status == "REVIEW" for c in items)
    overall_status = "FAIL" if critical_failures else ("REVIEW" if reviews else "PASS")
    return {
        "overall_status": overall_status,
        "ready": critical_failures == 0,
        "critical_failures": critical_failures,
        "review_count": reviews,
        "check_count": len(items),
    }
