from __future__ import annotations

import argparse
import hashlib
import platform
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from src.common.io_utils import atomic_write_json
from src.common.paths import OUTPUTS_DIR, project_root, relative_to_project


VERSION = "final-environment-snapshot-v1.0"
PROJECT_ROOT = project_root()
OUTPUT_PATH = OUTPUTS_DIR / "qa" / "final_pipeline" / "environment_versions.json"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

CORE_DISTRIBUTIONS = (
    "polars",
    "pyarrow",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "PyYAML",
    "psutil",
    "joblib",
    "jupyter",
    "pytest",
    "fastapi",
    "uvicorn",
)


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat()


def _installed_version(distribution: str) -> str | None:
    """Return an installed distribution version without importing the package."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _requirement_name(line: str) -> str | None:
    """Extract a distribution name from a standard requirements.txt line."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith(("-", "--")):
        return None
    stripped = stripped.split(";", 1)[0].strip()
    stripped = stripped.split(" #", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", stripped)
    return match.group(1) if match else None


def _requirements_snapshot(requirements_path: Path) -> dict[str, Any]:
    """Record declared requirement names and installed versions without storing raw lines."""
    if not requirements_path.exists():
        return {
            "path": "requirements.txt",
            "present": False,
            "sha256": None,
            "declared_distributions": {},
            "missing_declared_distributions": [],
        }

    raw = requirements_path.read_bytes()
    names: list[str] = []
    for line in raw.decode("utf-8-sig", errors="replace").splitlines():
        name = _requirement_name(line)
        if name and name.lower() not in {item.lower() for item in names}:
            names.append(name)

    declared = {name: _installed_version(name) for name in names}
    missing = [name for name, installed in declared.items() if installed is None]
    return {
        "path": "requirements.txt",
        "present": True,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "declared_distributions": declared,
        "missing_declared_distributions": missing,
    }


def build_environment_snapshot(output_path: Path = OUTPUT_PATH) -> Path:
    """Write a privacy-safe runtime/library version snapshot for reproducibility QA."""
    core_versions = {
        distribution: _installed_version(distribution)
        for distribution in CORE_DISTRIBUTIONS
    }
    missing_core = [
        distribution
        for distribution, installed in core_versions.items()
        if installed is None
    ]

    requirements = _requirements_snapshot(REQUIREMENTS_PATH)
    missing_declared = list(requirements["missing_declared_distributions"])

    overall_status = "PASS" if not missing_core and not missing_declared else "REVIEW"
    payload: dict[str, Any] = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "overall_status": overall_status,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
        },
        "core_distributions": core_versions,
        "missing_core_distributions": missing_core,
        "requirements": requirements,
        "privacy_contract": {
            "username_recorded": False,
            "home_directory_recorded": False,
            "working_directory_recorded": False,
            "interpreter_absolute_path_recorded": False,
            "environment_variables_recorded": False,
            "secrets_or_tokens_recorded": False,
        },
        "interpretation": (
            "This artifact records runtime/library versions for reproducibility. "
            "A REVIEW indicates one or more declared/core distributions were not resolvable "
            "through importlib.metadata; it does not change analytical results."
        ),
    }

    output_path = output_path.resolve()
    atomic_write_json(payload, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the final IHMI runtime/library version snapshot."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when the environment snapshot status is REVIEW.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = build_environment_snapshot(args.output)
    payload_status = "PASS"
    core_missing = [
        distribution
        for distribution in CORE_DISTRIBUTIONS
        if _installed_version(distribution) is None
    ]
    requirements = _requirements_snapshot(REQUIREMENTS_PATH)
    if core_missing or requirements["missing_declared_distributions"]:
        payload_status = "REVIEW"

    print("FINAL ENVIRONMENT SNAPSHOT GENERATED")
    print(f"status: {payload_status}")
    print(f"output: {relative_to_project(path)}")
    if args.strict and payload_status != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
