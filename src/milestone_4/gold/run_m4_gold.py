from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.milestone_4.gold import build_gold, gold_qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate the canonical M4 Gold layer (10 marts + 5 dimensions).")
    parser.add_argument("--write-csv-copies", action="store_true")
    parser.add_argument("--reset-dashboard-metadata", action="store_true")
    parser.add_argument("--reset-gold-structure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_gold.run(
        write_csv_copies=args.write_csv_copies,
        reset_dashboard_metadata=args.reset_dashboard_metadata,
        reset_gold_structure=args.reset_gold_structure,
    )
    outputs = gold_qa.run()
    manifest = json.loads(Path(outputs["manifest"]).read_text(encoding="utf-8"))
    if not manifest.get("gold_data_contract_ready", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
