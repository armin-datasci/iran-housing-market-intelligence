from __future__ import annotations

import argparse
import json
from urllib.request import urlopen


def get_json(url: str) -> dict:
    with urlopen(url, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running IHMI FastAPI deployment.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health = get_json(f"{base}/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"API health is not ok: {health}")

    meta = get_json(f"{base}/api/v1/meta")
    architecture = meta.get("architecture", {})
    if architecture.get("mart_count") != 10 or architecture.get("dimension_count") != 5:
        raise RuntimeError(f"Unexpected Gold architecture: {architecture}")

    market = get_json(f"{base}/api/v1/market/monthly?limit=1")
    if market.get("dataset") != "mart_market_monthly":
        raise RuntimeError("Monthly market endpoint did not return the expected dataset.")

    model = get_json(f"{base}/api/v1/model/quality?limit=1")
    if model.get("dataset") != "mart_model_quality":
        raise RuntimeError("Model-quality endpoint did not return the expected dataset.")

    print("IHMI FASTAPI SMOKE TEST: PASS")
    print("health:", health.get("status"))
    print("gold_qa_status:", health.get("gold_qa_status"))
    print("marts:", architecture.get("mart_count"))
    print("dimensions:", architecture.get("dimension_count"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
