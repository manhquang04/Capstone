"""Generate Pareto data using gradient inversion final loss as privacy proxy.

This script is intentionally standalone and does not import experiment modules.
Run from the FL-DNA root:

    python generate_pareto.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
ATTACK_DIR = PROJECT_ROOT / "artifacts" / "gradient_inversion" / "combined_sweeps"
FRAUD_DIR = PROJECT_ROOT / "results" / "fraud"
LEGACY_PARETO_JSON = PROJECT_ROOT / "results" / "pareto" / "pareto_data.json"
OUTPUT_JSON = FRAUD_DIR / "pareto_data.json"
OUTPUT_CSV = FRAUD_DIR / "pareto_data.csv"


METHOD_ATTACK_SELECTORS: dict[str, list[dict[str, str]]] = {
    "FL Baseline": [{"method": "FL_Baseline"}],
    "FL + DNA": [
        {"method": "FL_DNA_PreAggregationLeakage"},
        {"method": "FL_DNA"},
    ],
    "FL + DNA Transform": [
        {"method": "FL_DNA_TransformDefense", "strength": "current"},
        {"method": "FL_DNA"},
    ],
    "FL + DNA Transform (conservative)": [
        {"method": "FL_DNA_TransformDefense", "strength": "conservative"},
    ],
    "FL + DNA Transform (medium)": [
        {"method": "FL_DNA_TransformDefense", "strength": "medium"},
    ],
    "FL + DNA Transform (stronger)": [
        {"method": "FL_DNA_TransformDefense", "strength": "stronger"},
    ],
    "FL + DNA Transform (pre-agg leak)": [
        {"method": "FL_DNA_TransformDefense_PreAggregationLeakage"},
        {"method": "FL_DNA_PreAggregationLeakage"},
        {"method": "FL_PreAggregationLeakage"},
    ],
    "FL + DP": [
        {"method": "FL_DP", "dp_noise_preset": "medium"},
        {"method": "FL_DP"},
    ],
    "FL + DP (mild)": [
        {"method": "FL_DP", "dp_noise_preset": "mild"},
    ],
    "FL + DP (utility)": [
        {"method": "FL_DP", "dp_noise_preset": "utility"},
    ],
    "FL + DNA+DP": [{"method": "FL_DNA_DP"}],
    "FL + SecureAgg": [{"method": "FL_SecureAgg"}],
    "FL + DNA+SecureAgg": [{"method": "FL_DNA_SecureAgg"}],
    "FL + DNA Transform+SecureAgg": [{"method": "FL_DNA_TransformDefense_SecureAgg"}],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_loss(row: dict[str, Any]) -> float | None:
    for key in ("final_loss", "mean_final_loss", "mean_best_attack_loss", "best_attack_loss"):
        value = as_float(row.get(key))
        if value is not None:
            return value

    losses = row.get("losses")
    if isinstance(losses, list):
        values = [as_float(value) for value in losses]
        numeric = [value for value in values if value is not None]
        if numeric:
            return sum(numeric) / len(numeric)

    return None


def extract_samples(row: dict[str, Any]) -> int:
    for key in ("samples", "sample_count", "n_samples", "num_samples"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return max(int(value), 1)
        except (TypeError, ValueError):
            continue
    return 1


def flatten_attack_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "methods", "rows", "attacks", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if "method" in payload:
            return [payload]
    return []


def load_attack_rows() -> list[dict[str, Any]]:
    if not ATTACK_DIR.is_dir():
        print(f"WARNING: gradient inversion folder missing: {ATTACK_DIR}")
        return []

    rows: list[dict[str, Any]] = []
    json_files = sorted(ATTACK_DIR.glob("*.json"))
    if not json_files:
        print(f"WARNING: no JSON files found in {ATTACK_DIR}")
        return []

    for path in json_files:
        try:
            payload = load_json(path)
        except json.JSONDecodeError as exc:
            print(f"WARNING: cannot parse {path}: {exc}")
            continue
        for row in flatten_attack_payload(payload):
            row = dict(row)
            row["_source_file"] = str(path.relative_to(PROJECT_ROOT))
            rows.append(row)

    if not rows:
        print(f"WARNING: no attack rows found in {ATTACK_DIR}")
    return rows


def row_matches(row: dict[str, Any], selector: dict[str, str]) -> bool:
    for key, expected in selector.items():
        if str(row.get(key, "")).lower() != expected.lower():
            return False
    return True


def weighted_mean_loss(rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    total_weighted_loss = 0.0
    total_samples = 0
    for row in rows:
        loss = extract_loss(row)
        if loss is None:
            continue
        samples = extract_samples(row)
        total_weighted_loss += loss * samples
        total_samples += samples
    if total_samples == 0:
        return None, 0
    return total_weighted_loss / total_samples, total_samples


def attack_stats_for_method(method: str, attack_rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    selectors = METHOD_ATTACK_SELECTORS.get(method, [])
    for selector in selectors:
        matched = [row for row in attack_rows if row_matches(row, selector)]
        mean_loss, sample_count = weighted_mean_loss(matched)
        if mean_loss is not None:
            return mean_loss, sample_count
    return None, 0


def load_utility_methods() -> list[dict[str, Any]]:
    source = OUTPUT_JSON if OUTPUT_JSON.is_file() else LEGACY_PARETO_JSON
    if not source.is_file():
        print(f"WARNING: no existing Pareto utility file found at {OUTPUT_JSON} or {LEGACY_PARETO_JSON}")
        return []

    if source != OUTPUT_JSON:
        print(f"WARNING: {OUTPUT_JSON.relative_to(PROJECT_ROOT)} missing; using {source.relative_to(PROJECT_ROOT)} for utility metrics")

    payload = load_json(source)
    methods = payload.get("methods", []) if isinstance(payload, dict) else []
    if not isinstance(methods, list):
        print(f"WARNING: existing Pareto file has no methods list: {source}")
        return []

    return [item for item in methods if isinstance(item, dict)]


def utility_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def build_rows(utility_rows: list[dict[str, Any]], attack_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_losses: list[float] = []

    for utility_row in utility_rows:
        method = str(utility_row.get("method"))
        mean_loss, n_samples = attack_stats_for_method(method, attack_rows)
        if mean_loss is not None:
            raw_losses.append(mean_loss)
        rows.append(
            {
                "method": method,
                "f1": utility_value(utility_row, "f1", "final_f1"),
                "roc_auc": utility_value(utility_row, "roc_auc", "final_auc_roc"),
                "pr_auc": utility_value(utility_row, "pr_auc", "final_pr_auc"),
                "privacy_proxy": None,
                "grad_inv_final_loss_mean": mean_loss,
                "grad_inv_n_samples": n_samples,
            }
        )

    max_loss = max(raw_losses) if raw_losses else None
    if max_loss is None or max_loss <= 0:
        print("WARNING: no positive gradient inversion final loss found; all privacy_proxy values remain null")
        return rows

    for row in rows:
        mean_loss = row["grad_inv_final_loss_mean"]
        if mean_loss is not None:
            row["privacy_proxy"] = float(mean_loss) / max_loss
    return rows


def write_outputs(rows: list[dict[str, Any]]) -> None:
    FRAUD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "privacy_metric": "gradient_inversion_final_loss_normalized",
        "note": "privacy_proxy = mean_final_loss / max_mean_final_loss; higher = more private",
        "methods": rows,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "method",
        "f1",
        "roc_auc",
        "pr_auc",
        "privacy_proxy",
        "grad_inv_final_loss_mean",
        "grad_inv_n_samples",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def print_report(rows: list[dict[str, Any]]) -> None:
    with_data = [row for row in rows if row.get("privacy_proxy") is not None]
    missing = [row for row in rows if row.get("privacy_proxy") is None]

    print(f"Methods with gradient inversion data: {len(with_data)}")
    print(f"Methods missing gradient inversion data: {len(missing)}")
    for row in missing:
        print(f"WARNING: null privacy_proxy for {row['method']} (no matching gradient inversion final_loss)")

    print("Top 3 most private methods:")
    for row in sorted(with_data, key=lambda item: float(item["privacy_proxy"]), reverse=True)[:3]:
        print(
            f"  {row['method']}: proxy={row['privacy_proxy']:.6f}, "
            f"loss={row['grad_inv_final_loss_mean']:.8f}, n={row['grad_inv_n_samples']}"
        )

    print("Top 3 best utility methods:")
    utility_rows = [row for row in rows if row.get("f1") is not None]
    for row in sorted(utility_rows, key=lambda item: float(item["f1"]), reverse=True)[:3]:
        proxy = row.get("privacy_proxy")
        proxy_text = "null" if proxy is None else f"{proxy:.6f}"
        print(f"  {row['method']}: F1={float(row['f1']):.6f}, proxy={proxy_text}")


def verify(rows: list[dict[str, Any]]) -> None:
    non_null = [row["privacy_proxy"] for row in rows if row.get("privacy_proxy") is not None]
    if non_null and len({round(float(value), 12) for value in non_null}) == 1:
        print("WARNING: all non-null privacy_proxy values are identical; check attack data mapping")

    by_method = {str(row["method"]): row for row in rows}
    baseline = by_method.get("FL Baseline", {}).get("privacy_proxy")
    dna = by_method.get("FL + DNA", {}).get("privacy_proxy")
    if baseline is not None and dna is not None:
        if float(baseline) < float(dna):
            print("VERIFY: FL Baseline proxy is lower than FL + DNA proxy")
        else:
            print("WARNING: FL Baseline proxy is not lower than FL + DNA proxy")

    transform = by_method.get("FL + DNA Transform", {}).get("privacy_proxy")
    if baseline is not None and transform is not None:
        if float(baseline) < float(transform):
            print("VERIFY: FL Baseline proxy is lower than FL + DNA Transform proxy")
        else:
            print("WARNING: FL Baseline proxy is not lower than FL + DNA Transform proxy")


def main() -> None:
    attack_rows = load_attack_rows()
    utility_rows = load_utility_methods()
    rows = build_rows(utility_rows, attack_rows)
    write_outputs(rows)
    print_report(rows)
    verify(rows)
    print(f"Output JSON: {OUTPUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Output CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
