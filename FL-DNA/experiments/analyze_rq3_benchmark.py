"""Analyze the frozen RQ3 matrix and apply Bundle B without retuning."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _p95(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.95, method="linear"))


def _bootstrap_p95_ci(values: np.ndarray, *, seed: int, samples: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = np.quantile(values[indices], 0.95, axis=1, method="linear")
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "p95": _p95(values),
    }


def _load_cells(run: Path) -> dict[tuple[str, str, int], list[dict]]:
    cells = {}
    for path in sorted((run / "cells").glob("*.json")):
        if path.name.endswith(".log.json"):
            continue
        document = json.loads(path.read_text())
        rows = [row for row in document["rows"] if row["phase"] == "measured"]
        key = (document["profile"], document["method"], int(document["client_count"]))
        if len(rows) != 50:
            raise ValueError(f"{path} has {len(rows)} measured rows, expected 50")
        cells[key] = sorted(rows, key=lambda row: row["measured_index"])
    return cells


def _array(rows: list[dict], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def _criterion(name: str, statistic: float, ci: tuple[float, float] | None, threshold: float) -> dict:
    upper = ci[1] if ci else statistic
    return {"criterion": name, "statistic": statistic, "ci95": list(ci) if ci else None, "threshold": threshold, "pass": bool(upper <= threshold)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cells = _load_cells(args.run)
    expected = len(config["network_profiles"]) * len(config["benchmark"]["methods"]) * len(config["benchmark"]["client_counts"])
    if len(cells) != expected:
        raise ValueError(f"found {len(cells)} cells, expected {expected}")
    thresholds = config["acceptance_thresholds"]
    bootstrap = config["uncertainty"]
    summaries = []
    decisions = []
    for (profile, method, count), rows in sorted(cells.items()):
        for field in (
            "client_encode_serialize_seconds_max", "client_encode_serialize_seconds_total",
            "server_decode_aggregate_seconds", "transfer_seconds", "modeled_transfer_seconds",
            "end_to_end_seconds", "payload_bytes_total", "client_peak_alloc_bytes", "server_peak_alloc_bytes",
        ):
            summaries.append({"profile": profile, "method": method, "client_count": count, "metric": field, **_summary(_array(rows, field))})
        if method == "RAW_FLOAT32":
            continue
        raw = cells[(profile, "RAW_FLOAT32", count)]
        seed_base = int.from_bytes(f"{profile}|{method}|{count}".encode(), "little") % (2**32)
        e2e = (_array(rows, "end_to_end_seconds") - _array(raw, "end_to_end_seconds")) / _array(raw, "end_to_end_seconds")
        client = _array(rows, "client_encode_serialize_seconds_max") * 1000.0
        server = _array(rows, "server_decode_aggregate_seconds") * 1000.0
        client_mem_delta = (_array(rows, "client_peak_alloc_bytes") - _array(raw, "client_peak_alloc_bytes")) / (1024.0**2)
        client_mem_fraction = (_array(rows, "client_peak_alloc_bytes") - _array(raw, "client_peak_alloc_bytes")) / np.maximum(_array(raw, "client_peak_alloc_bytes"), 1.0)
        server_mem_delta = (_array(rows, "server_peak_alloc_bytes") - _array(raw, "server_peak_alloc_bytes")) / (1024.0**2)
        server_mem_fraction = (_array(rows, "server_peak_alloc_bytes") - _array(raw, "server_peak_alloc_bytes")) / np.maximum(_array(raw, "server_peak_alloc_bytes"), 1.0)
        added_transfer = _array(rows, "transfer_seconds") - _array(raw, "transfer_seconds")
        vectors = [
            ("end_to_end_round_overhead_p95_fraction", e2e, thresholds["end_to_end_round_overhead_p95_fraction"]),
            ("client_encode_serialize_p95_ms", client, thresholds["client_encode_serialize_p95_ms"]),
            ("server_decode_aggregate_p95_ms", server, thresholds["server_decode_aggregate_p95_ms"]),
            ("peak_client_memory_overhead_mb", client_mem_delta, thresholds["peak_client_memory_overhead_mb"]),
            ("peak_client_memory_overhead_fraction", client_mem_fraction, thresholds["peak_client_memory_overhead_fraction"]),
            ("peak_server_memory_overhead_mb", server_mem_delta, thresholds["peak_server_memory_overhead_mb"]),
            ("peak_server_memory_overhead_fraction", server_mem_fraction, thresholds["peak_server_memory_overhead_fraction"]),
            ("added_transfer_latency_p95_seconds", added_transfer, thresholds["added_transfer_latency_p95_seconds"][profile]),
        ]
        criteria = []
        for offset, (name, values, threshold) in enumerate(vectors):
            ci = _bootstrap_p95_ci(values, seed=bootstrap["seed"] + seed_base + offset, samples=bootstrap["resamples"])
            criteria.append(_criterion(name, _p95(values), ci, threshold))
        payload_ratio = rows[0]["payload_bytes_total"] / raw[0]["payload_bytes_total"]
        criteria.append(_criterion("application_payload_expansion_ratio", payload_ratio, None, thresholds["application_payload_expansion_ratio"]))
        failures = sum(bool(row["authentication_or_correctness_failure"]) for row in rows)
        timeouts = sum(bool(row["timed_out"]) for row in rows)
        criteria.append(_criterion("authentication_or_correctness_failures", failures, None, thresholds["authentication_or_correctness_failures"]))
        criteria.append(_criterion("timeout_rate_fraction", timeouts / len(rows), None, thresholds["timeout_rate_fraction"]))
        decisions.append({
            "profile": profile,
            "method": method,
            "client_count": count,
            "criteria": criteria,
            "decision": "ACCEPTABLE" if all(item["pass"] for item in criteria) else "NOT_ACCEPTABLE",
        })

    profiles = {}
    for profile in config["network_profiles"]:
        members = [row for row in decisions if row["profile"] == profile]
        profiles[profile] = "ACCEPTABLE" if members and all(row["decision"] == "ACCEPTABLE" for row in members) else "NOT_ACCEPTABLE"
    missing = expected - len(cells)
    overall = "INCONCLUSIVE" if missing else ("ACCEPTABLE" if all(value == "ACCEPTABLE" for value in profiles.values()) else "NOT_ACCEPTABLE")
    decision_document = {"overall_decision": overall, "profile_decisions": profiles, "missing_cells": missing, "cell_decisions": decisions}
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "acceptance_decision.json").write_text(json.dumps(decision_document, indent=2, sort_keys=True) + "\n")
    with (args.results / "benchmark_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    with (args.results / "acceptance_criteria.csv").open("w", newline="") as handle:
        flat = []
        for decision in decisions:
            for criterion in decision["criteria"]:
                flat.append({"profile": decision["profile"], "method": decision["method"], "client_count": decision["client_count"], "cell_decision": decision["decision"], **criterion, "ci95": json.dumps(criterion["ci95"])})
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    print(json.dumps({"overall_decision": overall, "profile_decisions": profiles, "cells": len(cells)}, indent=2))


if __name__ == "__main__":
    main()

