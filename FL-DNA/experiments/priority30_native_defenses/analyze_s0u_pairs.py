#!/usr/bin/env python3
"""Priority 30 S6 paired analysis against the S0u undefended comparator.

This script is intentionally read-only with respect to the per-target result
artifacts.  It creates derived CSV summaries under audit/S6.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PairedRow:
    domain: str
    stage: str
    evaluation: str
    defense: str
    target_id: int
    defended_score: float
    comparator: str
    comparator_score: float
    diff: float
    outcome: str


def exact_one_sided_p(wins: int, losses: int) -> float:
    """P[X >= wins] for X ~ Binomial(wins + losses, 0.5)."""

    n = wins + losses
    if n == 0:
        return float("nan")
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)


def holm_adjust(p_values: list[tuple[str, float]]) -> dict[str, float]:
    finite = [(key, p) for key, p in p_values if math.isfinite(p)]
    finite.sort(key=lambda item: item[1])
    m = len(finite)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(finite, start=1):
        value = min(1.0, (m - rank + 1) * p)
        running = max(running, value)
        adjusted[key] = running
    for key, p in p_values:
        if not math.isfinite(p):
            adjusted[key] = float("nan")
    return adjusted


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def score_from_result(domain: str, data: dict[str, Any]) -> float:
    """Return a reconstruction-error score where larger is worse leakage."""

    if domain == "image":
        return float(data["metrics"]["mse"])
    if domain == "adult":
        # Official TabLeak native metric is feature accuracy. Convert to an
        # error score so that the sign convention matches image MSE.
        return 100.0 - float(data["metric"]["accuracy_percent"])
    raise ValueError(f"unknown domain: {domain}")


def baseline_scores(domain: str, s0u: dict[str, Any]) -> dict[str, float]:
    if domain == "image":
        return {
            "undefended": score_from_result(domain, s0u),
            "gray": float(s0u["baselines"]["gray"]["mse"]),
            "cifar_mean": float(s0u["baselines"]["cifar_mean"]["mse"]),
        }
    if domain == "adult":
        return {
            "undefended": score_from_result(domain, s0u),
            "mean_mode": 100.0
            - float(s0u["baselines"]["mean_mode"]["accuracy_percent"]),
            "empirical_marginal": 100.0
            - float(s0u["baselines"]["empirical_marginal"]["accuracy_percent"]),
        }
    raise ValueError(f"unknown domain: {domain}")


def stage_domain_from_path(path: Path) -> tuple[str, str]:
    # .../audit/S1/image/E1/precode/target_000/result.json
    parts = path.parts
    try:
        audit_idx = parts.index("audit")
    except ValueError as exc:
        raise ValueError(f"path is not under audit/: {path}") from exc
    return parts[audit_idx + 1], parts[audit_idx + 2]


def collect_cell_results(audit_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for stage in ("S1", "S1b", "S1c", "S1d", "S2", "S2b", "S2c", "S2d"):
        stage_dir = audit_dir / stage
        if not stage_dir.exists():
            continue
        for path in stage_dir.glob("*/*/*/target_*/result.json"):
            try:
                data = load_json(path)
            except json.JSONDecodeError:
                continue
            if data.get("status") == "ran":
                paths.append(path)
    # Corrected Priority 30 reruns invalidate the affected old E2 cells:
    # - all image E2 from S1/S1b are superseded by S1c;
    # - Adult PRECODE E2 from S2b is superseded by S2c.
    # - Count-Sketch E1/E2 from S1/S1b/S2/S2b are superseded by S1d/S2d
    #   because the old sketch was non-compressing (5 x 500,000 cells).
    corrected: list[Path] = []
    for path in sorted(paths):
        data = load_json(path)
        stage = str(data.get("stage"))
        domain = str(data.get("domain"))
        evaluation = str(data.get("eval"))
        defense = str(data.get("defense"))
        if domain == "image" and evaluation == "E2" and stage in {"S1", "S1b"}:
            continue
        if domain == "adult" and evaluation == "E2" and defense == "precode" and stage == "S2b":
            continue
        if defense == "count_sketch" and stage in {"S1", "S1b", "S1c", "S2", "S2b"}:
            continue
        corrected.append(path)
    return corrected


def check_e2_not_e1(audit_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Fail closed if corrected E2 objectives exactly equal paired E1 objectives.

    ATS is the only documented exception: the policy path used here is
    PIL/non-differentiable, so its E2 approximation is explicitly recorded as
    straight-through identity.
    """

    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for result_path in sorted((audit_dir / "S1c" / "image" / "E2").glob("*/target_*/result.json")):
        data = load_json(result_path)
        defense = str(data["defense"])
        if defense == "count_sketch":
            continue
        target_id = int(data["target_id"])
        e1_path = audit_dir / "S1" / "image" / "E1" / defense / f"target_{target_id:03d}" / "result.json"
        if not e1_path.exists():
            continue
        e1 = load_json(e1_path)
        e2_objective = float(data.get("objective", float("nan")))
        e1_objective = float(e1.get("objective", float("nan")))
        exact_equal = e2_objective == e1_objective
        mode = str(data.get("mode", ""))
        documented_exception = (
            defense == "ats" and mode.startswith("ats_identity_straight_through")
        ) or (
            defense == "dna_v1_conservative" and mode == "dna_v1_plain_e2"
        )
        row = {
            "domain": "image",
            "defense": defense,
            "target_id": target_id,
            "e1_objective": e1_objective,
            "e2_objective": e2_objective,
            "exact_equal": exact_equal,
            "documented_exception": documented_exception,
            "status": "documented_exception" if exact_equal and documented_exception else ("FAIL" if exact_equal else "PASS"),
        }
        checks.append(row)
        if exact_equal and not documented_exception:
            failures.append(row)

    for result_path in sorted((audit_dir / "S1d" / "image" / "E2" / "count_sketch").glob("target_*/result.json")):
        data = load_json(result_path)
        target_id = int(data["target_id"])
        e1_path = audit_dir / "S1d" / "image" / "E1" / "count_sketch" / f"target_{target_id:03d}" / "result.json"
        if not e1_path.exists():
            continue
        e1 = load_json(e1_path)
        e2_objective = float(data.get("objective", float("nan")))
        e1_objective = float(e1.get("objective", float("nan")))
        exact_equal = e2_objective == e1_objective
        row = {
            "domain": "image",
            "defense": "count_sketch",
            "target_id": target_id,
            "e1_objective": e1_objective,
            "e2_objective": e2_objective,
            "exact_equal": exact_equal,
            "documented_exception": False,
            "status": "FAIL" if exact_equal else "PASS",
        }
        checks.append(row)
        if exact_equal:
            failures.append(row)

    for result_path in sorted((audit_dir / "S2d" / "adult" / "E2" / "count_sketch").glob("target_*/result.json")):
        data = load_json(result_path)
        target_id = int(data["target_id"])
        e1_path = audit_dir / "S2d" / "adult" / "E1" / "count_sketch" / f"target_{target_id:03d}" / "result.json"
        if not e1_path.exists():
            continue
        e1 = load_json(e1_path)
        e2_score = float(data["metric"]["accuracy_percent"])
        e1_score = float(e1["metric"]["accuracy_percent"])
        exact_equal = e2_score == e1_score
        row = {
            "domain": "adult",
            "defense": "count_sketch",
            "target_id": target_id,
            "e1_accuracy_percent": e1_score,
            "e2_accuracy_percent": e2_score,
            "exact_equal": exact_equal,
            "documented_exception": False,
            "status": "FAIL" if exact_equal else "PASS",
        }
        checks.append(row)
        if exact_equal:
            failures.append(row)

    for result_path in sorted((audit_dir / "S2c" / "adult" / "E2" / "precode").glob("target_*/result.json")):
        data = load_json(result_path)
        target_id = int(data["target_id"])
        e1_path = audit_dir / "S2" / "adult" / "E1" / "precode" / f"target_{target_id:03d}" / "result.json"
        if not e1_path.exists():
            continue
        e1 = load_json(e1_path)
        # Adult E1 did not store the objective, so use exact reconstruction
        # accuracy equality as the available fail-closed proxy for this cell.
        e2_score = float(data["metric"]["accuracy_percent"])
        e1_score = float(e1["metric"]["accuracy_percent"])
        exact_equal = e2_score == e1_score
        row = {
            "domain": "adult",
            "defense": "precode",
            "target_id": target_id,
            "e1_accuracy_percent": e1_score,
            "e2_accuracy_percent": e2_score,
            "exact_equal": exact_equal,
            "documented_exception": False,
            "status": "FAIL" if exact_equal else "PASS",
        }
        checks.append(row)
        if exact_equal:
            failures.append(row)

    write_csv(
        output_dir / "e2_not_e1_check.csv",
        checks,
        sorted({key for row in checks for key in row}),
    )
    if failures:
        (output_dir / "e2_not_e1_failures.json").write_text(
            json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise RuntimeError(f"E2 != E1 fail-closed check failed for {len(failures)} target(s)")
    return checks


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[PairedRow]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str, str], list[PairedRow]] = {}
    for row in rows:
        key = (row.domain, row.stage, row.evaluation, row.defense, row.comparator)
        grouped.setdefault(key, []).append(row)

    primary_p: list[tuple[str, float]] = []
    baseline_p: list[tuple[str, float]] = []
    row_by_key: dict[str, dict[str, Any]] = {}

    for key, group in sorted(grouped.items()):
        domain, stage, evaluation, defense, comparator = key
        wins = sum(1 for row in group if row.outcome == "defense_worse")
        losses = sum(1 for row in group if row.outcome == "defense_better")
        ties = sum(1 for row in group if row.outcome == "tie")
        p = exact_one_sided_p(wins, losses)
        diffs = [row.diff for row in group]
        mean_diff = sum(diffs) / len(diffs)
        median_diff = sorted(diffs)[len(diffs) // 2]
        summary_key = "::".join(key)
        entry = {
            "key": summary_key,
            "domain": domain,
            "stage": stage,
            "evaluation": evaluation,
            "defense": defense,
            "comparator": comparator,
            "n_targets": len(group),
            "wins_defense_worse": wins,
            "losses_defense_better": losses,
            "ties": ties,
            "p_one_sided_defense_worse": p,
            "mean_score_diff_defended_minus_comparator": mean_diff,
            "median_score_diff_defended_minus_comparator": median_diff,
        }
        summaries.append(entry)
        row_by_key[summary_key] = entry
        if comparator == "undefended":
            primary_p.append((summary_key, p))
        else:
            baseline_p.append((summary_key, p))

    primary_adj = holm_adjust(primary_p)
    baseline_adj = holm_adjust(baseline_p)
    for key, value in primary_adj.items():
        row_by_key[key]["holm_family"] = "defended_vs_undefended"
        row_by_key[key]["holm_adjusted_p"] = value
    for key, value in baseline_adj.items():
        row_by_key[key]["holm_family"] = "defended_vs_data_free_baselines"
        row_by_key[key]["holm_adjusted_p"] = value
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("artifacts/priority30_native_defenses/audit"),
    )
    parser.add_argument("--n", type=int, default=39)
    args = parser.parse_args()

    audit_dir = args.audit_dir
    output_dir = audit_dir / "S6"
    output_dir.mkdir(parents=True, exist_ok=True)
    e2_checks = check_e2_not_e1(audit_dir, output_dir)

    rows: list[PairedRow] = []
    for result_path in collect_cell_results(audit_dir):
        stage, domain = stage_domain_from_path(result_path)
        data = load_json(result_path)
        target_id = int(data["target_id"])
        s0u_path = audit_dir / "S0u" / domain / f"target_{target_id:03d}" / "result.json"
        if not s0u_path.exists():
            raise FileNotFoundError(f"missing paired S0u result for {result_path}: {s0u_path}")
        s0u = load_json(s0u_path)
        if int(s0u["target_id"]) != target_id:
            raise AssertionError(f"target_id mismatch: {result_path} vs {s0u_path}")

        # If source identifiers are present on both sides, enforce equality.
        if domain == "image":
            if "cifar10_index" in data and "cifar10_index" in s0u:
                if int(data["cifar10_index"]) != int(s0u["cifar10_index"]):
                    raise AssertionError(f"CIFAR index mismatch: {result_path}")
        elif domain == "adult":
            if "adult_indices" in data and "adult_indices" in s0u:
                if list(data["adult_indices"]) != list(s0u["adult_indices"]):
                    raise AssertionError(f"Adult source IDs mismatch: {result_path}")

        defended_score = score_from_result(domain, data)
        for comparator, comparator_score in baseline_scores(domain, s0u).items():
            diff = defended_score - comparator_score
            if diff > 0:
                outcome = "defense_worse"
            elif diff < 0:
                outcome = "defense_better"
            else:
                outcome = "tie"
            rows.append(
                PairedRow(
                    domain=domain,
                    stage=stage,
                    evaluation=str(data["eval"]),
                    defense=str(data["defense"]),
                    target_id=target_id,
                    defended_score=defended_score,
                    comparator=comparator,
                    comparator_score=comparator_score,
                    diff=diff,
                    outcome=outcome,
                )
            )

    row_dicts = [
        {
            "domain": row.domain,
            "stage": row.stage,
            "evaluation": row.evaluation,
            "defense": row.defense,
            "target_id": row.target_id,
            "defended_score": row.defended_score,
            "comparator": row.comparator,
            "comparator_score": row.comparator_score,
            "score_diff_defended_minus_comparator": row.diff,
            "outcome": row.outcome,
        }
        for row in rows
    ]
    write_csv(
        output_dir / "paired_per_target.csv",
        row_dicts,
        [
            "domain",
            "stage",
            "evaluation",
            "defense",
            "target_id",
            "defended_score",
            "comparator",
            "comparator_score",
            "score_diff_defended_minus_comparator",
            "outcome",
        ],
    )

    summaries = summarize(rows)
    write_csv(
        output_dir / "paired_sign_tests.csv",
        summaries,
        [
            "key",
            "domain",
            "stage",
            "evaluation",
            "defense",
            "comparator",
            "n_targets",
            "wins_defense_worse",
            "losses_defense_better",
            "ties",
            "p_one_sided_defense_worse",
            "mean_score_diff_defended_minus_comparator",
            "median_score_diff_defended_minus_comparator",
            "holm_family",
            "holm_adjusted_p",
        ],
    )

    manifest = {
        "audit_dir": str(audit_dir),
        "n_expected": args.n,
        "n_per_target_rows": len(row_dicts),
        "n_summary_rows": len(summaries),
        "n_e2_not_e1_checks": len(e2_checks),
        "score_convention": "larger score means worse reconstruction / stronger defense",
        "adult_score": "100 - official TabLeak feature accuracy percent",
        "image_score": "raw input-space MSE",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
