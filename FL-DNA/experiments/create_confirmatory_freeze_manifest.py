"""Create the RQ1/RQ2 confirmatory-freeze integrity manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT = ROOT / "protocols/confirmatory_freeze_manifest_2026-09-13.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    paths = [
        WORKSPACE / "PROJECT.md",
        ROOT / "protocols/rq1_confirmatory_protocol.md",
        ROOT / "protocols/rq2_multiseed_protocol.md",
        ROOT / "protocols/rq3_acceptance_criteria.md",
        ROOT / "protocols/supervisor_decision_package.md",
        ROOT / "protocols/stage1_execution_report_2026-09-12.md",
        ROOT / "protocols/confirmatory_freeze_report_2026-09-13.md",
        ROOT / "protocols/amendments/2026-09-12_rq1_distortion_grid_expansion.md",
        ROOT / "protocols/amendments/2026-09-13_stage1_supervisor_approvals.md",
        ROOT / "protocols/amendments/2026-09-13_rq2_multiplicity_and_execution_authorization.md",
        ROOT / "protocols/config/rq1_pre_pilot.yaml",
        ROOT / "protocols/config/rq2_pre_pilot.yaml",
        ROOT / "protocols/config/rq3_pre_pilot.yaml",
        ROOT / "protocols/config/rq1_confirmatory.yaml",
        ROOT / "protocols/config/rq2_confirmatory.yaml",
        ROOT / "attacks/inversion_metrics.py",
        ROOT / "attacks/pseudo_image.py",
        ROOT / "data/load_creditcard.py",
        ROOT / "privacy/dp_engine.py",
        ROOT / "experiments/fraud_fl_common.py",
        ROOT / "experiments/run_phase4_simple_defense_attack.py",
        ROOT / "experiments/run_phase4_dna_level1_forward_attack.py",
        ROOT / "experiments/create_phase4_source_disjoint_targets.py",
        ROOT / "experiments/verify_rq1_target_disjointness.py",
        ROOT / "experiments/measure_rq1_development_tie_rate.py",
        ROOT / "experiments/rq1_power_analysis.py",
        ROOT / "experiments/materialize_rq2_development_contract.py",
        ROOT / "experiments/run_fraud_fl_baseline.py",
        ROOT / "experiments/run_fraud_fl_dna.py",
        ROOT / "experiments/run_fraud_fl_dna_transform.py",
        ROOT / "experiments/run_fraud_fl_dp.py",
        ROOT / "experiments/run_rq1_confirmatory.py",
        ROOT / "experiments/analyze_rq1_confirmatory.py",
        ROOT / "experiments/run_rq2_confirmatory.py",
        ROOT / "experiments/analyze_rq2_confirmatory.py",
        ROOT / "datasets/creditcard.csv",
        ROOT / "artifacts/rq1/stage1_tie_threshold_20260912/tie_threshold_report.json",
        ROOT / "artifacts/rq1/stage1_selected_dp_tie_rate_20260913/tie_rate_report.json",
        ROOT / "artifacts/rq1/stage1_selected_dp_tie_rate_20260913/tie_rate_rows.csv",
        ROOT / "artifacts/rq1/confirmatory_power_analysis_20260913/rq1_power_analysis.json",
        ROOT / "artifacts/rq1/confirmatory_power_analysis_20260913/rq1_power_sensitivity.csv",
        ROOT / "artifacts/rq1/confirmatory_freeze_20260913/rq1_confirmatory_targets.pt",
        ROOT / "artifacts/rq1/confirmatory_freeze_20260913/rq1_confirmatory_targets.provenance.json",
        ROOT / "artifacts/rq1/confirmatory_freeze_20260913/source_overlap_matrix.json",
        ROOT / "artifacts/rq2/stage1_development_20260912/power_analysis.csv",
        ROOT / "artifacts/rq2/stage1_development_20260912/batch_summary.json",
        ROOT / "artifacts/rq2/confirmatory_freeze_20260913/per_seed_contract.json",
        ROOT / "experiments/create_confirmatory_freeze_manifest.py",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"freeze inputs missing: {missing}")
    entries = []
    for path in paths:
        entries.append({
            "path": str(path.relative_to(WORKSPACE)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "RQ1/RQ2 confirmatory execution inputs frozen before outcomes",
        "confirmatory_execution_authorized": True,
        "rq1_target_generation_count": 1,
        "rq1_target_overlap_gate": "PASS",
        "rq2_seed_count": 21,
        "rq3_status": "HELD_PENDING_NETWORK_EMULATION_TOOL",
        "git_head_commit": git("rev-parse", "HEAD"),
        "git_head_tree": git("rev-parse", "HEAD^{tree}"),
        "git_status_porcelain_sha256": hashlib.sha256(
            git("status", "--porcelain=v1").encode()
        ).hexdigest(),
        "file_count": len(entries),
        "files": entries,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "created_at", "git_head_commit", "git_head_tree", "file_count",
        "confirmatory_execution_authorized"
    )}, indent=2))


if __name__ == "__main__":
    main()
