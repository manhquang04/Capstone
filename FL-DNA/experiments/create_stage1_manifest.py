"""Create a complete SHA-256 manifest for RQ1/RQ2 Stage-1 outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT = ROOT / "artifacts/stage1_20260912/stage1_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    source_paths = [
        WORKSPACE / "PROJECT.md",
        *sorted((ROOT / "protocols").glob("*.md")),
        *sorted((ROOT / "protocols/config").glob("*.yaml")),
        *sorted((ROOT / "protocols/amendments").glob("*.md")),
        ROOT / "protocols/pre_pilot_freeze_manifest.sha256",
        ROOT / "attacks/inversion_metrics.py",
        ROOT / "attacks/pseudo_image.py",
        ROOT / "data/load_creditcard.py",
        ROOT / "experiments/fraud_fl_common.py",
        ROOT / "experiments/run_phase4_simple_defense_attack.py",
        ROOT / "experiments/run_fraud_fl_dp.py",
        ROOT / "experiments/run_fraud_fl_dna_dp.py",
        ROOT / "experiments/run_fraud_fl_dna_transform.py",
        *sorted((ROOT / "experiments").glob("*rq1*.py")),
        *sorted((ROOT / "experiments").glob("*rq2*.py")),
        ROOT / "experiments/create_stage1_manifest.py",
        ROOT / "privacy/dp_engine.py",
        ROOT / "datasets/creditcard.csv",
    ]
    artifact_paths = [
        *sorted((ROOT / "artifacts/rq1").rglob("*")),
        *sorted((ROOT / "artifacts/rq2/stage1_development_20260912").rglob("*")),
    ]
    unique_files = sorted({path.resolve() for path in source_paths + artifact_paths if path.is_file()})
    entries = []
    for path in unique_files:
        try:
            name = str(path.relative_to(WORKSPACE))
        except ValueError:
            name = str(path)
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "RQ1/RQ2 Stage-1 source, protocol, config, logs and artifacts",
        "git_head_commit": git("rev-parse", "HEAD"),
        "git_head_tree": git("rev-parse", "HEAD^{tree}"),
        "git_status_porcelain_sha256": hashlib.sha256(git("status", "--porcelain=v1").encode()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "pycryptodome")
        },
        "file_count": len(entries),
        "files": entries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("created_at", "git_head_commit", "git_head_tree", "file_count")}, indent=2))


if __name__ == "__main__":
    main()
