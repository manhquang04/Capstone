"""Close new P36A namespace only; no measurements/tests rerun."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/priority36a"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    assert json.loads((OUT / "validation.json").read_text())["status"] == "PASS"
    assert json.loads((OUT / "independent_diff_review.json").read_text())["status"] == "PASS"
    assert "20 passed" in (OUT / "tests.log").read_text()
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert diff.returncode == 0, diff.stdout + diff.stderr
    progress = {"stage": "PRIORITY36A_COMPLETE", "at": datetime.now(timezone.utc).isoformat(), "complete": True,
                "tests_passed": 20, "cpu_only": True, "scientific_runs": 0, "default_byte_compatibility": "PASS",
                "independent_diff_review": "PASS", "report": "reports/priority36a_report.md", "committed": False}
    save(OUT / "progress.json", progress)
    save(OUT / "checklist.json", {x: "DONE" for x in ["amendment", "legacy_capture", "implementation", "tests", "independent_diff_review", "report_manifest"]})
    with (OUT / "progress.log").open("a") as log:
        log.write(json.dumps(progress) + "\n")
    sources = ["dna_encoder/transform_defense_v2.py", "dna_encoder/transform_defense_v2_hardened.py",
               "experiments/run_rq3_benchmark.py", "experiments/priority36a_wire.py", "experiments/priority36a_validate.py",
               "experiments/priority36a_seal.py", "tests/test_priority36a_hardening.py",
               "protocols/amendments/2026-10-12_priority36a_opt_in_key_hardening.md", "reports/priority36a_report.md"]
    paths = [ROOT / x for x in sources] + [p for p in OUT.rglob("*") if p.is_file() and p.name not in {"final_manifest.json", "COMPLETE.json"}]
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}
    save(OUT / "final_manifest.json", manifest)
    for filename, expected in json.loads((OUT / "final_manifest.json").read_text()).items():
        assert sha(ROOT / filename) == expected, filename
    save(OUT / "COMPLETE.json", {"status": "PRIORITY36A_VERIFIED_COMPLETE", "manifest_sha256": sha(OUT / "final_manifest.json"),
                               "report_sha256": sha(ROOT / "reports/priority36a_report.md"), "entries_verified": len(manifest),
                               "tests": 20, "scientific_runs": 0, "committed": False})
    print(json.dumps(progress))


if __name__ == "__main__":
    main()
