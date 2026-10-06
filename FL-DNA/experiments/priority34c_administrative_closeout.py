"""Administrative close-out only: never imports or reruns research drivers."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/priority34c"
PROJECT = ROOT.parent / "PROJECT.md"
SNAPSHOT = OUT / "PROJECT_pre_priority34c_completion.md"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def audit():
    complete = json.loads((OUT / "COMPLETE.json").read_text())
    assert complete["status"] == "PASS" and complete["confirmatory_jobs"] == 1326
    manifest = OUT / "sha256_manifest.csv"
    assert sha(manifest) == complete["manifest_sha256"]
    rows = list(csv.DictReader(manifest.open()))
    assert len({r["path"] for r in rows}) == len(rows)
    for row in rows:
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    assert sha(ROOT / complete["report"]) == complete["report_sha256"]
    lines = subprocess.check_output(["ps", "-axo", "pid,command"], text=True).splitlines()
    live = [line for line in lines if any(name in line for name in (
        "priority34c_qualify.py", "priority34c_development.py",
        "priority34c_confirmatory.py", "analyze_priority34c.py"))]
    assert not live, "P34C process remains active"
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    tests = json.loads((OUT / "independent_statistics.json").read_text())["tests"]
    assert len(tests) == 72
    return dict(manifest_entries=len(rows), manifest_sha256=sha(manifest),
                report_sha256=complete["report_sha256"], confirmatory_jobs=1326,
                qualification_jobs=264, development_jobs=192, fixed_tests=72,
                assessable_directional_tests=sum(r["status"] == "ASSESSABLE" for r in tests),
                significant_dna_higher=sum(r["direction"] == "dna_higher_recovery" and r["holm_p"] < .05 for r in tests),
                significant_dna_lower=sum(r["direction"] == "dna_lower_recovery" and r["holm_p"] < .05 for r in tests),
                no_live_workers=True, git_diff_check="PASS", status="PASS")


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    assert args.snapshot != args.seal
    receipt = audit()
    receipt["at"] = datetime.now(timezone.utc).isoformat()
    if args.snapshot:
        assert not SNAPSHOT.exists()
        shutil.copyfile(PROJECT, SNAPSHOT)
        receipt["project_snapshot_sha256"] = sha(SNAPSHOT)
        write_new(OUT / "final_verified_receipt.json", receipt)
    else:
        assert SNAPSHOT.exists()
        verified = json.loads((OUT / "final_verified_receipt.json").read_text())
        assert sha(SNAPSHOT) == verified["project_snapshot_sha256"]
        receipt["files"] = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p)
                            for p in (SNAPSHOT, PROJECT, OUT / "final_verified_receipt.json",
                                      OUT / "COMPLETE.json", Path(__file__).resolve(),
                                      ROOT / "reports/priority34c_verified_summary.md")}
        write_new(OUT / "administrative_closeout_seal.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
