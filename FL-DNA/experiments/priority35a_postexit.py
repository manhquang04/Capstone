"""Administrative qualification post-exit audit; no scientific measurements."""
import os
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.priority35a_runtime import Q, OUT, load, sha, check_hashes, freeze_check, write, stage, now


def main():
    freeze_check()
    completion = load(Q / "VERIFIED.json")
    manifest = load(Q / "final_manifest.json")
    check_hashes(manifest["hashes"])
    for key, name in (("table_sha256", "qualification_table.json"),
                      ("report_sha256", "qualification_summary.md"), ("manifest_sha256", "final_manifest.json")):
        if completion[key] != sha(Q / name):
            raise ValueError("qualification completion digest mismatch")
    launch = load(Q / "launch_receipt.json")
    pids = [launch["pid"]]
    for path in Q.glob("worker_pool_*.json"):
        pids.extend(load(path)["worker_pids"])
    process = subprocess.run(["ps", "-p", ",".join(map(str, pids)), "-o", "pid,args"], capture_output=True, text=True)
    if len(process.stdout.strip().splitlines()) > 1:
        raise ValueError("recorded scientific process is still live")
    all_processes = subprocess.check_output(["ps", "-axo", "pid,args"], text=True)
    for line in all_processes.splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[0] != str(os.getpid()) and "python" in Path(fields[1]).name.lower() and "priority35a" in line:
            raise ValueError("another Priority35A Python process is alive")
    if Path(ROOT / launch["stderr"]).read_text():
        raise ValueError("detached stderr is not empty")
    failures = [p for p in Q.rglob("*.json") if "FAILURE" in p.name or "REQUIRES_DIRECTION" in p.name or p.name == "failure.json"]
    if failures:
        raise ValueError("preserved qualification failure requires direction")
    check = load(OUT / "checklist.json")
    check.update(no_live_scientific_workers=True, requires_post_exit_audit=False,
                 qualification_post_exit_hash_audit_pass=True)
    write(OUT / "checklist.json", check)
    stage("qualification_verified_waiting_user", status="QUALIFICATION_VERIFIED_WAITING_USER",
          no_live_scientific_workers=True, qualification_complete=True, qualified_cells=0,
          qualification_post_exit_hash_audit_pass=True, full_priority35a_complete=False)
    paths = [ROOT / launch["stdout"], ROOT / launch["stderr"], Q / "VERIFIED.json",
             OUT / "progress.json", OUT / "progress.log", OUT / "checklist.json", Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    write(OUT / "qualification_post_exit_seal_20261010.json", dict(at=now(), hashes=hashes,
          original_manifest_preserved=True, scientific_manifest_files=len(manifest["hashes"]),
          scientific_manifest_sha256=sha(Q / "final_manifest.json"), no_live_scientific_workers=True,
          scope="administrative qualification-only seal, not full Priority35A COMPLETE"), exclusive=True)
    check_hashes(hashes)
    print("Qualification post-exit audit PASS; original manifest preserved.")


if __name__ == "__main__":
    main()
