"""Seal post-completion PROJECT bookkeeping without altering scientific artifacts."""
import hashlib
import json
import py_compile
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/priority34a"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    from experiments.verify_priority34a_completion import live_workers
    assert not live_workers(), "live training workload"
    manifest = OUT / "sha256_manifest_final.json"
    hashes = json.loads(manifest.read_text())
    for relative, digest in hashes.items():
        assert sha(ROOT / relative) == digest, relative
    receipt = json.loads((OUT / "COMPLETE.json").read_text())
    assert receipt["status"] == "PASS" and receipt["jobs"] == 189
    assert receipt["checkpoint_probability_replays_bit_exact"] == 1134
    snapshot = OUT / "PROJECT_pre_priority34a_completion.md"
    project = ROOT.parent / "PROJECT.md"
    assert project.read_bytes().startswith(snapshot.read_bytes()), "PROJECT history changed"
    assert "Priority34A verified completion" in project.read_text()
    result = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    py_compile.compile(str(Path(__file__).resolve()), cfile=str(OUT / "compile/administrative_seal.pyc"), doraise=True)
    verified = {"status": "PASS", "at": datetime.now(timezone.utc).isoformat(),
                "scientific_final_manifest_verified": len(hashes),
                "scientific_manifest_sha256": sha(manifest),
                "project_history_preserved": True, "project_before_sha256": sha(snapshot),
                "project_after_sha256": sha(project), "live_workers": [],
                "py_compile": "PASS", "git_diff_check": "PASS",
                "scope": "administrative only; no science replay or artifact overwrite"}
    output = OUT / "administrative_completion_receipt.json"
    assert not output.exists(), "never overwrite sealed receipt"
    output.write_text(json.dumps(verified, indent=2) + "\n")
    seal = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else "../PROJECT.md": sha(path)
            for path in (manifest, snapshot, project, Path(__file__).resolve(), output,
                         OUT / "compile/administrative_seal.pyc")}
    seal_path = OUT / "sha256_administrative_completion.json"
    assert not seal_path.exists(), "never overwrite administrative seal"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
