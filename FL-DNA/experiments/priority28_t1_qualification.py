"""Priority 28 T1 BN-statistics qualification runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import fraud_fl_common as common  # noqa: E402
from experiments.priority24_t1_bn_valid_rq1 import capture_groups, load_population, qualify  # noqa: E402


AMENDMENT = ROOT / "protocols/amendments/2026-09-30_priority28_defense_audit.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pop_mean, std, feature_names = load_population()
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "seed": int(args.seed),
        "torch_num_threads": torch.get_num_threads(),
        "instrument": "T1_BN_RUNNING_MEAN_BATCH_MEAN_RECOVERY",
        "stages": {},
        "feature_names": feature_names,
    }
    for stage, target in (("n8", args.n8_target), ("n24", args.n24_target)):
        captures = capture_groups(target.resolve(), args.seed, f"priority28_t1_{stage}")
        result = qualify(captures, pop_mean, std)
        csv_path = out / f"t1_{stage}_qualification.csv"
        write_csv(csv_path, result["rows"])
        payload["stages"][stage] = {
            "target": str(target.resolve().relative_to(ROOT)),
            "target_sha256": sha256(target.resolve()),
            "rows_csv": str(csv_path.relative_to(ROOT)),
            "rows_csv_sha256": sha256(csv_path),
            "summary": result["summary"],
        }
    n8_ok = bool(payload["stages"]["n8"]["summary"]["qualified"])
    n24_ok = bool(payload["stages"]["n24"]["summary"]["qualified"])
    payload["qualified"] = n8_ok and n24_ok
    dump(out / "t1_qualification_summary.json", payload)
    dump(out / "manifest.json", {"summary_sha256": sha256(out / "t1_qualification_summary.json")})
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n8-target", type=Path, required=True)
    parser.add_argument("--n24-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026093002)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
