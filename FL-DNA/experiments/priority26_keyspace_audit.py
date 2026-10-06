"""Priority 26 key-space audit for v1/v2 and seed-verification probe."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dna_encoder import transform_defense, transform_defense_v2  # noqa: E402
from dna_encoder.transform_defense_v2 import (  # noqa: E402
    DNATransformV2Config,
    DNATransformV2Metadata,
    reconstruct_update_array_v2,
    transform_update_array_v2,
)
from experiments.priority24_t1_bn_valid_rq1 import BN_KEY, capture_groups, load_population, mse_std, recover_mean  # noqa: E402
from privacy import seed_manager  # noqa: E402
from privacy.seed_manager import derive_seed  # noqa: E402


AMENDMENT = ROOT / "protocols/amendments/2026-09-29_priority26_v2sb_server_blind.md"
V2_BASE_SEED = 20260916


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_of(obj) -> dict:
    lines, start = inspect.getsourcelines(obj)
    return {"file": inspect.getsourcefile(obj), "start_line": start, "end_line": start + len(lines) - 1}


def wrong_metadata(original_meta: DNATransformV2Metadata, seed: int) -> DNATransformV2Metadata:
    sampled = transform_defense_v2._sampled_indices(original_meta.padded_size, original_meta.sketch_size, seed)
    return DNATransformV2Metadata(
        original_shape=original_meta.original_shape,
        original_size=original_meta.original_size,
        padded_size=original_meta.padded_size,
        sketch_size=original_meta.sketch_size,
        compression_ratio=original_meta.compression_ratio,
        quantization_eta=original_meta.quantization_eta,
        quantization_delta=original_meta.quantization_delta,
        seed=int(seed),
        tensor_index=original_meta.tensor_index,
        sampled_indices=tuple(int(i) for i in sampled.tolist()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026092601)
    parser.add_argument("--wrong-seeds", type=int, default=1000)
    args = parser.parse_args()

    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pop_mean, std, _features = load_population()
    captures = capture_groups(args.target.resolve(), args.seed, "priority26_keyspace")
    cfg = DNATransformV2Config(compression_ratio=0.95, quantization_eta=0.01, seed=V2_BASE_SEED)

    rows = []
    timings = []
    rng = np.random.default_rng(args.seed)
    for capture in captures:
        tensor_index = list(capture["observed"].keys()).index(BN_KEY)
        bn = capture["observed"][BN_KEY].detach().cpu().numpy().astype(np.float32)
        sketch, meta = transform_update_array_v2(bn, cfg, tensor_index=tensor_index, quantization_seed=derive_seed(args.seed, "q", capture["group_id"]))
        for _ in range(200):
            start = time.perf_counter()
            reconstruct_update_array_v2(sketch, meta)
            timings.append(time.perf_counter() - start)
        rec_true, _ = reconstruct_update_array_v2(sketch, meta)
        true_mean_rec = recover_mean(capture["model"], torch.from_numpy(rec_true))
        true_mse = mse_std(true_mean_rec, capture["true_mean"], std)
        wrong_mses = []
        better_or_equal_wrong = 0
        for _ in range(args.wrong_seeds):
            wrong_seed = int(rng.integers(1, 2**32 - 1))
            wm = wrong_metadata(meta, wrong_seed)
            rec_wrong, _ = reconstruct_update_array_v2(sketch, wm)
            mean_wrong = recover_mean(capture["model"], torch.from_numpy(rec_wrong))
            mse_wrong = mse_std(mean_wrong, capture["true_mean"], std)
            wrong_mses.append(mse_wrong)
            if mse_wrong <= true_mse:
                better_or_equal_wrong += 1
        rows.append(
            {
                "group": capture["group_id"],
                "source_ids": capture["source_ids"],
                "true_seed": meta.seed,
                "true_seed_mse": true_mse,
                "wrong_seed_count": args.wrong_seeds,
                "wrong_seed_min_mse": float(np.min(wrong_mses)),
                "wrong_seed_median_mse": float(np.median(wrong_mses)),
                "wrong_seed_better_or_equal_count": int(better_or_equal_wrong),
                "criterion_true_seed_rank": int(1 + np.sum(np.asarray(wrong_mses) < true_mse)),
            }
        )

    per_candidate = float(np.median(timings))
    projections = {
        "v2_2pow32_single_process_seconds": per_candidate * (2**32),
        "v2_2pow32_9_process_seconds": per_candidate * (2**32) / 9.0,
        "v1_2pow31_single_process_seconds": per_candidate * (2**31),
        "v1_2pow31_9_process_seconds": per_candidate * (2**31) / 9.0,
    }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "target": str(args.target.resolve().relative_to(ROOT)),
        "target_sha256": sha256(args.target.resolve()),
        "line_citations": {
            "v2_derive_seed": line_of(transform_defense_v2._derive_seed),
            "v2_rng": line_of(transform_defense_v2._rng),
            "v1_transform_update_array": line_of(transform_defense.transform_update_array),
            "v1_dna_block_seed_from_float32_block": line_of(transform_defense._dna_block_seed_from_float32_block),
            "derive_seed": line_of(seed_manager.derive_seed),
        },
        "effective_key_space": {
            "v2_rng_seed_space": "2^32 because _rng uses int(seed) % (2**32)",
            "v1_block_seed_space": "2^31 per block because _dna_block_seed_from_float32_block returns seed % (2**31) and transform_update_array passes that seed to np.random.default_rng",
            "derive_seed_range": "[1, 2**31 - 1]",
        },
        "decode_timing_seconds": {
            "median_per_candidate": per_candidate,
            "mean_per_candidate": float(np.mean(timings)),
            "n_timing_trials": len(timings),
        },
        "projected_enumeration": projections,
        "verification_criterion": "candidate seed score = standardized T1 batch-mean MSE after reconstructing BN running_mean; lower is better",
        "wrong_seeds_per_target": args.wrong_seeds,
        "rows": rows,
        "true_seed_rank_summary": {
            "all_true_seed_rank_1": bool(all(row["criterion_true_seed_rank"] == 1 for row in rows)),
            "max_true_seed_rank": int(max(row["criterion_true_seed_rank"] for row in rows)),
            "total_wrong_better_or_equal": int(sum(row["wrong_seed_better_or_equal_count"] for row in rows)),
        },
    }
    (output / "keyspace_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
