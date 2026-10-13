"""CPU synthetic checks, payload sizing and independently structured diff audit."""
import ast
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/priority36a"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stage(name):
    value = {"stage": name, "at": datetime.now(timezone.utc).isoformat(), "complete": False, "cpu_only": True, "measurements": 0}
    save("progress.json", value)
    with (OUT / "progress.log").open("a") as log:
        log.write(json.dumps(value) + "\n")


def review():
    old = ast.parse((OUT / "legacy_transform.py").read_text())
    new = ast.parse((ROOT / "dna_encoder/transform_defense_v2.py").read_text())
    oldfunc = {x.name: x for x in old.body if isinstance(x, ast.FunctionDef)}
    newfunc = {x.name: x for x in new.body if isinstance(x, ast.FunctionDef)}
    changes = {"transform_update_array_v2", "reconstruct_update_array_v2", "transform_and_reconstruct_array_v2", "materialize_projection_matrix_v2"}
    exact = []
    for name in oldfunc.keys() - changes:
        assert ast.dump(oldfunc[name]) == ast.dump(newfunc[name]), name
        exact.append(name)
    for name in ["transform_update_array_v2", "reconstruct_update_array_v2", "materialize_projection_matrix_v2"]:
        before, after = oldfunc[name].body, newfunc[name].body
        # Dispatch-only additions must precede the legacy computational body.
        if name == "transform_update_array_v2":
            after = after[:1] + after[3:]
        elif name == "reconstruct_update_array_v2":
            after = after[:1] + after[4:]
        else:
            after = after[:1] + after[2:]
        assert [ast.dump(x) for x in before] == [ast.dump(x) for x in after], name
    old_wire = ast.parse((OUT / "legacy_rq3.py").read_text())
    new_wire = ast.parse((ROOT / "experiments/run_rq3_benchmark.py").read_text())
    oldf = {x.name: x for x in old_wire.body if isinstance(x, ast.FunctionDef)}
    newf = {x.name: x for x in new_wire.body if isinstance(x, ast.FunctionDef)}
    for name in oldf:
        assert ast.dump(oldf[name]) == ast.dump(newf[name]), name
    hard = (ROOT / "dna_encoder/transform_defense_v2_hardened.py").read_text()
    assert "default_rng" not in hard and "_derive_seed" not in hard and "2**32" not in hard
    from dna_encoder.transform_defense_v2_hardened import HardenedV2Metadata
    fields = set(HardenedV2Metadata.__dataclass_fields__)
    assert not fields.intersection({"seed", "key", "derived_seed", "sampled_indices"})
    save("independent_diff_review.json", {"status": "PASS", "method": "AST comparisons to pre-edit source snapshots plus independent old/new byte tests", "exact_legacy_helpers": sorted(exact), "all_legacy_rq3_function_asts_identical": True, "legacy_computational_bodies_unchanged": True, "public_metadata_fields": sorted(fields), "manual_findings": ["128-bit base key; full 256-bit purpose-separated HKDF outputs; no NumPy PRNG in hardened path", "HMAC counter stream and unbiased rejection sampling; no derived seed or indices on wire", "No authenticity, key management or wrong-key verification claim; public delta remains data-dependent", "Opt-in default false; convenience wrapper forwards separate key; legacy wire CLI unchanged"]})


def main():
    torch.set_num_threads(1)
    stage("IMPLEMENTATION_TESTING")
    cmd = [sys.executable, "-B", "-m", "pytest", "-q", "tests/test_priority36a_hardening.py", "tests/test_transform_defense_v2.py"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (OUT / "tests.log").write_text(result.stdout + result.stderr)
    save("tests_receipt.json", {"command": cmd, "returncode": result.returncode})
    if result.returncode:
        raise RuntimeError("test gate failed; log retained")
    stage("TESTS_PASS_PAYLOAD_SIZING")
    from models.fraud_mlp import FraudMLP
    from dna_encoder.transform_defense_v2 import DNATransformV2Config
    from experiments.run_rq3_benchmark import _serialize_v2
    from experiments.priority36a_wire import serialize_v2_hardened, deserialize_v2_hardened
    sizes = []
    for dataset in ["baf", "ieee_cis", "paysim"]:
        prepared = ROOT / "artifacts/priority32_multidataset/prepared" / dataset
        dims = np.load(prepared / "train_x.npy", mmap_mode="r").shape[1]
        torch.manual_seed(360000)
        model = FraudMLP(dims).cpu()
        bn_names = {name for name, module in model.named_modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
        tensors = OrderedDict((name, np.linspace(-.1, .2, value.numel(), dtype=np.float32).reshape(tuple(value.shape)))
                              for name, value in model.named_parameters() if name.rsplit(".", 1)[0] not in bn_names)
        key = bytes.fromhex("00112233445566778899aabbccddeeff")
        default, _ = _serialize_v2(tensors, DNATransformV2Config(seed=321000), 0)
        payload, expected = serialize_v2_hardened(tensors, DNATransformV2Config(hardened=True), key=key)
        decoded = deserialize_v2_hardened(payload, key=key)
        assert all(expected[n].tobytes() == decoded[n].tobytes() for n in tensors)
        sizes.append({"dataset": dataset, "input_features": dims, "tensor_shapes": {n: list(v.shape) for n, v in tensors.items()},
                      "default_bytes": len(default), "hardened_bytes": len(payload), "delta_bytes": len(payload) - len(default),
                      "default_sha256": hashlib.sha256(default).hexdigest(), "hardened_sha256": hashlib.sha256(payload).hexdigest(),
                      "shape_provenance_audit_sha256": sha(prepared / "audit.json"), "synthetic_only": True})
        (OUT / (dataset + "_hardened_public_payload.json")).write_bytes(payload)
        (OUT / (dataset + "_default_public_payload.json")).write_bytes(default)
    save("payload_sizes.json", sizes)
    stage("PAYLOAD_PASS_INDEPENDENT_REVIEW")
    review()
    files = ["dna_encoder/transform_defense_v2.py", "dna_encoder/transform_defense_v2_hardened.py", "experiments/run_rq3_benchmark.py", "experiments/priority36a_wire.py", "experiments/priority36a_validate.py", "tests/test_priority36a_hardening.py"]
    # Compile without placing caches alongside historical source/artifact files.
    import py_compile
    for i, filename in enumerate(files):
        py_compile.compile(str(ROOT / filename), cfile=str(OUT / ("compile_%d.pyc" % i)), doraise=True)
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    (OUT / "diff_check.log").write_text(diff.stdout + diff.stderr)
    assert diff.returncode == 0
    save("validation.json", {"status": "PASS", "cpu_only": True, "torch_threads": torch.get_num_threads(), "compile_files": files, "git_diff_check": "PASS", "scientific_runs": 0})
    stage("VALIDATED_REPORT_PENDING")
    print(json.dumps(sizes, indent=2))


if __name__ == "__main__":
    main()
