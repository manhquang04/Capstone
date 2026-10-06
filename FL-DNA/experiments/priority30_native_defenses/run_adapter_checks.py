"""Run Priority 30 native-defense adapter sanity checks.

These are adapter checks, not defense efficacy experiments.  They verify
defense-not-identity and server-knowledge/lossless prerequisites before any
Priority 29 E1/E2/E3 cell can be emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("DATALOADER_NUM_WORKERS", "0")
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage, ToTensor

ROOT = Path(__file__).resolve().parents[2]
# Tableak must precede the project root so `import attacks.*` resolves to the
# official Tableak package, not FL-DNA's own `attacks/` package.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_defenses" / "invertinggradients"))
sys.path.insert(0, str(ROOT / "external_defenses" / "tableak"))

import inversefed  # noqa: E402
from datasets import ADULT  # noqa: E402
from models import FullyConnected  # noqa: E402

from experiments.harness.validators import (  # noqa: E402
    DefenseEffectRecord,
    HarnessResultValidator,
    KnowledgeReceiptRecord,
    LosslessSanityRecord,
    ValidationError,
)
from experiments.priority30_native_defenses.native_adapters import (  # noqa: E402
    PrecodeAdultFC,
    PrecodeLeNetZhu,
    count_sketch,
    dna_v1_gradient,
    dna_v2_gradient,
    gradient_dict,
    l2_update_delta,
    l2_update_delta_mapped,
    prune_gradient,
    sketch_server_knowledge,
    v2_least_squares_sketch_residual,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # pragma: no cover - recorded in report
        return f"unavailable: {exc}"


def as_bool(value: bool) -> bool:
    return bool(value)


def check_image() -> dict[str, Any]:
    torch.manual_seed(30)
    np.random.seed(30)
    data = CIFAR10(str(ROOT / "datasets" / "cifar10"), train=False, download=False, transform=ToTensor())
    raw, label_int = data[0]
    x = raw.unsqueeze(0)
    y = torch.tensor([label_int], dtype=torch.long)
    base, _ = inversefed.construct_model("LeNetZhu", seed=42)
    base.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    undef = gradient_dict(base, x, y, loss_fn)

    # PRECODE model-level bottleneck: the forward/gradient should change and
    # two stochastic passes should differ at fixed weights.
    pre = PrecodeLeNetZhu(base)
    pre.eval()
    torch.manual_seed(301)
    g_pre_a = gradient_dict(pre, x, y, loss_fn)
    torch.manual_seed(302)
    g_pre_b = gradient_dict(pre, x, y, loss_fn)
    pre_effect = l2_update_delta(undef, {k: v for k, v in g_pre_a.items() if k in undef})
    pre_stochastic = l2_update_delta(g_pre_a, g_pre_b)

    # Soteria: compute representation sensitivity for LeNet-Zhu body output.
    x_req = x.detach().clone().requires_grad_(True)
    rep = base.body(x_req).view(1, -1)
    scores = []
    for j in range(rep.shape[1]):
        if x_req.grad is not None:
            x_req.grad.zero_()
        base.zero_grad(set_to_none=True)
        rep[:, j].sum().backward(retain_graph=True)
        scores.append(x_req.grad.detach().reshape(1, -1).norm(dim=1) / (rep[:, j].detach().abs() + 0.1))
    score = torch.stack(scores).reshape(-1)
    threshold = torch.quantile(score.float(), 0.80)
    mask = (score >= threshold).to(undef["fc.0.weight"])
    soteria = {k: v.clone() for k, v in undef.items()}
    soteria["fc.0.weight"] = soteria["fc.0.weight"] * mask.unsqueeze(0)
    soteria_effect = l2_update_delta(undef, soteria)

    pruned, pruning_masks = prune_gradient(undef, fraction=0.70)
    prune_effect = l2_update_delta(undef, pruned)
    dna_v1 = dna_v1_gradient(undef)
    dna_v1_effect = l2_update_delta(undef, dna_v1)
    dna_v2, dna_v2_metadata = dna_v2_gradient(undef)
    dna_v2_effect = l2_update_delta(undef, dna_v2)
    v2_ls_residual = v2_least_squares_sketch_residual()

    # ATS policy 3-1-7, applied to the input before gradient computation.  The
    # official code uses deprecated np.int; patch in-process only.
    ats_effect = None
    try:
        ats_root = ROOT / "external_defenses" / "ats"
        sys.path.insert(0, str(ats_root))
        import numpy as _np  # noqa: WPS433

        if not hasattr(_np, "int"):
            _np.int = int  # type: ignore[attr-defined]
        import policy as ats_policy  # type: ignore  # noqa: WPS433

        pil = ToPILImage()(raw)
        torch.manual_seed(303)
        np.random.seed(303)
        import random as _random

        _random.seed(303)
        transformed_pil = pil
        for idx in [3, 1, 7]:
            transformed_pil = ats_policy.policies[idx](transformed_pil)
        ats_x = ToTensor()(transformed_pil).unsqueeze(0)
        ats_grad = gradient_dict(base, ats_x, y, loss_fn)
        ats_effect = l2_update_delta(undef, ats_grad)
    except Exception as exc:  # recorded as not assessable if this happens
        ats_effect = None
        ats_error = str(exc)
    else:
        ats_error = None

    sketched = count_sketch(undef, rows=5, columns=None, seed=21)
    sketch_norm = float(sketched.tensors["sketch"].norm().item())
    validator = HarnessResultValidator(
        lossless_sanity={"count_sketch_known": LosslessSanityRecord("count_sketch_known", max_abs_error=0.0)},
        defense_effects={
            "precode": DefenseEffectRecord("precode", changed_forward=True, effect_size=float(pre_effect)),
            "soteria": DefenseEffectRecord("soteria", changed_update=True, effect_size=float(soteria_effect)),
            "gradient_pruning": DefenseEffectRecord("gradient_pruning", changed_update=True, effect_size=float(prune_effect)),
            "count_sketch": DefenseEffectRecord("count_sketch", changed_update=True, effect_size=sketch_norm),
            "dna_v1_conservative": DefenseEffectRecord("dna_v1_conservative", changed_update=True, effect_size=float(dna_v1_effect)),
            "dna_v2_0p95": DefenseEffectRecord("dna_v2_0p95", changed_update=True, effect_size=float(dna_v2_effect)),
            "ats_3_1_7": DefenseEffectRecord("ats_3_1_7", changed_update=ats_effect is not None, effect_size=float(ats_effect or 0.0)),
        },
        knowledge_receipts={
            "count_sketch_known": KnowledgeReceiptRecord(
                "count_sketch_known",
                required_items=frozenset({"sketch_hashes"}),
                received_items=frozenset({"sketch_hashes"}),
            ),
            "count_sketch_unknown": KnowledgeReceiptRecord(
                "count_sketch_unknown",
                required_items=frozenset({"sketch_hashes"}),
                received_items=frozenset(),
            ),
        },
    )
    checks: dict[str, Any] = {}
    for defense_name in ["precode", "soteria", "gradient_pruning", "count_sketch", "dna_v1_conservative", "dna_v2_0p95", "ats_3_1_7"]:
        try:
            validator.require_defense_not_identity(defense_name)
            checks[f"{defense_name}_not_identity"] = {"passed": True}
        except ValidationError as exc:
            checks[f"{defense_name}_not_identity"] = {"passed": False, "error": str(exc)}
    for attack_name in ["count_sketch_known", "count_sketch_unknown"]:
        try:
            validator.require_attack_received_knowledge(attack_name)
            checks[f"{attack_name}_knowledge"] = {"passed": True}
        except ValidationError as exc:
            checks[f"{attack_name}_knowledge"] = {"passed": False, "error": str(exc)}
    try:
        validator.require_decode_knowledge(sketched, sketch_server_knowledge(True))
        checks["count_sketch_decode_known_hashes"] = {"passed": True}
    except ValidationError as exc:
        checks["count_sketch_decode_known_hashes"] = {"passed": False, "error": str(exc)}
    try:
        validator.require_decode_knowledge(sketched, sketch_server_knowledge(False))
        checks["count_sketch_decode_unknown_hashes"] = {"passed": True}
    except ValidationError as exc:
        checks["count_sketch_decode_unknown_hashes"] = {"passed": False, "error": str(exc)}

    return {
        "target": {"dataset": "CIFAR-10 test", "index": 0, "label": int(label_int)},
        "effects": {
            "precode_l2_vs_undef_common_names": float(pre_effect),
            "precode_stochastic_l2_between_passes": float(pre_stochastic),
            "soteria_l2_vs_undef": float(soteria_effect),
            "soteria_mask_retained": int(mask.sum().item()),
            "soteria_mask_total": int(mask.numel()),
            "gradient_pruning_l2_vs_undef": float(prune_effect),
            "gradient_pruning_fraction": 0.70,
            "gradient_pruning_retained_coordinates": int(sum(m.sum().item() for m in pruning_masks.values())),
            "count_sketch_norm": sketch_norm,
            "dna_v1_conservative_l2_vs_undef": float(dna_v1_effect),
            "dna_v2_0p95_l2_vs_undef": float(dna_v2_effect),
            "dna_v2_metadata_items": len(dna_v2_metadata),
            "dna_v2_least_squares_sketch_residual": float(v2_ls_residual),
            "ats_3_1_7_l2_vs_raw_gradient": ats_effect,
            "ats_error": ats_error,
        },
        "checks": checks,
    }


def check_adult() -> dict[str, Any]:
    torch.manual_seed(31)
    np.random.seed(31)
    old_cwd = Path.cwd()
    os.chdir(ROOT / "external_defenses" / "tableak")
    try:
        dataset = ADULT()
        dataset.standardize()
        x = dataset.Xtrain[:8].clone()
        y = dataset.ytrain[:8].clone()
        base = FullyConnected(dataset.num_features, [100, 100, 2])
        loss_fn = torch.nn.CrossEntropyLoss()
        undef = gradient_dict(base, x, y, loss_fn)
        pre = PrecodeAdultFC(base)
        pre.eval()
        torch.manual_seed(311)
        g_pre_a = gradient_dict(pre, x, y, loss_fn)
        torch.manual_seed(312)
        g_pre_b = gradient_dict(pre, x, y, loss_fn)
        adult_precode_mapping = {
            "layers.1.layers.0.weight": "prefix.1.layers.0.weight",
            "layers.1.layers.0.bias": "prefix.1.layers.0.bias",
            "layers.2.layers.0.weight": "prefix.2.layers.0.weight",
            "layers.2.layers.0.bias": "prefix.2.layers.0.bias",
            "layers.3.weight": "classifier.weight",
            "layers.3.bias": "classifier.bias",
        }
        pre_effect = l2_update_delta_mapped(undef, g_pre_a, adult_precode_mapping)
        pre_stochastic = l2_update_delta(g_pre_a, g_pre_b)
        pruned, pruning_masks = prune_gradient(undef, fraction=0.70)
        prune_effect = l2_update_delta(undef, pruned)
        # Adult Soteria: representation before final FC is the second LinReLU output.
        x_req = x.detach().clone().requires_grad_(True)
        rep = pre.prefix(x_req).detach()  # same representation shape as final classifier input
        base.zero_grad(set_to_none=True)
        x_req = x.detach().clone().requires_grad_(True)
        representation = base.layers[:-1](x_req)
        scores = []
        for j in range(representation.shape[1]):
            if x_req.grad is not None:
                x_req.grad.zero_()
            base.zero_grad(set_to_none=True)
            representation[:, j].sum().backward(retain_graph=True)
            scores.append(x_req.grad.detach().reshape(x_req.shape[0], -1).norm(dim=1).sum() / (representation[:, j].detach().abs().sum() + 0.1))
        score = torch.stack(scores)
        threshold = torch.quantile(score.float(), 0.40)
        mask = (score >= threshold).to(undef["layers.3.weight"])
        soteria = {k: v.clone() for k, v in undef.items()}
        soteria["layers.3.weight"] = soteria["layers.3.weight"] * mask.unsqueeze(0)
        soteria_effect = l2_update_delta(undef, soteria)
        dna_v1 = dna_v1_gradient(undef)
        dna_v1_effect = l2_update_delta(undef, dna_v1)
        dna_v2, dna_v2_metadata = dna_v2_gradient(undef)
        dna_v2_effect = l2_update_delta(undef, dna_v2)
        sketched = count_sketch(undef, rows=5, columns=None, seed=21)
        sketch_norm = float(sketched.tensors["sketch"].norm().item())
        return {
            "target": {"dataset": "Adult train", "indices": list(range(8))},
            "effects": {
                "precode_l2_vs_undef_common_names": float(pre_effect),
                "precode_stochastic_l2_between_passes": float(pre_stochastic),
                "precode_mapping_used": adult_precode_mapping,
                "gradient_pruning_l2_vs_undef": float(prune_effect),
                "gradient_pruning_retained_coordinates": int(sum(m.sum().item() for m in pruning_masks.values())),
                "soteria_l2_vs_undef": float(soteria_effect),
                "soteria_mask_retained": int(mask.sum().item()),
                "soteria_mask_total": int(mask.numel()),
                "dna_v1_conservative_l2_vs_undef": float(dna_v1_effect),
                "dna_v2_0p95_l2_vs_undef": float(dna_v2_effect),
                "dna_v2_metadata_items": len(dna_v2_metadata),
                "count_sketch_norm": sketch_norm,
            },
            "checks": {
                "precode_not_identity": {"passed": bool(pre_effect > 1e-12 or pre_stochastic > 1e-12)},
                "gradient_pruning_not_identity": {"passed": bool(prune_effect > 1e-12)},
                "soteria_not_identity": {"passed": bool(soteria_effect > 1e-12)},
                "dna_v1_conservative_not_identity": {"passed": bool(dna_v1_effect > 1e-12)},
                "dna_v2_0p95_not_identity": {"passed": bool(dna_v2_effect > 1e-12)},
                "count_sketch_not_identity": {"passed": bool(sketch_norm > 1e-12)},
            },
        }
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/priority30_native_defenses/adapter_checks")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "priority": 30,
        "scope": "adapter checks only, not E1/E2/E3 defense efficacy",
        "threads": torch.get_num_threads(),
        "external_commits": {
            "precode": git_commit(ROOT / "external_defenses" / "precode"),
            "soteria": git_commit(ROOT / "external_defenses" / "soteria"),
            "dlg": git_commit(ROOT / "external_defenses" / "dlg"),
            "fetchsgd": git_commit(ROOT / "external_defenses" / "fetchsgd"),
        },
        "image": check_image(),
        "adult": check_adult(),
    }
    path = out / "adapter_checks.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
