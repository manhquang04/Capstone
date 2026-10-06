"""P34C integration primitives; no jobs or target allocation on import."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from collections import OrderedDict
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
import torch
from experiments import priority33b_core as native
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from dna_encoder.transform_defense_v2 import DNATransformV2Config, transform_update_array_v2, reconstruct_update_array_v2
from experiments.priority34c_solvers import known_key_least_squares, ratio_record

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/priority34c"
PREPARED = ROOT / "artifacts/priority32_multidataset/prepared"
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    if torch.get_num_interop_threads() != 1:
        raise
V1 = DNATransformConfig(block_size=256, mix_ratio=.08, keep_ratio=.88,
                        shrink_factor=.45, seed=681958327)
V2 = DNATransformV2Config(compression_ratio=.95, quantization_eta=.01, seed=20260916)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, doc):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError("P34C-only write contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def full_adapter(dataset):
    folder = PREPARED / dataset
    audit = json.loads((folder / "audit.json").read_text())
    for name, expected in audit["prepared_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError("prepared input changed: " + name)
    for name, expected in audit["raw_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("raw input changed: " + name)
    meta = json.loads((folder / "preprocessing.json").read_text())
    blocks = native.logical_blocks(meta)
    values = np.asarray(np.load(folder / "train_x.npy", mmap_mode="r"), dtype=np.float64)
    native.finite(values, "training adapter population")
    std = values.std(0, ddof=1)
    std[std < 1e-12] = 1.
    return dict(dataset=dataset, selected_blocks=blocks, columns=list(range(values.shape[1])),
                mean=values.mean(0).tolist(), std=std.tolist(), lower=values.min(0).tolist(),
                upper=values.max(0).tolist(), encoded_features=meta["feature_names"],
                original_encoded_dimension=values.shape[1], selected_encoded_dimension=values.shape[1],
                logical_feature_count=len(blocks), prepared_sha256=audit["prepared_sha256"],
                raw_sha256=audit["raw_sha256"])


def prepare_adapters(dataset):
    full = full_adapter(dataset)
    if dataset == "paysim":
        tabular = full
        provenance = None
    else:
        earlier = ROOT / "artifacts/priority33b/attempt2" / dataset / "adapter.json"
        tabular = json.loads(earlier.read_text())
        if tabular["prepared_sha256"] != full["prepared_sha256"]:
            raise ValueError("P33B/P32 adapter input mismatch")
        provenance = dict(path=str(earlier.relative_to(ROOT)), sha256=sha(earlier))
    for cell, doc in (("ratio_batch1", full), ("tableak_batch1", tabular), ("tableak_batch2", tabular)):
        path = OUT / dataset / cell / "adapter.json"
        if path.exists() and json.loads(path.read_text()) != doc:
            raise ValueError("adapter drift")
        if not path.exists():
            write(path, doc)
    return dict(native_adapter_provenance=provenance, full_dimension=full["selected_encoded_dimension"],
                native_dimension=tabular["selected_encoded_dimension"])


class PreparedDataset(native.BaseDataset):
    def __init__(self, dataset, cell):
        super().__init__(name="P34C_" + dataset + "_" + cell, device="cpu", random_state=333042)
        doc = json.loads((OUT / dataset / cell / "adapter.json").read_text())
        self.train_features = OrderedDict((b["name"], b["categories"]) for b in doc["selected_blocks"])
        self.label = "fraud"
        self.features = OrderedDict(self.train_features)
        self.features[self.label] = ["0", "1"]
        self.mean = torch.tensor(doc["mean"], dtype=torch.float32)
        self.std = torch.tensor(doc["std"], dtype=torch.float32)
        self.num_features = len(doc["columns"])
        self.columns = doc["columns"]
        folder = PREPARED / dataset
        self.Xtrain = torch.from_numpy(np.array(np.load(folder / "train_x.npy", mmap_mode="r")[:, self.columns], copy=True))
        self.ytrain = torch.from_numpy(np.load(folder / "train_y.npy")).long()
        self.Xtest, self.ytest = self.Xtrain[:0], self.ytrain[:0]
        self._create_index_maps()
        self.continuous_bounds, self.standardized_continuous_bounds = {}, {}
        for name, (kind, ids) in self.train_feature_index_map.items():
            if kind == "cont":
                i = ids[0]
                lo, hi = doc["lower"][i], doc["upper"][i]
                self.continuous_bounds[name] = (lo, hi)
                self.standardized_continuous_bounds[name] = ((lo-doc["mean"][i])/doc["std"][i], (hi-doc["mean"][i])/doc["std"][i])
        self.histograms_and_continuous_bounds_calculated = True
        self.standardize()

    decode_batch = native.PreparedDataset.decode_batch


def strict_model(net):
    for name, value in net.state_dict().items():
        native.finite(value, name)
        if name.endswith("running_var") and (value < 0).any():
            raise FloatingPointError("negative BN variance: " + name)
        if value.device.type != "cpu":
            raise ValueError("CPU-only model")


def ratio_model(dim):
    spec = importlib.util.spec_from_file_location("p34c_public_fraud_mlp", ROOT / "models/fraud_mlp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(340042)
        net = module.FraudMLP(dim).cpu().eval()
    strict_model(net)
    return net


def gradient(net, x, labels, ratio=False):
    strict_model(net)
    logits = net(x)
    native.finite(logits, "gradient logits")
    if ratio:
        targets = labels.float().reshape_as(logits)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1-probabilities) * (1-targets)
        alpha = .95 * targets + .05 * (1-targets)
        loss = (alpha * (1-pt).pow(2) * bce).mean()
    else:
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
    native.finite(loss, "gradient loss")
    bn_names = {prefix + "." + name if prefix else name
                for prefix, layer in net.named_modules()
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
                for name, _ in layer.named_parameters(recurse=False)}
    pairs = [(name, p) for name, p in net.named_parameters() if name not in bn_names]
    values = torch.autograd.grad(loss, [p for _, p in pairs])
    payload = OrderedDict((name, g.detach().clone()) for (name, _), g in zip(pairs, values))
    for name, g in payload.items():
        native.finite(g, "upload:" + name)
    if set(payload) & bn_names:
        raise AssertionError("BN payload firewall")
    strict_model(net)
    return payload, dict(no_bn_payload=True, omitted_bn_parameters=sorted(bn_names),
                         payload_names=list(payload), loss=float(loss))


def defend(raw, arm, noise_seed=None, clip=None, noise_sd=None):
    names, values = list(raw), list(raw.values())
    if arm == "unprotected":
        return [g.clone() for g in values], None, None
    if arm == "dna_v1_conservative":
        return [torch.from_numpy(transform_update_array(g.numpy(), V1, tensor_index=i)[0].copy())
                for i, g in enumerate(values)], None, None
    if arm == "dna_v2_0p95":
        sketches, metadata = zip(*(transform_update_array_v2(g.numpy(), V2, tensor_index=i)
                                    for i, g in enumerate(values)))
        plans = [dict(padded_size=m.padded_size, sampled=torch.tensor(m.sampled_indices),
                      signs=torch.from_numpy(__import__("dna_encoder.transform_defense_v2", fromlist=["_signs"])._signs(m.padded_size, m.seed)).float(),
                      scale=float(np.sqrt(m.padded_size / m.sketch_size))) for m in metadata]
        return [torch.from_numpy(q.copy()) for q in sketches], plans, list(metadata)
    if arm.startswith("dp_"):
        if noise_seed is None or clip is None or noise_sd is None or clip <= 0 or noise_sd < 0:
            raise ValueError("missing valid private DP configuration")
        norm = float(torch.sqrt(sum(g.double().square().sum() for g in values)))
        factor = min(1., clip/norm) if norm > 0 else 1.
        generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        payload = [(g.double()*factor + torch.randn(g.shape, dtype=torch.float64, generator=generator)*noise_sd).float() for g in values]
        for g in payload:
            native.finite(g, "client-side DP upload")
        return payload, None, None
    raise ValueError("unknown defense " + arm)


def ratio_decode(names, payload, metadata=None):
    receipts = []
    if metadata is not None:
        decoded = []
        for q, meta in zip(payload, metadata):
            array, receipt = known_key_least_squares(q.numpy(), meta)
            decoded.append(torch.from_numpy(array))
            receipts.append(receipt)
        payload = decoded
    visible = dict(zip(names, payload))
    record, receipt = ratio_record(visible["network.0.weight"].numpy(), visible["network.0.bias"].numpy())
    return torch.from_numpy(record).float().unsqueeze(0), dict(ratio=receipt, least_squares=receipts)
