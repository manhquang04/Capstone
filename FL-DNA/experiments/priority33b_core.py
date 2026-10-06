"""P33b native TabLeak adapter and payload-only individual-record measurement."""
from __future__ import annotations
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import torch
from experiments.priority29.tabular_native_positive_control import official_config, mean_mode_baseline
from datasets.base_dataset import BaseDataset
from models import FullyConnected
from utils import match_reconstruction_ground_truth, batch_feature_wise_accuracy_score, post_process_continuous
from utils.encoder_decoder import to_categorical
import attacks.gradient_inversion_attack as gia
from experiments.priority30_native_defenses.run_audit import v2_sketch_payload, v2_sketch_loss, cosine_loss_lists, dna_v1_debias
from experiments.priority30_native_defenses.native_adapters import dna_v1_gradient

torch.set_num_threads(1)
OUT = ROOT / "artifacts/priority33b/attempt2"
PREPARED = ROOT / "artifacts/priority32_multidataset/prepared"
PROTOCOL = ROOT / "protocols/amendments/2026-10-02_priority33b_tabular_record_recovery.md"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to((ROOT / "artifacts/priority33b").resolve()):
        raise ValueError("P33b artifacts-only write contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temp.replace(path)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")


def finite(value, context):
    good = torch.isfinite(value).all().item() if torch.is_tensor(value) else np.isfinite(value).all()
    if not good:
        raise FloatingPointError(f"nonfinite: {context}")


def exact_p(w, l):
    return sum(math.comb(w+l, k) for k in range(w, w+l+1))/2**(w+l) if w+l else 1.


def logical_blocks(meta):
    names = meta["feature_names"]
    blocks = []
    for i, name in enumerate(names):
        parent = next((k for k, levels in meta["one_hot_levels"].items() if name == f"{k}={levels[0]}"), None)
        if parent is not None:
            levels = meta["one_hot_levels"][parent]
            ids = [names.index(f"{parent}={v}") for v in levels]
            if ids != list(range(i, i+len(levels))):
                raise ValueError("noncontiguous categorical block")
            blocks.append(dict(name=parent, indices=ids, categories=levels))
        elif not any(name == f"{k}={v}" for k, levels in meta["one_hot_levels"].items() for v in levels):
            blocks.append(dict(name=name, indices=[i], categories=None))
    assert sorted(i for b in blocks for i in b["indices"]) == list(range(len(names)))
    return blocks


def select_blocks(blocks, scores, limit):
    selected, used = [], 0
    for index in sorted(range(len(blocks)), key=lambda i: (-scores[i], i)):
        width = len(blocks[index]["indices"])
        if used+width <= limit:
            selected.append(index)
            used += width
    return sorted(selected)


def prepare(dataset):
    folder = PREPARED / dataset
    audit = json.loads((folder / "audit.json").read_text())
    for name, expected in audit["prepared_sha256"].items():
        assert sha(folder/name) == expected, f"P32 input changed: {name}"
    for name, expected in audit["raw_sha256"].items():
        assert sha(ROOT/name) == expected, f"raw input changed: {name}"
    meta = json.loads((folder / "preprocessing.json").read_text())
    blocks = logical_blocks(meta)
    x = np.load(folder / "train_x.npy", mmap_mode="r")
    y = np.load(folder / "train_y.npy")
    yy = y.astype(np.float64)-y.mean()
    scores = []
    for b in blocks:
        values = np.asarray(x[:, b["indices"]], dtype=np.float64)
        centered = values-values.mean(0)
        denominator = np.sqrt(np.square(centered).sum(0)*np.square(yy).sum())
        corr = np.divide((centered*yy[:, None]).sum(0), denominator, out=np.zeros_like(denominator), where=denominator>0)
        scores.append(float(np.max(np.abs(corr))))
    chosen = select_blocks(blocks, scores, 60) if dataset == "ieee_cis" else list(range(len(blocks)))
    columns = [i for j in chosen for i in blocks[j]["indices"]]
    values = np.asarray(x[:, columns], dtype=np.float64)
    mean, std = values.mean(0), values.std(0, ddof=1)
    std[std < 1e-12] = 1.
    result = dict(dataset=dataset, all_blocks=[dict(**b, training_score=s) for b,s in zip(blocks,scores)],
        selected_blocks=[blocks[j] for j in chosen], columns=columns,
        encoded_features=[meta["feature_names"][i] for i in columns],
        mean=mean.tolist(), std=std.tolist(), lower=values.min(0).tolist(), upper=values.max(0).tolist(),
        prepared_sha256=audit["prepared_sha256"], raw_sha256=audit["raw_sha256"],
        original_encoded_dimension=x.shape[1], selected_encoded_dimension=len(columns), logical_feature_count=len(chosen))
    path = OUT / dataset / "adapter.json"
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        write(path, result)
    return result


class PreparedDataset(BaseDataset):
    def __init__(self, dataset):
        super().__init__(name=f"P33B_{dataset}", device="cpu", random_state=333042)
        doc = json.loads((OUT / dataset / "adapter.json").read_text())
        folder = PREPARED/dataset
        self.train_features = OrderedDict((b["name"], b["categories"]) for b in doc["selected_blocks"])
        self.label = "fraud"
        self.features = OrderedDict(self.train_features)
        self.features[self.label] = ["0", "1"]
        self.mean = torch.tensor(doc["mean"], dtype=torch.float32)
        self.std = torch.tensor(doc["std"], dtype=torch.float32)
        self.num_features = len(doc["columns"])
        self.Xtrain = torch.from_numpy(np.array(np.load(folder/"train_x.npy", mmap_mode="r")[:, doc["columns"]], copy=True))
        self.ytrain = torch.from_numpy(np.load(folder/"train_y.npy")).long()
        # No heldout records/bounds enter the attack. Base interface only requires Xtest for helper APIs.
        self.Xtest, self.ytest = self.Xtrain[:0], self.ytrain[:0]
        self._create_index_maps()
        self.continuous_bounds, self.standardized_continuous_bounds = {}, {}
        for name, (kind, ids) in self.train_feature_index_map.items():
            if kind == "cont":
                i = ids[0]
                lo, hi = doc["lower"][i], doc["upper"][i]
                self.continuous_bounds[name] = (lo,hi)
                self.standardized_continuous_bounds[name] = ((lo-doc["mean"][i])/doc["std"][i], (hi-doc["mean"][i])/doc["std"][i])
        self.histograms_and_continuous_bounds_calculated = True
        self.standardize()

    def decode_batch(self, batch, standardized=True):
        raw = self.de_standardize(batch) if standardized else batch
        return to_categorical(raw.detach().cpu().numpy(), self.train_features, single_bit_binary=False, nearest_int=False)


def measure(dataset, truth, recovered):
    finite(recovered, "reconstruction before official bounds")
    rec = post_process_continuous(recovered.detach().clone(), dataset)
    a,b = dataset.decode_batch(truth),dataset.decode_batch(rec)
    aligned, errors, cats, conts = match_reconstruction_ground_truth(a,b,dataset.create_tolerance_map())
    metric = dict(accuracy_percent=100*(1-float(np.mean(errors))),
        per_feature_accuracy_percent={k:100*(1-float(v)) for k,v in batch_feature_wise_accuracy_score(a,aligned,dataset.create_tolerance_map(),dataset.train_features).items()})
    finite(metric["accuracy_percent"], "scored accuracy")
    return metric


def model():
    raise RuntimeError("call make_model with adapter dimension")


def make_model(dim):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(333042)
        net = FullyConnected(dim,[100,100,2])
    def check(module, args, output):
        assert output.device.type == "cpu"
        finite(output,"actual model logits")
    net.register_forward_hook(check)
    return net


def native_recover(net, dataset, payload, labels, seed, mode="plain", plans=None):
    if mode not in ("plain", "v2_sketch"):
        raise ValueError(f"unknown adaptive mode {mode}")
    for g in payload:
        finite(g,"server payload")
    original = gia._cosine_similarity_loss
    def observable_loss(reconstruct_gradient, true_grad, device, weights=None, alpha=None):
        candidate, observed = reconstruct_gradient, true_grad
        assert observed is payload, "native attack received non-transmitted gradient object"
        for g in candidate:
            finite(g,"candidate gradient")
        value = v2_sketch_loss(candidate,payload,plans) if mode=="v2_sketch" else original(candidate,payload,device,weights,alpha)
        finite(value,"adaptive loss")
        return value
    torch.manual_seed(seed)
    np.random.seed(seed)
    gia._cosine_similarity_loss = observable_loss
    config = official_config()
    try:
        result = gia.invert_grad(net=net,training_criterion=torch.nn.CrossEntropyLoss(), true_grad=payload,
            true_label=labels,true_data=torch.empty(len(labels),dataset.num_features),dataset=dataset,**config)
    finally:
        gia._cosine_similarity_loss = original
    rec, ensemble, losses = result
    finite(rec,"pooled result")
    for rec_i in ensemble:
        finite(rec_i,"ensemble member")
    finite(np.asarray(losses),"ensemble objectives")
    assert len(ensemble)==len(losses)==30
    return rec,ensemble,losses


def controls(dataset, truth, seed):
    mm=mean_mode_baseline(dataset,truth)
    rng=np.random.default_rng(seed)
    scores,guesses=[],[]
    for _ in range(30):
        guess=torch.empty_like(truth)
        for _, (_,ids) in dataset.train_feature_index_map.items():
            rows=rng.integers(0,len(dataset.Xtrain),len(truth))
            guess[:,ids]=dataset.Xtrain[rows][:,ids]
        scores.append(measure(dataset,truth,guess)["accuracy_percent"])
        guesses.append(guess)
    return dict(mean_mode=measure(dataset,truth,mm)["accuracy_percent"], empirical_mean=float(np.mean(scores)),
        empirical_single=scores[0],empirical_sd=float(np.std(scores,ddof=1))),mm,guesses[0]
