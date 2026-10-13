"""Replay unchanged P34A baseline with non-RNG-consuming snapshots."""
import copy
import hashlib
import random
import time
from collections import OrderedDict
import numpy as np
import torch
from experiments import priority34a_local_bn as old
from experiments.priority35a_runtime import ROOT, Q, load, write, sha, tensor_save, stage, check_hashes

PREPARED = ROOT / "artifacts/priority32_multidataset/prepared"


def arrays_partitions(dataset):
    folder = PREPARED / dataset
    x = np.load(folder / "train_x.npy", mmap_mode="r")
    y = np.load(folder / "train_y.npy", mmap_mode="r")
    categories = np.load(folder / "train_categories.npy")
    parts = old.p._mild_non_iid_client_indices(old.p.pd.DataFrame({"type": categories}), y, 3, 321000)
    historic = load(old.OUT / "jobs" / dataset / "baseline/321000.json")
    if [hashlib.sha256(indices.tobytes()).hexdigest() for indices in parts] != historic["partition_sha256"]:
        raise ValueError("P34A partition mismatch")
    if np.unique(np.concatenate(parts)).size != len(y) or sum(map(len, parts)) != len(y):
        raise ValueError("partition coverage failed")
    if any(len(indices) % 1024 == 1 for indices in parts):
        raise ValueError("singleton P34A batch")
    return x, y, parts


def state(global_parameters, clients, bn):
    return dict(global_non_bn=OrderedDict((k, v.clone()) for k, v in global_parameters.items()),
                client_bn=[OrderedDict((name, local.state_dict()[name].clone()) for name in sorted(bn)) for local in clients])


def exact_states(a, b):
    if set(a) != set(b) or tuple(a["global_non_bn"]) != tuple(b["global_non_bn"]) or len(a["client_bn"]) != 3 or len(b["client_bn"]) != 3:
        raise ValueError("checkpoint schema mismatch")
    pairs = [(a["global_non_bn"], b["global_non_bn"]), *zip(a["client_bn"], b["client_bn"])]
    for left, right in pairs:
        if tuple(left) != tuple(right):
            raise ValueError("checkpoint keys mismatch")
        for name in left:
            if left[name].dtype != right[name].dtype or not torch.equal(left[name], right[name]):
                raise ValueError("P34A bit-exact replay mismatch: " + name)
    return True


def replay(dataset):
    result_path = Q / "checkpoints" / dataset / "verified.json"
    if result_path.exists():
        result = load(result_path)
        if result["status"] != "VERIFIED_BIT_EXACT" or result["torch_threads"] != 1:
            raise ValueError("invalid checkpoint receipt")
        check_hashes(result["hashes"])
        return result
    historic = load(old.OUT / "jobs" / dataset / "baseline/321000.json")
    check_hashes(historic["config"]["source_sha256"])
    old.valid_result(old.OUT / "jobs" / dataset / "baseline/321000.json", historic["config"])
    config = historic["config"]
    if config["training"] != dict(alpha=.95, lr=.001, rounds=50) or config["method"] != "baseline":
        raise ValueError("P34A baseline config drift")
    random.seed(321000)
    np.random.seed(321000)
    torch.manual_seed(321000)
    torch.set_num_threads(1)
    x, y, partitions = arrays_partitions(dataset)
    loaders = [old.p.DataLoader(old.p.TensorDataset(torch.from_numpy(np.array(x[indices], copy=True)),
               torch.from_numpy(np.array(y[indices], copy=True)[:, None])), batch_size=1024,
               shuffle=True, num_workers=0, drop_last=False,
               generator=torch.Generator().manual_seed(321000 + client)) for client, indices in enumerate(partitions)]
    initial = old.p.FraudMLP(x.shape[1]).cpu()
    bn, allowed = old.domains(initial)
    old.guard(initial, "initial")
    clients = [copy.deepcopy(initial) for _ in range(3)]
    global_parameters = OrderedDict((name, initial.state_dict()[name].clone()) for name in allowed)
    loss_function = old.p.BinaryFocalLoss(alpha=.95, gamma=2.)
    attempt = Q / "checkpoints" / dataset / ("attempt_" + str(time.time_ns()))
    attempt.mkdir(parents=True, exist_ok=False)
    hashes, snapshots = {}, {}
    for round_number in range(1, 51):
        states = []
        for client, (local, loader) in enumerate(zip(clients, loaders)):
            old.broadcast(local, global_parameters, bn, allowed)
            local.train()
            old.guard(local, "broadcast")
            optimizer = torch.optim.Adam(local.parameters(), lr=.001)
            for features, labels in loader:
                optimizer.zero_grad()
                logits = local(features)
                loss = loss_function(logits, labels)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise ValueError("nonfinite checkpoint forward")
                loss.backward()
                if any(param.grad is not None and not torch.isfinite(param.grad).all() for param in local.parameters()):
                    raise ValueError("nonfinite checkpoint gradient")
                optimizer.step()
                old.guard(local, "trained")
            states.append(old.upload(local, global_parameters, "baseline", 321000, round_number, client, bn, allowed))
        global_parameters = old.p.fed_avg(states, list(map(len, partitions)))
        old.assert_payload(global_parameters, bn, allowed)
        if round_number in (10, 30, 50):
            path = attempt / ("round" + str(round_number) + ".pt")
            tensor_save(path, state(global_parameters, clients, bn), private=True)
            snapshots[str(round_number)] = str(path.relative_to(ROOT))
            hashes[str(path.relative_to(ROOT))] = sha(path)
        if round_number % 10 == 0:
            stage("checkpoint_replay", dataset=dataset, checkpoint_replay_round=round_number)
    historic_checkpoint = next(old.OUT / name for name in historic["artifact_sha256"] if name.endswith("final_checkpoint.pt"))
    exact_states(torch.load(ROOT / snapshots["50"], map_location="cpu", weights_only=False),
                 torch.load(historic_checkpoint, map_location="cpu", weights_only=False))
    result = dict(status="VERIFIED_BIT_EXACT", dataset=dataset, seed=321000, torch_threads=1,
                  snapshots=snapshots, hashes=hashes, historical_final_sha256=sha(historic_checkpoint),
                  no_bn_uploaded=True, all_50_rounds_replayed=True)
    write(result_path, result, exclusive=True)
    return result
