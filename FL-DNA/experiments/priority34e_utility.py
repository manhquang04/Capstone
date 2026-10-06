"""Priority34E utility: P34A algorithm generalized only in K and replicate list."""
import copy
import hashlib
import json
import random
import time
from collections import OrderedDict
from pathlib import Path
from experiments import priority34a_local_bn as a
from experiments.priority34e import OUT, write
p, np, torch = a.p, a.np, a.torch
domains, guard, broadcast, upload = a.domains, a.guard, a.broadcast, a.upload
assert_payload, checked_probabilities = a.assert_payload, a.checked_probabilities

def valid_result(path, config):
    path = Path(path)
    if not path.exists():
        return False
    doc = json.loads(path.read_text())
    if doc.get("status") != "COMPLETED" or doc.get("config_sha256") != p.canonical_sha(config):
        raise ValueError("existing output/config mismatch; do not overwrite " + str(path))
    if not doc["no_bn_transmitted"] or doc["payload_assertions"] != config["training"]["rounds"]*config["K"]:
        raise ValueError("payload assertion count failed")
    if len(doc["clients"]) != config["K"] or doc["torch_threads"] != 1 or doc["torch_interop_threads"] != 1 or doc["device"] != 'cpu':
        raise ValueError("client/thread contract failed")
    for split in ("validation", "test"):
        for endpoint in ("f1", "auc_roc", "pr_auc"):
            values = [client[split][endpoint] for client in doc["clients"]]
            if not np.isfinite(values).all() or not np.isclose(np.mean(values), doc[split][endpoint], atol=1e-14, rtol=0):
                raise ValueError("nonfinite or incorrect mean-client metrics")
    for relative, digest in doc["artifact_sha256"].items():
        if p.sha(OUT / relative) != digest:
            raise ValueError("changed output " + relative)
    return True


def train_job(config, path_string):
    path = Path(path_string)
    if valid_result(path, config):
        return {"path": str(path), "skipped": True}
    started = time.time()
    seed, dataset, method = config["seed"], config["dataset"], config["method"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    prepared = p.OUT / "prepared" / dataset
    x = np.load(prepared / "train_x.npy", mmap_mode="r")
    y = np.load(prepared / "train_y.npy", mmap_mode="r")
    categories = np.load(prepared / "train_categories.npy")
    partitions = p._mild_non_iid_client_indices(p.pd.DataFrame({"type": categories}), y, config["K"], seed)
    if np.unique(np.concatenate(partitions)).size != len(y) or sum(map(len, partitions)) != len(y):
        raise ValueError("partition coverage failed")
    if any(len(indices) == 0 or len(indices)%1024 == 1 for indices in partitions):
        raise ValueError("singleton batch; requires direction")
    loaders = [p.DataLoader(p.TensorDataset(torch.from_numpy(np.array(x[indices], copy=True)),
              torch.from_numpy(np.array(y[indices], copy=True)[:, None])), batch_size=1024,
              shuffle=True, num_workers=0, drop_last=False,
              generator=torch.Generator().manual_seed(seed+client)) for client, indices in enumerate(partitions)]
    initial = p.FraudMLP(x.shape[1]).cpu()
    bn, allowed = domains(initial)
    guard(initial, "initialization")
    clients = [copy.deepcopy(initial) for _ in range(config["K"])]
    global_parameters = OrderedDict((name, initial.state_dict()[name].clone()) for name in allowed)
    loss_function = p.BinaryFocalLoss(alpha=config["training"]["alpha"], gamma=2.)
    # New attempt journal retains any interrupted partial rather than truncating it.
    attempt = path.parent / (path.stem + ".attempt_" + str(time.time_ns()))
    attempt.mkdir(parents=True, exist_ok=False)
    journal = attempt / "rounds.jsonl"
    for round_number in range(1, config["training"]["rounds"]+1):
        states, bn_minima, trajectory_minima = [], [], []
        for client, (local, loader) in enumerate(zip(clients, loaders)):
            broadcast(local, global_parameters, bn, allowed)
            local.train()
            minimum = guard(local, f"round{round_number}/client{client}/broadcast")
            optimizer = torch.optim.Adam(local.parameters(), lr=config["training"]["lr"])
            for batch, (features, labels) in enumerate(loader):
                optimizer.zero_grad()
                logits = local(features)
                if not torch.isfinite(logits).all():
                    raise ValueError(f"nonfinite logits round{round_number}/client{client}/batch{batch}")
                loss = loss_function(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in local.parameters()):
                    raise ValueError("nonfinite gradient")
                optimizer.step()
                current = guard(local, f"round{round_number}/client{client}/batch{batch}")
                minimum = {name: min(minimum[name], value) for name, value in current.items()}
            bn_minima.append(guard(local, f"round{round_number}/client{client}/trained"))
            trajectory_minima.append(minimum)
            states.append(upload(local, global_parameters, method, seed, round_number, client, bn, allowed))
        global_parameters = p.fed_avg(states, list(map(len, partitions)))
        assert_payload(global_parameters, bn, allowed)
        with journal.open("a") as handle:
            handle.write(json.dumps({"at": p.now(), "round": round_number,
                "bn_minima_per_client": bn_minima, "bn_minima_all_steps": trajectory_minima,
                "transmitted_keys": list(allowed), "bn_keys": sorted(bn),
                "no_bn_transmitted": True, "upload_assertions": config["K"],
                "elapsed_seconds": time.time()-started}, allow_nan=False) + "\n")
    client_docs, artifacts = [], [journal]
    for client, local in enumerate(clients):
        broadcast(local, global_parameters, bn, allowed)
        values = {}
        for split in ("validation", "test"):
            features = np.load(prepared / (split+"_x.npy"), mmap_mode="r")
            labels = np.load(prepared / (split+"_y.npy"))
            probability = checked_probabilities(local, features)
            probability_path = attempt / f"client{client}_{split}_probabilities.npy"
            np.save(probability_path, probability, allow_pickle=False)
            artifacts.append(probability_path)
            if split == "validation":
                threshold = float(p.tune_threshold(labels, probability))
            values[split] = p.metrics(labels, probability, threshold)
        client_docs.append({"client": client, "threshold": threshold, **values,
                            "bn_minima": guard(local, "final")})
    checkpoint = attempt / "final_checkpoint.pt"
    torch.save({"global_non_bn": global_parameters,
                "client_bn": [OrderedDict((name, local.state_dict()[name].clone()) for name in sorted(bn)) for local in clients]}, checkpoint)
    artifacts.append(checkpoint)
    doc = {"status": "COMPLETED", "config": config, "config_sha256": p.canonical_sha(config),
           "dataset": dataset, "method": method, "seed": seed, "clients": client_docs,
           "client_rows": list(map(len, partitions)), "client_fraud_counts": [int(y[indices].sum()) for indices in partitions],
           "partition_sha256": [hashlib.sha256(indices.tobytes()).hexdigest() for indices in partitions],
           "payload_keys": list(allowed), "bn_keys": sorted(bn), "no_bn_transmitted": True,
           "payload_assertions": config["training"]["rounds"]*config["K"], "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "device": "cpu",
           "artifact_sha256": {str(item.relative_to(OUT)): p.sha(item) for item in artifacts},
           "elapsed_seconds": time.time()-started, "completed_at": p.now()}
    for split in ("validation", "test"):
        doc[split] = {endpoint: float(np.mean([client[split][endpoint] for client in client_docs]))
                      for endpoint in ("f1", "auc_roc", "pr_auc")}
    write(path, doc)
    valid_result(path, config)
    return {"path": str(path), "skipped": False}


