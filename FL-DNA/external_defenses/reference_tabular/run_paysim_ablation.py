"""Prespecified diagnostic: same saved real gradients, omit wide-range sigmoid only."""
from run_reference import HERE, ROOT, PaySimAdapter, load_module, official_config, gradient, measure
import json
import time
import numpy as np
import torch
from models import FullyConnected
from attacks import invert_grad

def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    base = HERE/'runs/default_batch8_seed42_v3'
    loader = load_module('ablation_readonly_loader', ROOT/'data/load_creditcard.py')
    clients, _, _, dim, pos_weight, metadata = loader.load_creditcard_data(batch_size=8,max_rows=100000,seed=42)
    train = (torch.cat([c.dataset.tensors[0] for c in clients]), torch.cat([c.dataset.tensors[1] for c in clients]))
    dataset = PaySimAdapter(train, metadata, loader.NUMERIC_COLUMNS)
    for name in ('paysim_official_fc','paysim_project_eval'):
        source = base/name
        out = HERE/'runs/paysim_no_sigmoid'/name
        out.mkdir(parents=True,exist_ok=False)
        original = json.loads((source/'result.json').read_text())
        inputs = np.load(source/'inputs.npz')
        x, y = torch.tensor(inputs['truth']), torch.tensor(inputs['labels'])
        if name == 'paysim_official_fc':
            net = FullyConnected(dim,[100,100,2])
            criterion = torch.nn.CrossEntropyLoss()
        else:
            module = load_module('ablation_project_model',ROOT/'models/fraud_mlp.py')
            net = module.FraudMLP(dim)
            net.eval()
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        net.load_state_dict(torch.load(source/'model.pt',weights_only=True))
        true_grad = gradient(net,criterion,x,y)
        config = official_config()
        config['sigmoid_trick'] = False
        np.random.seed(42)
        torch.manual_seed(42)
        print('START diagnostic no-sigmoid',name,flush=True)
        start = time.monotonic()
        rec,ensemble,losses = invert_grad(net=net,training_criterion=criterion,true_grad=true_grad,true_label=y,true_data=torch.empty_like(x),dataset=dataset,**config)
        elapsed = time.monotonic()-start
        np.savez(out/'arrays.npz',truth=x.numpy(),labels=y.numpy(),reconstruction=rec.detach().numpy(),ensemble=np.stack([r.numpy() for r in ensemble]),objective_losses=np.array(losses))
        np.savez(out/'inputs.npz',**{k:inputs[k] for k in inputs.files})
        (out/'adapter.json').write_text((source/'adapter.json').read_text())
        result = dict(original,config=config,metric=measure(dataset,x,rec),runtime_seconds=elapsed,diagnostic='Disable sigmoid reparameterization only; preserve original robust-scaled inputs, identical source checkpoint/batch/criterion',source_run=str(source.relative_to(ROOT)))
        result['normalized_mse_unaligned_not_primary'] = float(((rec-x)**2).mean())
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print('DONE diagnostic no-sigmoid',name,result['metric'],elapsed,flush=True)

if __name__ == '__main__':
    main()
