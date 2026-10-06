"""CPU official Geiping controls; run from FL-DNA with -B and two threads."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'external_defenses/invertinggradients'))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity
from PIL import Image
import inversefed


def metrics(raw, estimate):
    a = raw[0].detach().permute(1, 2, 0).numpy()
    b = estimate[0].detach().permute(1, 2, 0).numpy()
    mse = float(np.mean((a-b)**2))
    return dict(raw_mse=mse, psnr_db=float(-10*np.log10(mse)),
                ssim=float(structural_similarity(a, b, data_range=1., channel_axis=2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', choices=['published', 'project', 'project_local'], required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--tv', type=float, default=0.01)
    parser.add_argument('--iterations', type=int, default=4800)
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--output-name')
    parser.add_argument('--unsigned', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    out = ROOT / 'external_defenses/reference_image' / (args.output_name or args.setting)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists():
        raise FileExistsError('Preserve existing result; choose a new output before rerunning')
    data = CIFAR10(str(ROOT/'datasets/cifar10'), train=False, download=False, transform=ToTensor())
    raw, label = data[args.index]
    raw = raw.unsqueeze(0)
    labels = torch.tensor([label])
    train = CIFAR10(str(ROOT/'datasets/cifar10'), train=True, download=False)
    # Accumulate without copying the dataset to another directory.
    mean = torch.from_numpy(train.data.mean(axis=0).astype('float32')).permute(2,0,1).unsqueeze(0)/255.
    config = dict(signed=not args.unsigned, boxed=True, cost_fn='sim', indices='def', weights='equal',
                  lr=0.1, optim='adam', restarts=args.restarts, max_iterations=args.iterations,
                  total_variation=args.tv, init='randn', filter='none', lr_decay=True,
                  scoring_choice='loss')
    if args.setting == 'published':
        model, key = inversefed.construct_model('LeNetZhu', seed=42)
        dm = torch.tensor(inversefed.consts.cifar10_mean)[:,None,None]
        ds = torch.tensor(inversefed.consts.cifar10_std)[:,None,None]
        inputs = (raw-dm)/ds
        model.eval()
        loss = F.cross_entropy(model(inputs), labels)
        observed = [v.detach() for v in torch.autograd.grad(loss, model.parameters())]
        attack = inversefed.GradientReconstructor(model, (dm,ds), config.copy(), num_images=1)
    else:
        from experiments.run_priority7_image_domain_gate import LeNetCIFAR, _named_update
        model = LeNetCIFAR().train()
        dm, ds = torch.zeros(3,1,1), torch.ones(3,1,1)
        inputs = raw
        update = _named_update(model, inputs, labels, 0.01)
        # Single SGD delta is -0.01*g; restore g with the correct sign.
        observed = [v.detach()/(-0.01) for v in update.values()]
        attack = inversefed.GradientReconstructor(model, (dm,ds), config.copy(), num_images=1)
    start = time.monotonic()
    if args.setting == 'project_local':
        from experiments.priority23_repaired_cifar_attack import optimize_one
        update = {k: v.detach() for k,v in update.items()}
        local = optimize_one(model=model, labels=labels, target_images=raw, signal=update,
            payload=('none', None), branch='none', budget=dict(iterations=3000,attack_lr=0.05,tv_lambda=1e-4),
            seed=42, group=0, restart=0, local_lr=0.01)
        reconstructed, stats = local['reconstruction'], dict(opt=local['best_objective'], best_step=local['best_step'])
        config = dict(optim='adam', lr=0.05, max_iterations=3000, restarts=1,
                      total_variation=1e-4, signed=False, lr_decay=False, parameterization='sigmoid logits')
    else:
        output, stats = attack.reconstruct(observed, labels, img_shape=(3,32,32))
        reconstructed = output*ds+dm
    elapsed = time.monotonic()-start
    baseline = dict(gray=metrics(raw, torch.full_like(raw,.5)), cifar_train_mean=metrics(raw,mean))
    result = dict(setting=args.setting, index=args.index, split='test', model_seed=42, label=label,
        parameter_count=sum(p.numel() for p in model.parameters()), config=config,
        local_sgd_lr=0.01 if args.setting!='published' else None,
        metrics=metrics(raw,reconstructed), baselines=baseline, elapsed_seconds=elapsed, stats=dict(stats),
        torch_version=torch.__version__, threads=torch.get_num_threads())
    result['beats_both_baselines'] = all(result['metrics']['raw_mse']<v['raw_mse'] for v in baseline.values())
    torch.save(reconstructed, out/'reconstruction.pt')
    Image.fromarray((reconstructed[0].clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')).save(out/'reconstruction.png')
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()
