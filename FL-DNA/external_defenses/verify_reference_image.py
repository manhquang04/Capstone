"""Independently recompute image metrics from final saved tensors."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'external_defenses/invertinggradients'))
import inversefed

torch.set_num_threads(1)
dataset = CIFAR10(str(ROOT/'datasets/cifar10'), train=False, download=False, transform=ToTensor())
checked = {}
for folder in sorted((ROOT/'external_defenses/reference_image').iterdir()):
    if not (folder/'metrics.json').exists():
        continue
    measured = json.loads((folder/'metrics.json').read_text())
    truth = dataset[measured['index']][0].unsqueeze(0)
    estimate = torch.load(folder/'reconstruction.pt', weights_only=True)
    a = truth[0].permute(1,2,0).numpy()
    b = estimate[0].permute(1,2,0).numpy()
    psnr = float(peak_signal_noise_ratio(a,b,data_range=1.))
    ssim = float(structural_similarity(a,b,data_range=1.,channel_axis=2))
    mse = float(torch.square(truth-estimate).mean())
    assert abs(mse-measured['metrics']['raw_mse']) < 1e-7
    assert abs(psnr-measured['metrics']['psnr_db']) < 1e-5
    assert abs(ssim-measured['metrics']['ssim']) < 1e-7
    official_psnr = float(inversefed.metrics.psnr(estimate,truth,factor=1.))
    checked[folder.name] = dict(raw_mse=mse,psnr_db=psnr,ssim=ssim,official_psnr_db=official_psnr)
out = ROOT/'external_defenses/reference_image/verification.json'
out.write_text(json.dumps(checked,indent=2)+'\n')
print(json.dumps(checked,indent=2))
