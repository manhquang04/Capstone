"""Summarize locked follow-up trials; never select configurations on confirmation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.run_phase3_full_client import dump, checksum


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    out=args.directory
    frozen=json.loads((out/'frozen.json').read_text())
    assert checksum(out/'protocol.json')==frozen['protocol_sha256']
    provenance=json.loads((out/'heldout_provenance.json').read_text())
    assert provenance['source_disjoint']
    records=[];pairs=[]
    for client in range(3):
        for restart in range(frozen['restarts']):
            folder=out/f'heldout_{client}'/f'restart_{restart}'
            assert (folder/'complete.json').exists()
            data=json.loads((folder/'results.json').read_text())
            assert data['variant']==frozen['selected']['variant']
            assert len(data['rows'])==2
            by_method={r['method']:r for r in data['rows']}
            saved=[]
            for method in ('baseline','zero_update'):
                row=by_method[method]
                assert row['budget']==frozen['iterations']
                tensor=torch.load(folder/f'{method}_{row["budget"]}.pt',weights_only=False)
                assert tensor['best']==min(tensor['history'])
                assert len(tensor['history'])==row['budget']+1
                saved.append(tensor['initial'])
                records.append(dict(client=client,restart=restart,**row))
            assert torch.equal(*saved)
            b=by_method['baseline'];z=by_method['zero_update']
            pairs.append(dict(client=client,restart=restart,
                baseline_minus_prior=b['metrics']['mean_mse']-b['prior']['mean_mse'],
                baseline_minus_zero=b['metrics']['mean_mse']-z['metrics']['mean_mse']))
    summary=[]
    for client in range(3):
        for method in ('baseline','zero_update'):
            rows=[r for r in records if r['client']==client and r['method']==method]
            vals=[r['metrics']['mean_mse'] for r in rows]
            summary.append(dict(client=client,method=method,mean_mse=float(np.mean(vals)),std=float(np.std(vals)),
                prior_mean=float(np.mean([r['prior']['mean_mse'] for r in rows])),
                median_mse=float(np.mean([r['metrics']['median_mse'] for r in rows])),
                best_at_budget=sum(r['best_step']==frozen['iterations'] for r in rows)))
    with (out/'confirmation_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    dev={v:json.loads((out/v/'results.json').read_text())['rows'] for v in ('raw','scaled','categorical','balanced')}
    for variant,rows in dev.items():
        for row in rows:
            saved=torch.load(out/variant/f'{row["method"]}_{row["budget"]}.pt',weights_only=False)
            assert saved['best']==min(saved['history'])
            assert len(saved['history'])==row['budget']+1
        last=500 if variant=='raw' else 100
        b=torch.load(out/variant/f'baseline_{last}.pt',weights_only=False)
        z=torch.load(out/variant/f'zero_update_{last}.pt',weights_only=False)
        assert torch.equal(b['initial'],z['initial'])
    dump(out/'followup_report.json',dict(frozen=frozen,development=dev,confirmation=summary,
        paired=pairs,details=records,provenance=provenance,
        baseline_prior_wins=sum(r['baseline_minus_prior']<0 for r in pairs),
        baseline_zero_wins=sum(r['baseline_minus_zero']<0 for r in pairs),
        clients_prior_consistent=sum(all(r['baseline_minus_prior']<0 for r in pairs if r['client']==c) for c in range(3)),
        note='Two restarts per client are not independent datasets; no automatic privacy claim'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in ('baseline','zero_update'):
        saved=torch.load(out/'raw'/f'{method}_500.pt',weights_only=False)
        axes[0].plot(np.minimum.accumulate(saved['history']),label=method)
        rows=[r for r in dev['raw'] if r['method']==method]
        axes[1].plot([r['budget'] for r in rows],[r['metrics']['mean_mse'] for r in rows],marker='o',label=method)
    axes[0].set(xlabel='Iteration',ylabel='Own objective (not leakage)',yscale='log')
    axes[1].set(xlabel='Iteration checkpoint',ylabel='Reconstruction MSE')
    for ax in axes: ax.legend()
    fig.tight_layout();fig.savefig(out/'development_convergence.png',dpi=150);plt.close(fig)
    dump(out/'manifest.json',{str(f.relative_to(out)):checksum(f) for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'})
    print(json.dumps(dict(summary=summary,pairs=pairs),indent=2))


if __name__=='__main__': main()
