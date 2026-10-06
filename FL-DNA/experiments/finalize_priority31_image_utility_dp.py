"""Finalize P31 reporting after the frozen runner completes; no training/attacks."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import priority31_image_utility_dp as p
import numpy as np


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |',
            '| '+' | '.join(['---']*len(headers))+' |',
            *['| '+' | '.join(str(x) for x in row)+' |' for row in rows], '']


def fmt(value):
    return f'{float(value):.9g}'


def finish():
    p.analysis()
    calibration = p.read(p.OUT/'calibration.json')
    selection = p.read(p.OUT/'selection.json')
    results = p.read(p.OUT/'analysis_summary.json')
    with (p.OUT/'expanded_holm.csv').open() as stream:
        holm = {r['hypothesis_id']:float(r['holm_p']) for r in csv.DictReader(stream)}
    rows = []
    if (p.OUT/'utility_per_seed.csv').exists():
        with (p.OUT/'utility_per_seed.csv').open() as stream:
            rows = list(csv.DictReader(stream))
    grid = []
    for method in sorted(set(r['method'] for r in rows)):
        subset = [r for r in rows if r['method']==method]
        val = [float(r['validation_accuracy']) for r in subset]
        delta = [float(r['paired_validation_delta']) for r in subset]
        test = [float(r['test_accuracy']) for r in subset]
        grid.append(dict(method=method,n=len(subset),validation_mean=float(np.mean(val)),
                         validation_sd=float(np.std(val,ddof=1)),delta_mean=float(np.mean(delta)),
                         delta_sd=float(np.std(delta,ddof=1)),test_mean=float(np.mean(test)),
                         test_sd=float(np.std(test,ddof=1))))
    p.csv_write(p.OUT/'utility_grid_summary.csv',grid)
    checks={}
    for module in ['experiments/priority31_image_utility_dp.py',
                   'experiments/finalize_priority31_image_utility_dp.py',
                   'experiments/supervise_priority31_image_utility_dp.py',
                   'tests/test_priority31_image_utility_dp.py']:
        run=subprocess.run([sys.executable,'-m','py_compile',module],cwd=ROOT,
                           capture_output=True,text=True)
        checks['py_compile:'+module]=dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr)
    run=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    checks['git_diff_check']=dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr)
    run=subprocess.run([sys.executable,'-m','tests.test_priority31_image_utility_dp'],
                       cwd=ROOT,capture_output=True,text=True)
    checks['synthetic_tests']=dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr)
    p.save(p.OUT/'final_checks.json',checks)
    assert all(r['returncode']==0 for r in checks.values()),checks
    events=[json.loads(line) for line in (p.OUT/'runs.jsonl').read_text().splitlines()]
    completions=[r for r in events if r['event']=='job_complete']
    attack_completions=[r for r in events if r['event']=='attack_complete']
    lines=['# Priority 31 — CIFAR-10 image utility-matched DP', '',
           'Status: COMPLETED (unavailable cells remain NOT_ASSESSABLE).', '',
           'This adds evidence; it does not replace any earlier RQ1 result. No Latex/ or external_defenses/ file was edited.', '',
           '## Frozen design and provenance', '',
           '- Amendment: protocols/amendments/2026-10-01_priority31_image_utility_dp.md.',
           '- Official inversefed LeNet-Zhu, initialization seed 42, 15,826 trainable parameters, no BN buffers.',
           '- Utility-development scale: 12,000 CIFAR train images and 3,000 validation images; split source-ID overlap=0. Test images were excluded from all selections.',
           '- Paired seeds 51016–51031 vary minibatch order; initial weights are shared. Frozen P30 transform adapters retain their deterministic seeds (v1 30001, v2 30002); quantization seed per tensor is unchanged.',
           '- This is a development-scale utility match, not a claim of full-data converged CIFAR utility. Baseline quality was checked on validation, not test.', '',
           f"Selected learning rate **{selection['selected']['lr']}**, epochs **{selection['selected']['epochs']}**, from two validation-selection seeds only. All six predeclared 100-epoch selection jobs were completed; their 30/60/100-epoch results are in selection.json.", '',
           f"C = **{p.read(p.OUT/'clip.json')['C']:.12g}**, p95 of 94 minibatch trainable-gradient norms at the fixed initial model. This is not a p95 estimated throughout training. Per-run actual clipping rates are in utility_per_seed.csv.", '',
           'DP clips the full trainable gradient vector and adds coordinate noise N(0,(sigma C)^2). RDP uses update-level add/remove adjacency, sensitivity ratio 1, delta=1e-5, one release; no record-level or subsampling claim.', '',
           '## Utility calibration', '']
    lines += table(['Method','n','Validation mean','Validation SD','Paired validation Δ','SD(Δ)','Test mean (descriptive)'],
                   [[r['method'],r['n'],fmt(r['validation_mean']),fmt(r['validation_sd']),
                     fmt(r['delta_mean']),fmt(r['delta_sd']),fmt(r['test_mean'])] for r in grid])
    if calibration['status']=='NOT_ASSESSABLE':
        lines += [f"**NOT_ASSESSABLE:** {calibration['reason']}. Baseline validation mean = {calibration['baseline_validation_mean']:.9f}; required >=0.40 (and >=0.35). No DP grid or reconstruction ran after this failed gate.", '']
    else:
        lines += [f"Baseline mean validation accuracy = {calibration['baseline_validation_mean']:.9f}; quality gate PASS.", '']
        lines += table(['Transform','Own paired Δ','Eligibility threshold','Selected sigma','Next sigma','Bracketed','ε (one release)'],
                       [[t,fmt(m['transform_delta']),fmt(m['threshold']),m['sigma'],m['next_sigma'],m['bracketed'],
                         fmt(m['epsilon_one_release']) if m['bracketed'] else 'NOT_ASSESSABLE']
                        for t,m in calibration['matches'].items()])
        lines += ['Matching used the largest eligible sigma and required the next-larger grid point to fail the threshold. Test accuracy did not enter this rule.', '']
    lines += ['## Reconstruction and exact paired tests', '',
              'Exact S1 test-image target IDs were reused, with source-ID equality asserted against S1, S1c and S0u per target. Existing E2 transform outputs were not rerun. Target IDs and image indices are in paired_*.csv.', '',
              'The official attack used cosine + TV=0.01, signed Adam lr=0.1, 4,800 iterations, one restart, boxed constraints and decay. Its input is only the noisy clipped gradient. Attack seed=30600+target_id; DP draw seed=510000+101*target_id+the frozen defense offset.', '',
              'A DNA win means DNA has strictly larger raw input-pixel MSE; exact equality is a tie. PSNR difference below is DNA minus DP: negative means less leakage under DNA.', '']
    lines += table(['Transform','DNA wins / DP wins / ties','Raw p(DNA greater error)','Holm p','Raw p(DP greater error)','Holm p'],
                   [[r['defense'],f"{r['dna_wins']} / {r['dp_wins']} / {r['ties']}",fmt(r['p_dna_greater']),
                     fmt(holm[f"P31::{r['defense']}::dna_greater"]),fmt(r['p_dp_greater']),
                     fmt(holm[f"P31::{r['defense']}::dp_greater"])] for r in results['comparisons']])
    lines += table(['Transform','Median PSNR DNA / DP','Median SSIM DNA / DP','Median paired PSNR Δ','Order interval ranks 13 / 27'],
                   [[r['defense'],f"{fmt(r['median_dna_psnr'])} / {fmt(r['median_dp_psnr'])}",
                     f"{fmt(r['median_dna_ssim'])} / {fmt(r['median_dp_ssim'])}",fmt(r['median_paired_psnr']),
                     f"[{fmt(r['rank13'])}, {fmt(r['rank27'])}]"] for r in results['comparisons']])
    lines += [f"Rank interval binomial coverage (continuous independent differences) = {results['rank_interval_coverage']:.9f}. This is the requested interval, not an automatically relabeled 95% interval.", '',
              f"Holm family size = **{results['holm_family_size']}**: the repaired P30 family's 111 finite raw hypotheses plus two directions for each available P31 cell. All old adjusted p-values were recomputed without altering their raw p-values or files. See expanded_holm.csv.", '',
              '## Direction and limitations', '']
    for r in results['comparisons']:
        a=holm[f"P31::{r['defense']}::dna_greater"]
        b=holm[f"P31::{r['defense']}::dp_greater"]
        verdict='DNA significantly larger reconstruction error' if a<.05 else ('DP significantly larger reconstruction error' if b<.05 else 'no significant direction under the expanded Holm family')
        lines += [f"- {r['defense']}: {verdict}."]
    if not results['comparisons']:
        lines += ['No qualified utility-matched reconstruction comparison is available; no privacy ranking is inferred.']
    lines += ['', 'All comparisons reuse fixed checkpoints and the existing targets, so this extension does not solve the checkpoint-clustering concerns of P18–P19. Non-significance is not proof of equivalence.', '',
              'all_image_ssim.csv exports descriptive per-arm medians from every already-run image E2/E3 stage (also E1), with superseded/NOT_APPLICABLE stages labeled explicitly. Invalid stages do not enter Holm.', '',
              '## Exact commands and every-run disclosure', '', '```sh',
              'PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -m tests.test_priority31_image_utility_dp',
              'PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -u experiments/priority31_image_utility_dp.py --stage analysis --workers 8',
              'nohup env PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -u experiments/priority31_image_utility_dp.py --stage all --workers 8 > artifacts/priority31_image_utility_dp/execution.log 2>&1 &',
              'nohup env PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -u experiments/priority31_image_utility_dp.py --stage all --workers 8 > artifacts/priority31_image_utility_dp/execution.log 2>&1 < /dev/null & disown',
              'PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -u experiments/priority31_image_utility_dp.py --stage all --workers 8',
              'PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -u experiments/finalize_priority31_image_utility_dp.py --wait',
              f'launchctl submit -l org.fl-dna.priority31 -o {p.OUT}/supervised_stdout.log -e {p.OUT}/supervised_stderr.log -- {ROOT}/external_defenses/.venv/bin/python -B -u {ROOT}/experiments/supervise_priority31_image_utility_dp.py',
              '```', '',
              'The first background launch terminated before invoking the runner. The second terminated after creating split/clip, before any training job. The tracked execution session resumed those unchanged outputs. No calibration outcome was used to alter settings. User interruption of the chat did not terminate the tracked training session.', '',
              'The tracked session subsequently disappeared after all six selection jobs and sixteen baseline replicates completed, while eight v1 jobs had started but had no completed output. Cause was not established. Those unfinished jobs restarted with identical settings; every completed job was skipped. A launchctl submission stalled before Python entered the supervisor (no journal or scientific outputs); that job was removed. The working supervisor instead used subprocess.Popen(start_new_session=True), with logs redirected to detached_stdout.log and detached_stderr.log. See detached_launch.json, runs.jsonl and supervisor.jsonl.', '',
              f"Completed utility/selection jobs: {len(completions)}. Completed reconstruction targets: {len(attack_completions)}. Full invocation/job/error/resume journal: runs.jsonl; timing per completed job is stored in JSON.", '',
              'Final checks: synthetic tests PASS; py_compile PASS; git diff --check PASS. final_checks.json contains exact check output. Experimental workload completed before finalization; thread count is 1 per worker.', '',
              '## Artifacts and hashes', '',
              'All new output paths and SHA-256 values are listed in sha256_manifest.csv (including every training result, reconstruction array/JSON, paired CSV, analysis CSV, split, clip, selection, calibration and checks). Report/manifest hashes are independently exported in final_hashes.json to avoid circular self-hashing.', '']
    p.REPORT.write_text('\n'.join(lines)+'\n')
    p.manifest()
    manifest=p.OUT/'sha256_manifest.csv'
    with manifest.open() as stream:
        hashes=list(csv.DictReader(stream))
    extra=[ROOT/'experiments/finalize_priority31_image_utility_dp.py',
           ROOT/'experiments/supervise_priority31_image_utility_dp.py']
    hashes += [dict(path=str(path.relative_to(ROOT)),sha256=p.sha(path)) for path in extra]
    p.csv_write(manifest,hashes)
    p.save(p.OUT/'final_hashes.json',dict(report_sha256=p.sha(p.REPORT),manifest_sha256=p.sha(manifest)))
    print('P31 final report and checks completed.',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    while True:
        events=[json.loads(line) for line in (p.OUT/'runs.jsonl').read_text().splitlines()]
        last=next((r for r in reversed(events) if r['event'] in ['invocation_complete','invocation_error','invocation']),None)
        if last and last['event']=='invocation_complete' and last.get('stage')=='all':
            break
        if last and last['event']=='invocation_error':
            raise RuntimeError('P31 runner failed; finalization refused: '+str(last))
        if not args.wait:
            raise RuntimeError('Frozen runner has not completed; finalization refused')
        time.sleep(30)
    finish()


if __name__=='__main__':
    main()
