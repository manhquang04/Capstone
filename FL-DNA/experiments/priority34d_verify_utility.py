"""Independent completed utility calibration/checkpoint audit, no training."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments import priority34d_image_utility as u
from experiments.priority34d_continue import live


def registry(grid, settings, mechanisms, core):
    result = []
    for setting in settings:
        arms = [('baseline', None, None)] + [(a, None, None) for a in u.DNA] if core else []
        arms += [(m+'_dp_'+format(s, 'g'), m, s) for m in mechanisms for s in grid]
        for method, mechanism, sigma in arms:
            for seed in range(51016, 51032):
                result.append(dict(id=setting+'__'+method+'__'+str(seed), setting=setting,
                                   method=method, mechanism=mechanism, sigma=sigma, seed=seed))
    return result


def paired_vector(values):
    array = np.asarray(values, dtype=float)
    if array.shape != (16,) or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError('invalid complete paired accuracy vector')
    return array


def bracket(baseline, dna, dp):
    base, target = paired_vector(baseline), paired_vector(dna)
    threshold = float(np.mean(target-base))-.005
    means = {float(s):float(np.mean(paired_vector(a)-base)) for s,a in dp.items()}
    if not means:
        raise ValueError('empty calibration grid')
    ordered = sorted(means)
    passing = [s for s in ordered if means[s] >= threshold]
    sigma = passing[-1] if passing else None
    higher = next((s for s in ordered if sigma is not None and s > sigma), None)
    baseline_mean = float(base.mean())
    matched = baseline_mean >= .40 and higher is not None and means[higher] < threshold
    return dict(status='MATCHED' if matched else 'NOT_ASSESSABLE', sigma=sigma,
                next_sigma=higher, target_delta=float(np.mean(target-base)), threshold=threshold,
                grid_means={str(s):v for s,v in means.items()}, baseline_mean=baseline_mean,
                extension_needed=baseline_mean >= .40 and sigma == ordered[-1])


def selection(rows, pairs=()):
    lookup = {(r['job']['setting'], r['job']['method'], r['job']['seed']):r['validation_accuracy'] for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('duplicate utility results')
    def vector(setting, arm):
        return [lookup[(setting, arm, seed)] for seed in range(51016,51032)]
    matches = {}
    for setting in u.SETTINGS:
        for mechanism in ('single', 'per_tensor'):
            grid = u.GRID+u.EXTENSION if (setting,mechanism) in pairs else u.GRID
            dp = {s:vector(setting,mechanism+'_dp_'+format(s,'g')) for s in grid}
            for dna in u.DNA:
                matches[setting+'::'+dna+'::'+mechanism] = bracket(vector(setting,'baseline'),vector(setting,dna),dp)
    return matches


def validation_accuracy(model, x, labels):
    model.eval()
    correct = 0
    with u.torch.no_grad():
        for start in range(0,len(x),512):
            logits = model(x[start:start+512])
            u.finite(logits, 'independent utility validation logits')
            correct += int((logits.argmax(1)==labels[start:start+512]).sum())
    return correct/len(x)


def verify():
    complete = u.read(u.BASE/'UTILITY_COMPLETE.json')
    if live():
        raise RuntimeError('science workers alive; defer utility audit')
    u.verify(u.PREP); freeze = u.verify()
    digest = u.sha(u.FREEZE)
    assert complete['freeze_sha256']==digest
    assert complete['calibration_sha256']==u.sha(u.BASE/'calibration.json')
    initial = registry(u.GRID, u.SETTINGS, ('single','per_tensor'), True)
    assert initial==freeze['jobs'] and len(initial)==1216
    calibration = u.read(u.BASE/'calibration.json')
    pairs = [tuple(p) for p in calibration['conditional_pairs']]
    assert pairs==sorted(set(pairs))
    scheduled = initial + [j for setting,mechanism in pairs
                           for j in registry(u.EXTENSION,[setting],[mechanism],False)]
    _, _, vx, vy = u.old.p31.utility_data()
    vx = vx.contiguous()
    rows = []
    for job in scheduled:
        parent = u.BASE/'jobs'/job['id']
        receipt = u.read(parent/'validated.json')
        assert receipt['job']==job and receipt['status']=='PASS' and receipt['freeze_sha256']==digest
        for name, sha in receipt['outputs'].items():
            path = (parent/name).resolve()
            assert path.is_relative_to(parent.resolve()) and u.sha(path)==sha
        names = [n for n in receipt['outputs'] if n.endswith('/result.json')]
        assert len(names)==1
        folder = (parent/names[0]).parent
        row = u.read(folder/'result.json')
        assert row['job']==job and row['status']=='PASS' and row['freeze_sha256']==digest
        assert row['steps']==4700 and row['device']=='cpu' and row['test_selection'] is False
        assert row['intra_threads']==row['inter_threads']==1
        assert str((folder/'final_state.pt').relative_to(parent)) in receipt['outputs']
        model = u.starting_model(job['setting'])
        model.load_state_dict(u.torch.load(folder/'final_state.pt',map_location='cpu',weights_only=False))
        u.check(model)
        accuracy = validation_accuracy(model,vx,vy)
        assert np.isfinite(row['validation_accuracy']) and abs(accuracy-row['validation_accuracy'])<=1e-12
        rows.append(dict(job=job,validation_accuracy=accuracy))
    triggers = selection(rows)
    expected_pairs = sorted({(k.split('::')[0],k.split('::')[2]) for k,v in triggers.items() if v['extension_needed']})
    assert pairs==expected_pairs
    conditional = u.BASE/'conditional_extension_schedule.json'
    if pairs:
        schedule = u.read(conditional)
        assert schedule['freeze_sha256']==digest and schedule['jobs']==scheduled[len(initial):]
        assert schedule['pairs']==[list(p) for p in pairs] and schedule['trigger_matches']==triggers
    else:
        assert not conditional.exists()
    assert calibration['matches']==selection(rows,pairs)
    assert calibration['initial_jobs']==1216 and calibration['freeze_sha256']==digest
    for setting in u.SETTINGS:
        clip = u.read(u.BASE/setting/'clip.json')
        assert clip['initial_state_sha256']==u.sha(u.BASE/setting/'initial_state.pt')
        assert clip['probe_seeds']==[51000,51003] and clip['training_batch']==256 and clip['quantile']==.95
        assert set(clip['names'])==set(clip['norms']) and len(set(clip['names']))==len(clip['names'])
        vectors = [clip['global_norms']] + [clip['norms'][n] for n in clip['names']]
        limits = [clip['C']]+clip['clips']
        assert len(vectors)==len(limits)
        for values, limit in zip(vectors,limits):
            a = np.asarray(values)
            assert a.shape==(94,) and np.isfinite(a).all() and (a>0).all()
            assert abs(float(np.quantile(a,.95))-limit)<=1e-12
    doc = dict(status='PASS_UTILITY_STAGE_ONLY',priority34d_complete=False,at=time.time(),
               freeze_sha256=digest,jobs=len(rows),results=rows,independent_brackets=True,
               checkpoint_validation_recomputed=True,probe_gradients_recomputed=False)
    u.write(u.OUT/'audits/independent_utility_results.json',doc)
    return doc


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--verify',action='store_true')
    if not parser.parse_args().verify:
        parser.error('choose --verify after completed utility science')
    try:
        doc = verify(); print(json.dumps(dict(status=doc['status'],jobs=doc['jobs'])))
    except Exception as error:
        u.write(u.OUT/('audits/utility_verifier_failure_'+str(time.time_ns())+'.json'),
                dict(error=repr(error),at=time.time(),priority34d_complete=False))
        raise
