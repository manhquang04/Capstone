"""Authorized label-only new version and descriptive existing-data reporting."""
import argparse
import copy
import json
import os
import py_compile
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from experiments import priority35b_scipy_verifier_adapter as audit
from experiments import priority35b_leakage_render_publication as render
from experiments import priority35b_leakage_publish_report as publication
from experiments import priority35b_leakage_figure as figure

ROOT=render.prepare.ROOT
FREEZE='artifacts/priority35b/b3/presentation_sensitivity_freeze.json'
VERSION='artifacts/priority35b/figure/label_layout_v2'
CSV_SHA='e815f3ba6261bd70fa7e989e812292398a486aa9e30b012e23389cd90bbf1172'
i=render.integrity;p=render.prepare


def verify():
    doc=i.read(Path(ROOT)/FREEZE)
    for name,h in {**doc['sources'],**doc['inputs']}.items():
        if i.sha(i.child(ROOT,name))!=h:raise ValueError('presentation input/source drift: '+name)
    return doc


def install():
    audit.install();doc=verify();old=audit.verify_freeze
    def augmented():
        d=copy.deepcopy(old());f=verify()
        d['files'].update(f['sources']);d['files'].update(f['inputs'])
        d['files'][FREEZE]=i.sha(Path(ROOT)/FREEZE)
        return d
    audit.verify_freeze=augmented
    publication.contract.ASSETS=frozenset(VERSION+'/'+n for n in ('frontier.pdf','frontier.png','frontier_data.csv','rendered.png'))
    original=publication.text.section
    def section(bundle,checks):
        b=dict(bundle)
        for key,name in (('figure_pdf','frontier.pdf'),('figure_png','frontier.png'),('figure_csv','frontier_data.csv')):
            b[key]=str(Path(ROOT)/VERSION/name)
        return original(b,checks)
    publication.text.section=section
    return doc


def fingerprint(fig):
    """All scientific line/marker/error-bar artists and fixed axes, excluding annotations."""
    return [dict(xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),
        lines=[dict(x=np.asarray(l.get_xdata()).tolist(),y=np.asarray(l.get_ydata()).tolist(),
                    color=l.get_color(),marker=l.get_marker(),ls=l.get_linestyle(),lw=l.get_linewidth(),ms=l.get_markersize())
               for l in ax.lines],
        collections=[dict(segments=[np.asarray(s).tolist() for s in c.get_segments()],
                          colors=c.get_colors().tolist(),widths=c.get_linewidths().tolist()) for c in ax.collections])
        for ax in fig.axes]


def label_positions(fig,ax,labels):
    from matplotlib.transforms import Bbox
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=ax.get_window_extent(renderer)
    occupied=[]
    # Protect every marker, error bar and cap, not just earlier text labels.
    for line in ax.lines:
        if line.get_marker() not in ('None',None,''):
            coords=ax.transData.transform(np.column_stack((line.get_xdata(),line.get_ydata())))
            for x,y in coords:occupied.append(Bbox.from_extents(x-5,y-5,x+5,y+5))
    for col in ax.collections:
        for seg in col.get_segments():
            coords=ax.transData.transform(seg)
            occupied.append(Bbox.from_extents(coords[:,0].min()-2,coords[:,1].min()-2,
                                             coords[:,0].max()+2,coords[:,1].max()+2))
    for label,x,y,color in labels:
        anchor=ax.transData.transform((x,y))
        candidates=[((u+.5)/28,(v+.5)/16) for u in range(28) for v in range(16)]
        candidates.sort(key=lambda z:sum((a-b)**2 for a,b in zip(ax.transAxes.transform(z),anchor)))
        for z in candidates:
            text=ax.text(*z,label,transform=ax.transAxes,fontsize=7,color=color,ha='center',va='center')
            box=text.get_window_extent(renderer).expanded(1.22,1.35)
            if bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1) and not any(box.overlaps(b) for b in occupied):
                occupied.append(box)
                ax.annotate('',xy=(x,y),xytext=z,textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle='-',lw=.35,color=color,alpha=.55),zorder=1)
                break
            text.remove()
        else:raise ValueError('new label-only layout cannot fit')


def make_version(ready):
    root=Path(ROOT);attempt,_=render.verify_ready(root,ready);verify()
    rows,controls,proof=render.recheck_data(root,attempt)
    old=figure.make_figure(rows,controls);expected=fingerprint(old)
    old_label=figure.place_labels
    try:
        figure.place_labels=label_positions;fig=figure.make_figure(rows,controls)
    finally:figure.place_labels=old_label
    if fingerprint(fig)!=expected:raise ValueError('point/errorbar/scale/style changed')
    import matplotlib.pyplot as plt
    plt.close(old)
    directory=root/VERSION;directory.mkdir(exist_ok=False)
    if i.sha(root/'artifacts/priority35b/figure/frontier_data.csv')!=CSV_SHA:raise ValueError('original CSV drift')
    p.new_bytes(directory/'frontier_data.csv',(root/'artifacts/priority35b/figure/frontier_data.csv').read_bytes())
    pdf,png=figure.save_new(fig,directory)
    command=[shutil.which('pdftoppm'),'-png','-singlefile','-r','150',str(pdf),str(directory/'rendered')]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:raise ValueError(result.stderr)
    if i.sha(directory/'frontier_data.csv')!=CSV_SHA:raise ValueError('copied CSV not identical')
    verify();p.no_live(root)
    run=attempt/('presentation_attempt_'+str(time.time_ns()));run.mkdir()
    p.new_json(run/'START.json',dict(at=p.now(),pid=os.getpid(),command=[sys.executable,*sys.argv],measurement_repeated=False))
    outputs={str(f.relative_to(root)):i.sha(f) for f in (pdf,png,directory/'frontier_data.csv',directory/'rendered.png')}
    receipt=dict(at=p.now(),pid=os.getpid(),status='RENDERED_AWAITING_VISUAL_QA',ready_receipt=ready,
        ready_sha256=i.sha(root/ready),outputs=outputs,measurement_repeated=False,full_priority35b_complete=False,
        data_recomputation=proof,numeric_artist_fingerprint_identical=True,csv_sha256_identical=CSV_SHA,
        presentation_freeze_sha256=i.sha(root/FREEZE),visual_qa='PENDING')
    p.new_json(run/'RENDERED.json',receipt)
    p.new_json(run/'NUMERIC_ARTISTS.json',expected)
    return str((run/'RENDERED.json').relative_to(root))


def sensitivity(attempt):
    records=render.tables.parse_csv((attempt/'per_target.csv').read_text())
    frontier=render.tables.parse_csv((attempt/'frontier.csv').read_text())
    matching=i.read(attempt/'utility_and_matching.json')['matching'];rows=[]
    for dataset in ('baf','ieee_cis','paysim'):
        arms=['unprotected','dna_v1_conservative','dna_v2_0p95','local_dp_global_eps10','local_dp_per_tensor_eps10']
        arms+=sorted({m['matched_arm'] for m in matching if m['dataset']==dataset and m['matched_arm']})
        for arm in arms:
            for cell in ('ratio_batch1','tableak_batch1','tableak_batch2'):
                vals=[r['feature_accuracy'] for r in records if (r['dataset'],r['cell'],r['arm'])==(dataset,cell,arm)]
                if dataset=='ieee_cis' and cell=='tableak_batch2':
                    if vals:raise ValueError('unexpected nonqualified IEEE b2 data')
                    rows.append(dict(dataset=dataset,arm=arm,estimator=cell,n=0,median=None,interval=None,status='NOT_APPLICABLE: not qualified in P34C'))
                    continue
                if len(vals)!=39 or not np.isfinite(vals).all():raise ValueError('39 finite sensitivity units required')
                vals=sorted(vals);median=vals[19];interval=[vals[12],vals[26]]
                source=[r for r in frontier if (r['dataset'],r['cell'],r['arm'],r['metric'])==(dataset,cell,arm,'feature_accuracy')]
                if len(source)!=1 or (source[0]['median'],source[0]['interval'])!=(median,interval):raise ValueError('descriptive summary mismatch')
                rows.append(dict(dataset=dataset,arm=arm,estimator=cell,n=39,median=median,interval=interval,status='DESCRIPTIVE_EXISTING_RESULTS'))
    if len(rows)!=54:raise ValueError('54 sensitivity rows including six NA required')
    return rows


def reading(ready):
    root=Path(ROOT);attempt,_=render.verify_ready(root,ready);render.recheck_data(root,attempt);verify()
    rows=sensitivity(attempt)
    text='''BAF is the only dataset with a DP matched-utility arm: joint C=0.7420758605003357, epsilon=1,000,000. IEEE-CIS and PaySim match only clip-only C_med, with NO DP guarantee. All per-tensor matches remain NOT_ASSESSABLE. The primary ratio estimator indicates greater recovery from both transforms than matched DP on BAF; lower accuracy on IEEE-CIS versus clip-only; and no feature-accuracy difference established on PaySim. These are estimator-specific, different-configuration comparisons, not a universal privacy frontier or deployable Config-C record-recovery claim. All existing 144 paired tests remain authoritative; consult bounded-MSE and all-estimator contrasts too.

### Post-hoc descriptive estimator sensitivity (no new tests)

Selection followed the pre-registered highest-unprotected-median rule with fixed estimator-order tie break. Ratio b=1 was selected from a three-way 100% tie on BAF and PaySim, and a two-way 100% tie on IEEE-CIS; TabLeak b=2 was not qualified on IEEE-CIS and is explicitly NOT_APPLICABLE. No protected result selected the primary estimator.

The closed-form ratio is invariant to positive common rescaling of the gradient: joint clip-only rescales numerator and denominator equally, leaving the ratio and its 100% median accuracy intact. Additive noise does not cancel; the ratio is highly noise-sensitive here. This can favor DP on the ratio axis even when optimization-based TabLeak still recovers records. Under TabLeak b=1 the BAF matched DP arm leaks about as much as the transforms (all median feature accuracies 100%); at b=2 it is lower, 93.24% versus 100%, so this is descriptive similarity/high residual recovery, NOT equivalence. Existing Holm-significant BAF b=2 differences remain reported. IEEE-CIS and PaySim's matched clip-only arms likewise have 100% TabLeak medians, but must not be described as matched DP. Stronger C=0.01 epsilon=10 DP reduces TabLeak recovery much more, at substantial training utility loss.

The following medians and ranks-13/27 intervals use the same 39 row sets per available cell. They were independently recomputed from the already verified per-target CSV; no new statistical test, measurement, family or decision was introduced. Intervals and medians below are in percent.

| Dataset | Arm | Estimator | n | Median accuracy (%) | Ranks 13/27 interval (%) | Status |
| --- | --- | --- | --- | --- | --- | --- |
'''
    for r in rows:
        med='NA' if r['median'] is None else format(100*r['median'],'.6f')
        ci='NA' if r['interval'] is None else '['+', '.join(format(100*x,'.6f') for x in r['interval'])+']'
        text+=f"| {r['dataset']} | {r['arm']} | {r['estimator']} | {r['n']} | {med} | {ci} | {r['status']} |\n"
    text+='''
### Preserved numerical audit and presentation provenance

All 9,984 measurements were preserved. Original failed-verifier outputs/logs remain unchanged. Exact integer-binomial p-values and separately recomputed Holm values remain authoritative and bit-for-bit unchanged. SciPy is only a cross-check at relative 1e-12 with absolute floor 1e-15; maximum discrepancy was 2.220446049250313e-16. No raw or Holm value is within that envelope of 0.05 (the only p-value decision threshold): minimum distances 0.012499999999999997 and 0.030309277074411514. Approved float64 MSE reduction-order cross-check allows 1e-12 + 2*gamma_N*max(abs(values)), while requiring stored/native scores exactly; maximum MSE difference 1.4901161193847656e-08, median zero. Neither numerical allowance changed wins/losses/ties, raw/adjusted p-values, thresholds, matching, NA status, gates or figure points.

The post-hoc label-layout version preserves all original assets and failed visual-QA evidence. Only annotation positions and their leader-line endpoints changed. Point/error-bar/curve/style/scale fingerprints are identical to the original renderer; both CSV copies have SHA-256 e815f3ba6261bd70fa7e989e812292398a486aa9e30b012e23389cd90bbf1172. The new PDF/PNG were visually inspected at single-column width before publication. Neither axis selection nor the interpretation was changed by presentation.
'''
    run=attempt/('sensitivity_attempt_'+str(time.time_ns()));run.mkdir()
    p.new_json(run/'sensitivity.json',dict(status='DESCRIPTIVE_RECOMPUTATION_PASS',rows=rows,new_tests=0,holm_changes=0,measurement_repeated=False))
    inputs={str((attempt/(n+'.csv')).relative_to(root)):i.sha(attempt/(n+'.csv')) for n in ('frontier','holm')}
    p.new_json(run/'interpretation.json',dict(status='FULL_RESULTS_DESCRIPTIVE_READING',inputs=inputs,text=text))
    return str((run/'interpretation.json').relative_to(root))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['render','reading','publish','seal'],required=True)
    for name in ('ready','render','qa','validation','interpretation','published'):parser.add_argument('--'+name)
    a=parser.parse_args();install()
    if a.stage=='render':result=make_version(a.ready)
    elif a.stage=='reading':result=reading(a.ready)
    elif a.stage=='publish':result=publication.publish(a.ready,a.render,a.qa,a.validation,a.interpretation)
    else:
        from experiments import priority35b_approved_infrastructure_seal as seal
        result=seal.seal(a.published)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
