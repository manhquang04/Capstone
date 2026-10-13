"""Post-exit, exclusive-create rendering; stops before visual QA/publication.

Invoke only after the stable-evidence preparation wrapper has itself exited.
The caller must emit the PDF artifact-operation marker before actual rendering.
No measurements or optimization are repeated, and this cannot mark COMPLETE.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from experiments import priority35b_leakage_verified_preparation as wrapper
from experiments import priority35b_leakage_tables as tables
from experiments import priority35b_leakage_data_bridge as bridge
from experiments import priority35b_leakage_figure as figure

prepare=wrapper.prepare
integrity=prepare.integrity
SEAL='render_integration_freeze.json'


def require_ready(doc):
    if (doc.get('status')!='VERIFIED_PREPARATION_READY_FOR_RENDER' or
        doc.get('measurements')!=9984 or doc.get('scientific_files')!=49920 or
        doc.get('measurement_repeated') is not False or
        doc.get('full_priority35b_complete') is not False):
        raise ValueError('complete stable preparation receipt required')


def require_actor_exit(snapshot,pid):
    if not isinstance(pid,int) or pid<=0:raise ValueError('preparation PID required')
    if any(actor==pid for actor,_,_ in snapshot):
        raise ValueError('preparation process still alive, including PID reuse')


def evidence_path(root,relative):
    path=integrity.child(root,relative)
    if not path.resolve().is_relative_to((Path(root)/'artifacts/priority35b/b3').resolve()):
        raise ValueError('render evidence outside B3')
    return path


def verify_ready(root,relative):
    prepare.no_live(root)  # Before reading any protected quality.
    wrapper.verify_sources(root)
    out=Path(root)/'artifacts/priority35b/b3'
    prepare.scientific.verify_additive_sources(root,(SEAL,))
    for path,digest in integrity.read(out/SEAL)['inputs'].items():
        if integrity.sha(integrity.child(root,path))!=digest:raise ValueError('render dependency drift')
    ready_path=evidence_path(root,relative);ready=integrity.read(ready_path);require_ready(ready)
    require_actor_exit(prepare.postexit.process_snapshot(),integrity.read(ready_path.parent/'START.json')['pid'])
    if (ready_path.parent/'FAILURE.json').exists():raise ValueError('failed preparation requires direction')
    attempt=evidence_path(root,ready['publication_attempt'])
    receipt=attempt/'PREPARED.json'
    if integrity.sha(receipt)!=ready['prepared_receipt_sha256']:raise ValueError('preparation receipt drift')
    doc=integrity.read(receipt)
    if (doc['status']!='B3_TABLES_VERIFIED_FIGURE_AND_PUBLICATION_PENDING' or
        doc['scientific_jobs']!=9984 or doc['controls']!=312 or doc['fixed_holm']!=144 or
        doc['csv_recomputation']!='PASS' or doc['figure_data']!='PASS' or
        doc['measurement_repeated'] is not False or doc['full_priority35b_complete'] is not False):
        raise ValueError('preparation audit incomplete')
    for section in ('inputs','outputs'):
        wrapper.manifest.verify_mapping(doc[section],lambda p:integrity.sha(integrity.child(root,p)))
    for field in ('ledger','stability'):
        path=evidence_path(root,ready[field+'_path'])
        if integrity.sha(path)!=ready[field+'_sha256']:raise ValueError(field+' receipt drift')
    hashes=integrity.read(evidence_path(root,ready['ledger_path']))
    stable=integrity.read(evidence_path(root,ready['stability_path']))
    if len(hashes)!=49920 or stable!=wrapper.check_stability(hashes,hashes):raise ValueError('stability inventory')
    wrapper.manifest.verify_mapping(hashes,lambda p:integrity.sha(integrity.child(root,p)))
    return attempt,ready_path


def recheck_data(root,attempt):
    arms=integrity.read(Path(root)/'artifacts/priority35b/b3/arms.json')
    bundle=integrity.read(attempt/'utility_and_matching.json')
    texts=[(attempt/(name+'.csv')).read_text() for name in ('per_target','frontier','holm')]
    csv_receipt=tables.verify_csvs(*texts,arms,bundle['matching'])
    records=tables.parse_csv(texts[0]);frontier=tables.parse_csv(texts[1])
    rows,controls=bridge.prepare_figure(records,arms,frontier,bundle['utility'],integrity.read(attempt/'estimator_selection.json'))
    if rows!=tables.parse_csv((attempt/'figure_data.csv').read_text()) or controls!=integrity.read(attempt/'data_free_lines.json'):
        raise ValueError('figure CSV/controls differ from independently checked join')
    proof=bridge.independently_verify_figure(rows,controls,records,bundle['utility'])
    return rows,controls,dict(csv=csv_receipt,figure=proof)


def render(relative,root=prepare.ROOT):
    root=Path(root).resolve();attempt,ready_path=verify_ready(root,relative)
    run=attempt/('render_attempt_'+str(time.time_ns()));run.mkdir()
    prepare.new_json(run/'START.json',dict(at=prepare.now(),pid=os.getpid(),command=[sys.executable,*sys.argv],
        ready_sha256=integrity.sha(ready_path),measurement_repeated=False))
    try:
        rows,controls,proof=recheck_data(root,attempt)
        renderer=shutil.which('pdftoppm')
        if not renderer:raise ValueError('Poppler pdftoppm required for PDF visual QA')
        directory=root/'artifacts/priority35b/figure'
        if any((directory/name).exists() for name in ('frontier.pdf','frontier.png','frontier_data.csv','rendered.png')):
            raise ValueError('preserve existing figure assets; no automatic replay')
        directory.mkdir(parents=True,exist_ok=True)
        prepare.new_bytes(directory/'frontier_data.csv',(attempt/'figure_data.csv').read_bytes())
        fig=figure.make_figure(rows,controls);pdf,png=figure.save_new(fig,directory)
        command=[renderer,'-png','-singlefile','-r','150',str(pdf),str(directory/'rendered')]
        result=subprocess.run(command,capture_output=True,text=True)
        prepare.new_json(run/'poppler.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode or not (directory/'rendered.png').is_file():raise ValueError('PDF rendering failed')
        verify_ready(root,relative);prepare.no_live(root)
        outputs={str(p.relative_to(root)):integrity.sha(p) for p in (pdf,png,directory/'frontier_data.csv',directory/'rendered.png')}
        receipt=dict(at=prepare.now(),status='RENDERED_AWAITING_VISUAL_QA',pid=os.getpid(),
            ready_receipt=relative,ready_sha256=integrity.sha(ready_path),outputs=outputs,data_recomputation=proof,
            visual_qa='PENDING',report_publication='PENDING',measurement_repeated=False,full_priority35b_complete=False)
        prepare.new_json(run/'RENDERED.json',receipt)
        return dict(receipt=str((run/'RENDERED.json').relative_to(root)),status=receipt['status'],full_priority35b_complete=False)
    except Exception:
        prepare.new_json(run/'FAILURE.json',dict(at=prepare.now(),traceback=traceback.format_exc(),
            measurement_repeated=False,requires_human_direction=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ready',required=True)
    args=parser.parse_args();print(__import__('json').dumps(render(args.ready),allow_nan=False))
