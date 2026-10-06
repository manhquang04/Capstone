"""Check the final deliverable set and record all touched branch paths/checksums."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def main():
    for script in HERE.glob('*.py'):
        ast.parse(script.read_text(), filename=str(script))
    required = ['CHECKLIST.md','run_reference.py','run_paysim_ablation.py','verify_reference.py','write_notes.py','run.log','ablation.log','failed_preflight.log','failed_torch_leaf.log','verification.json','verification_ablation.json','tabular_reference_notes.md']
    for filename in required:
        assert (HERE/filename).is_file() and (HERE/filename).stat().st_size > 0, filename
    original = json.loads((HERE/'verification.json').read_text())
    variant = json.loads((HERE/'verification_ablation.json').read_text())
    for scenario in ['adult_native','paysim_official_fc','paysim_project_eval']:
        base = HERE/'runs/default_batch8_seed42_v3'/scenario
        for filename in ['arrays.npz','inputs.npz','adapter.json','model.pt','result.json']:
            assert (base/filename).is_file(), (scenario,filename)
        expected = original[scenario]['array_sha256']
        assert hashlib.sha256((base/'arrays.npz').read_bytes()).hexdigest() == expected
    for scenario in ['paysim_official_fc','paysim_project_eval']:
        base = HERE/'runs/paysim_no_sigmoid'/scenario
        for filename in ['arrays.npz','inputs.npz','adapter.json','result.json']:
            assert (base/filename).is_file(), (scenario,filename)
        assert hashlib.sha256((base/'arrays.npz').read_bytes()).hexdigest() == variant[scenario]['array_sha256']
        assert variant[scenario]['accuracy_percent'] > original[scenario]['accuracy_percent']
    notes = (HERE/'tabular_reference_notes.md').read_text()
    for token in ['95.535714','83.333333','76.388889','zero fraud-positive','0.24451710283756256','0.39104539155960083']:
        assert token in notes, token
    paths = []
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and path.name != 'outputs_manifest.json':
            paths.append({'path':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    inventory = {'files':paths,'inventory_path':'external_defenses/reference_tabular/outputs_manifest.json','published_native_reference':'95.2 ± 8.8%, 50 batches; single measured batch is not aggregate reproduction','all_checks_passed':True}
    (HERE/'outputs_manifest.json').write_text(json.dumps(inventory,indent=2))
    print(f'PASS: complete final deliverables; {len(paths)} branch files inventoried plus outputs_manifest.json; frozen array checksums intact')

if __name__ == '__main__':
    main()
