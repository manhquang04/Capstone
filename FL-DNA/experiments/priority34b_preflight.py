"""Persist before-freeze test/compile/runtime evidence in P34B only."""
import platform
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import priority34b_core as b


def main():
    if (b.OUT/'execution_freeze.json').exists():
        raise ValueError('preflight is pre-freeze only')
    checks=[]
    commands=[[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_priority34b.py','-v'],['git','diff','--check']]
    for command in commands:
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        checks.append(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:
            b.save(b.OUT/'PREFLIGHT_FAILED.json',dict(checks=checks)); raise ValueError('preflight failure')
    files=list((ROOT/'experiments').glob('*priority34b*.py'))+[ROOT/'tests/test_priority34b.py']
    folder=b.OUT/'compile_preflight'; folder.mkdir(exist_ok=True)
    for path in files:
        py_compile.compile(str(path),cfile=str(folder/(path.stem+'.pyc')),doraise=True)
    free=shutil.disk_usage(ROOT).free
    if free<30*1024**3:
        raise ValueError('insufficient storage gate')
    b.save(b.OUT/'preflight_checks.json',dict(status='PASS',at=b.p.now(),checks=checks,
        compiled=[str(path.relative_to(ROOT)) for path in files],device='cpu',torch_threads=b.torch.get_num_threads(),
        torch_interop_threads=b.torch.get_num_interop_threads(),disk_free_bytes=free,platform=platform.platform(),
        python=sys.version,torch=b.torch.__version__,numpy=b.np.__version__,
        initial_import_failure_preserved='preflight_import_failure.json'))
    b.progress('preflight_pass',0,470)
    print('PREFLIGHT_PASS',len(files),'CPU/thread1')


if __name__=='__main__':
    main()
