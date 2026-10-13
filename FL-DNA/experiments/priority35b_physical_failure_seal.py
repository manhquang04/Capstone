"""Deduplicate only samefile aliases of exact approved preserved failure logs."""
from pathlib import Path
from experiments import priority35b_report_optional_columns as report
from experiments import priority35b_approved_infrastructure_seal as seal

FREEZE='artifacts/priority35b/b3/physical_failure_seal_freeze.json'


def verify():
    p=report.labels.presentation;d=p.i.read(Path(p.ROOT)/FREEZE)
    for f,h in {**d['sources'],**d['inputs']}.items():
        if p.i.sha(Path(p.ROOT)/f)!=h:raise ValueError('physical failure adapter drift')
    return d


def check(actual,approved):
    root=Path(report.labels.presentation.ROOT);seen={}
    for name,h in actual.items():
        candidates=[f for f,digest in approved.items() if h==digest and (root/name).samefile(root/f)]
        if len(candidates)!=1:raise ValueError('unapproved physical failure or changed digest')
        seen[candidates[0]]=h
    if seen!=approved:raise ValueError('missing or changed approved physical failure')


def main():
    verify();p=report.labels.presentation;old=p.install
    def install():
        old();base=p.audit.verify_freeze
        def augmented():
            d=base();f=verify();d['files'].update(f['sources']);d['files'].update(f['inputs'])
            d['files'][FREEZE]=p.i.sha(Path(p.ROOT)/FREEZE)
            return d
        p.audit.verify_freeze=augmented
    p.install=install;seal.verify_failure_allowlist=check;report.main()


if __name__=='__main__':main()
