"""Display NA for absent optional report metadata without editing any data."""
from pathlib import Path
from experiments import priority35b_label_coordinates as labels

FREEZE='artifacts/priority35b/b3/report_optional_columns_freeze.json'


def verify():
    p=labels.presentation;d=p.i.read(Path(p.ROOT)/FREEZE)
    for f,h in {**d['sources'],**d['inputs']}.items():
        if p.i.sha(Path(p.ROOT)/f)!=h:raise ValueError('report adapter drift')
    return d


def main():
    verify();p=labels.presentation;old=p.install
    def install():
        old();base=p.audit.verify_freeze
        def augmented():
            d=base();f=verify();d['files'].update(f['sources']);d['files'].update(f['inputs'])
            d['files'][FREEZE]=p.i.sha(Path(p.ROOT)/FREEZE)
            return d
        p.audit.verify_freeze=augmented
    p.install=install
    original=p.publication.text.table
    def table(rows,fields):
        # Report-only projection. Original dictionaries and numeric values remain untouched.
        fields=tuple(dict.fromkeys([*fields,*[f for r in rows for f in r]]))
        projected=[dict(r,**{f:None for f in fields if f not in r}) for r in rows]
        return original(projected,fields)
    p.publication.text.table=table
    labels.main()


if __name__=='__main__':main()
