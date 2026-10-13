"""Read-only P24–35 original-source manifest audit; no real target allocation."""
from __future__ import annotations
import json
import re
from pathlib import Path
from experiments import priority35a_history as previous
from experiments.priority35a_firewall import integer_ids
from experiments.priority35b_design import DATASETS


def sources(document, relative):
    """P35 mixed target manifests keep dataset identity on each target row."""
    result = {d: set() for d in DATASETS}
    if isinstance(document, dict) and document.get('dataset') in DATASETS:
        result[document['dataset']].update(previous.extract_ids(document))
    elif isinstance(document, dict) and isinstance(document.get('targets'), list):
        for row in document['targets']:
            identity = row.get('identity', '')
            dataset = row.get('dataset') or next((d for d in DATASETS if identity.startswith(d+'_')), None)
            if dataset not in DATASETS or not previous.extract_ids(row):
                raise ValueError('ambiguous target dataset/source IDs: '+relative)
            result[dataset].update(previous.extract_ids(row))
    else:
        dataset = previous.classify(relative, document)
        if dataset not in DATASETS:
            raise ValueError('P35 manifest requires explicit source namespace: '+relative)
        result[dataset].update(previous.extract_ids(document))
    if not any(result.values()):
        raise ValueError('target manifest missing original IDs: '+relative)
    return result


def candidates(root):
    paths = []
    for folder in sorted((root/'artifacts').iterdir()):
        if not folder.is_dir() or not re.match(r'^priority35(?:[^0-9]|$)', folder.name) or folder.name == 'priority35b':
            continue
        for path in sorted(folder.rglob('*')):
            if path.suffix in ('.json', '.pt') and ('target' in path.name.lower()
                    or 'reservation' in path.name.lower() or 'reservations' in path.parts):
                paths.append(path)
    return paths


def audit(root=previous.ROOT):
    root = Path(root).resolve()
    result = previous.audit(root)  # independently hash-audits every P24–34 candidate
    exclusions = {d: set(result['exclusions'][d]) for d in DATASETS}
    authoritative = {}
    for path in (root/'artifacts/priority35a').rglob('manifest.json'):
        for name, digest in json.loads(path.read_text()).get('hashes', {}).items():
            if name in authoritative and authoritative[name] != digest:
                # Authorized append-only report digests are irrelevant to target identity.
                if name.startswith('reports/'):
                    continue
                raise ValueError('conflicting source manifest hash: '+name)
            authoritative[name] = digest
    paths = candidates(root)
    if not paths:
        raise ValueError('missing P35 target inventory')
    for path in paths:
        relative = str(path.relative_to(root)); digest = previous.sha(path)
        if relative not in authoritative or authoritative[relative] != digest:
            raise ValueError('unsealed/drifted P35 source manifest: '+relative)
        doc = (json.loads(path.read_text()) if path.suffix == '.json'
               else previous.torch.load(path, map_location='cpu', weights_only=False))
        ids = sources(doc, relative)
        for d in DATASETS:
            exclusions[d].update(ids[d])
        result['inventory'].append(dict(path=relative, sha256=digest,
            dataset_counts={d: len(ids[d]) for d in DATASETS},
            source_ids_sha256={d: previous.digest(sorted(ids[d])) for d in DATASETS}))
    result.update(scope='P24–35 read-only firewall; no new real targets',
                  inventory_count=len(result['inventory']),
                  exclusions={d: sorted(exclusions[d]) for d in DATASETS},
                  exclusion_counts={d: len(exclusions[d]) for d in DATASETS})
    return result


def assert_fresh(dataset, batches, historical, available):
    if dataset not in DATASETS:
        raise ValueError('invalid dataset')
    forbidden, allowed = set(integer_ids(historical)), set(integer_ids(available))
    seen = set()
    if len(batches) != 39:
        raise ValueError('39 row sets required')
    for batch in batches:
        ids = integer_ids(batch)
        if not ids or len(ids) != len(set(ids)) or set(ids) & (forbidden | seen) or not set(ids) <= allowed:
            raise ValueError('source firewall violation')
        seen.update(ids)
    return True
