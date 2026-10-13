"""Hardened RQ3 JSON wire path. Key supplied separately, never in payload."""
import base64
from collections import OrderedDict
from dataclasses import asdict
import json

import numpy as np

from dna_encoder.transform_defense_v2 import transform_and_reconstruct_array_v2, reconstruct_update_array_v2
from dna_encoder.transform_defense_v2_hardened import HardenedV2Metadata, _key

FORMAT = "dna-transform-v2-hardened-sketch-json-v1"


def serialize_v2_hardened(tensors, config, quantization_seed=0, *, key=None):
    _key(key)
    if not config.hardened:
        raise ValueError("hardened wire requires hardened=True")
    entries, expected = [], OrderedDict()
    for index, (name, value) in enumerate(tensors.items()):
        if not isinstance(name, str):
            raise ValueError("tensor names must be strings")
        sketch, recovered, metadata, _ = transform_and_reconstruct_array_v2(value, config, index, quantization_seed, key=key)
        entries.append({"name": name, "metadata": asdict(metadata),
                        "sketch_f32_base64": base64.b64encode(np.ascontiguousarray(sketch, dtype="<f4").tobytes()).decode("ascii")})
        expected[name] = recovered
    return json.dumps({"format": FORMAT, "tensors": entries}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), expected


def deserialize_v2_hardened(payload, *, key=None):
    _key(key)
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON field")
            out[k] = v
        return out
    document = json.loads(payload, object_pairs_hook=unique,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite JSON")))
    if set(document) != {"format", "tensors"} or document["format"] != FORMAT or not isinstance(document["tensors"], list):
        raise ValueError("unexpected hardened wire schema")
    output = OrderedDict()
    fields = set(HardenedV2Metadata.__dataclass_fields__)
    for index, item in enumerate(document["tensors"]):
        if set(item) != {"name", "metadata", "sketch_f32_base64"} or set(item["metadata"]) != fields:
            raise ValueError("unexpected tensor wire schema")
        name = item["name"]
        if not isinstance(name, str) or name in output:
            raise ValueError("duplicate or invalid tensor name")
        public = dict(item["metadata"])
        if public["tensor_index"] != index or type(public["tensor_index"]) is not int:
            raise ValueError("tensor index/order mismatch")
        for field in ("original_size", "padded_size", "sketch_size"):
            if type(public[field]) is not int:
                raise ValueError("invalid dimension type")
        if not isinstance(public["original_shape"], list):
            raise ValueError("invalid shape type")
        public["original_shape"] = tuple(public["original_shape"])
        metadata = HardenedV2Metadata(**public)
        raw = base64.b64decode(item["sketch_f32_base64"], validate=True)
        if len(raw) != metadata.sketch_size * 4:
            raise ValueError("sketch byte length mismatch")
        sketch = np.frombuffer(raw, dtype="<f4")
        output[name], _ = reconstruct_update_array_v2(sketch, metadata, key=key)
    return output
