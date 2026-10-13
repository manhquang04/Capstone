import base64
from collections import OrderedDict
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dna_encoder import transform_defense_v2 as current
from dna_encoder import transform_defense_v2_hardened as hard
from experiments import run_rq3_benchmark as rq3
from experiments.priority36a_wire import serialize_v2_hardened, deserialize_v2_hardened

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def load_reference(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "artifacts/priority36a" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_reference("priority36a_legacy_transform", "legacy_transform.py")
old_wire = load_reference("priority36a_legacy_rq3", "legacy_rq3.py")
for name in ("DNATransformV2Config", "reconstruct_update_array_v2", "transform_and_reconstruct_array_v2", "transform_update_array_v2"):
    setattr(old_wire, name, getattr(legacy, name))

KEY = bytes.fromhex("00112233445566778899aabbccddeeff")  # public test fixture only


class HardeningTests(unittest.TestCase):
    def setUp(self):
        self.config = current.DNATransformV2Config(hardened=True)
        self.array = np.linspace(-3, 4, 221, dtype=np.float32).reshape(13, 17)

    def test_default_reference_bytes_metadata(self):
        for shape in [(), (0,), (1,), (13, 17), (128, 58), (128, 476), (128, 13)]:
            value = np.arange(np.prod(shape, dtype=int), dtype=np.float32).reshape(shape) / 97
            for seed in [0, 321000, 2**40 + 7]:
                for eta in [0, .01]:
                    cfg = current.DNATransformV2Config(.95, eta, seed)
                    old = legacy.transform_and_reconstruct_array_v2(value, legacy.DNATransformV2Config(.95, eta, seed), 3, 17)
                    new = current.transform_and_reconstruct_array_v2(value, cfg, 3, 17)
                    for a, b in zip(old[:2], new[:2]):
                        self.assertEqual(a.tobytes(), b.tobytes())
                    self.assertEqual(asdict(old[2]), asdict(new[2]))
                    self.assertEqual(asdict(old[3]), asdict(new[3]))

    def test_default_wire_reference(self):
        tensors = OrderedDict(weight=self.array, bias=np.arange(17, dtype=np.float32))
        old, expected = old_wire._serialize_v2(tensors, legacy.DNATransformV2Config(seed=321000), 4)
        new, recovered = rq3._serialize_v2(tensors, current.DNATransformV2Config(seed=321000), 4)
        self.assertEqual(old, new)
        for name in tensors:
            self.assertEqual(expected[name].tobytes(), recovered[name].tobytes())
            self.assertEqual(old_wire._deserialize_v2(old)[name].tobytes(), rq3._deserialize_v2(new)[name].tobytes())

    def test_generation(self):
        with patch.object(hard.secrets, "token_bytes", return_value=KEY) as draw:
            self.assertEqual(hard.generate_v2_key(), KEY)
            draw.assert_called_once_with(16)
        self.assertEqual(len(hard.generate_v2_key()), 16)

    def test_rfc5869_vector(self):
        out = hard._hkdf(bytes.fromhex("0b" * 22), bytes.fromhex("000102030405060708090a0b0c"), bytes.fromhex("f0f1f2f3f4f5f6f7f8f9"), 42)
        self.assertEqual(out.hex(), "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865")

    def test_full_subkeys_domains(self):
        values = {hard._subkey(KEY, i, purpose, context) for i in [0, 1] for purpose in [b"signs", b"sampling", b"quantization"] for context in [0, 1]}
        self.assertEqual(len(values), 12)
        self.assertTrue(all(len(k) == 32 for k in values))
        altered = bytes([KEY[0] ^ 128]) + KEY[1:]
        self.assertNotEqual(hard._subkey(KEY, 0, b"signs"), hard._subkey(altered, 0, b"signs"))

    def test_roundtrip_exact(self):
        for value in [self.array, np.array(1, np.float32), np.empty((0,), np.float32), np.zeros((4, 5), np.float32)]:
            q, expected, meta, _ = current.transform_and_reconstruct_array_v2(value, self.config, key=KEY)
            recovered, _ = current.reconstruct_update_array_v2(q, meta, key=KEY)
            self.assertEqual(expected.tobytes(), recovered.tobytes())
            self.assertEqual(value.shape, recovered.shape)

    def test_dense_independent_reconstruction(self):
        value = np.arange(7, dtype=np.float32)
        q, meta = current.transform_update_array_v2(value, self.config, key=KEY)
        signs, sampled = hard._projection(meta.padded_size, meta.sketch_size, KEY, 0)
        n = meta.padded_size
        matrix = np.array([[(-1.0)**bin(i & j).count("1") / np.sqrt(n) for j in range(n)] for i in range(n)])
        expected = (signs * (matrix.T[:, sampled] @ (np.sqrt(n / len(sampled)) * q.astype(np.float64))))[:7].astype(np.float32)
        recovered, _ = current.reconstruct_update_array_v2(q, meta, key=KEY)
        np.testing.assert_allclose(expected, recovered, atol=1e-6, rtol=1e-6)

    def test_missing_invalid_key(self):
        for key in [None, 123, b"a" * 15, b"a" * 17]:
            with self.assertRaises(ValueError):
                current.transform_update_array_v2(self.array, self.config, key=key)
        q, meta = current.transform_update_array_v2(self.array, self.config, key=KEY)
        with self.assertRaises(ValueError):
            current.reconstruct_update_array_v2(q, meta)
        with self.assertRaises(ValueError):
            deserialize_v2_hardened(b"{}")

    def test_different_keys(self):
        q1, _ = current.transform_update_array_v2(self.array, self.config, key=KEY)
        q2, _ = current.transform_update_array_v2(self.array, self.config, key=bytes(reversed(KEY)))
        self.assertNotEqual(q1.tobytes(), q2.tobytes())

    def test_public_only_wire_and_metadata(self):
        tensors = OrderedDict(weight=self.array)
        payload, expected = serialize_v2_hardened(tensors, self.config, key=KEY)
        decoded = deserialize_v2_hardened(payload, key=KEY)
        self.assertEqual(decoded["weight"].tobytes(), expected["weight"].tobytes())
        for token in [b'"seed"', b'"key"', b'"sampled_indices"', b'"derived_seed"', KEY, KEY.hex().encode(), base64.b64encode(KEY)]:
            self.assertNotIn(token, payload)
        for purpose in [b"signs", b"sampling", b"quantization"]:
            secret = hard._subkey(KEY, 0, purpose)
            for token in [secret, secret.hex().encode(), base64.b64encode(secret)]:
                self.assertNotIn(token, payload)
        _, meta = current.transform_update_array_v2(self.array, self.config, key=KEY)
        self.assertEqual(set(asdict(meta)), set(hard.HardenedV2Metadata.__dataclass_fields__))
        for token in ["seed", "key", "indices"]:
            self.assertNotIn(token, json.dumps(asdict(meta)))

    def test_fail_closed_finite(self):
        for bad in [np.nan, np.inf, -np.inf]:
            with self.assertRaises(ValueError):
                current.transform_update_array_v2(np.array([bad]), self.config, key=KEY)
        q, meta = current.transform_update_array_v2(self.array, self.config, key=KEY)
        q[0] = np.nan
        with self.assertRaises(ValueError):
            current.reconstruct_update_array_v2(q, meta, key=KEY)
        with self.assertRaises(ValueError):
            current.transform_update_array_v2(self.array, replace(self.config, quantization_eta=np.inf), key=KEY)

    def test_strict_wire(self):
        payload, _ = serialize_v2_hardened(OrderedDict(weight=self.array), self.config, key=KEY)
        for field in ["seed", "key", "sampled_indices"]:
            doc = json.loads(payload)
            doc["tensors"][0]["metadata"][field] = 123
            with self.assertRaises(ValueError):
                deserialize_v2_hardened(json.dumps(doc), key=KEY)
        with self.assertRaises(ValueError):
            deserialize_v2_hardened(b'{"format":"a","format":"b","tensors":[]}', key=KEY)

    def test_context_and_sampling(self):
        _, indices = hard._projection(256, 243, KEY, 1)
        self.assertEqual(len(set(indices)), 243)
        self.assertTrue(np.all(np.diff(indices) > 0))
        a, _ = current.transform_update_array_v2(self.array, self.config, 1, 1, key=KEY)
        b, _ = current.transform_update_array_v2(self.array, self.config, 1, 2, key=KEY)
        self.assertNotEqual(a.tobytes(), b.tobytes())

    def test_legacy_seed_rejected_in_hardened(self):
        with self.assertRaises(ValueError):
            current.transform_update_array_v2(self.array, replace(self.config, seed=7), key=KEY)
        with self.assertRaises(ValueError):
            current.transform_update_array_v2(self.array, current.DNATransformV2Config(seed=7), key=KEY)


if __name__ == "__main__":
    unittest.main()
