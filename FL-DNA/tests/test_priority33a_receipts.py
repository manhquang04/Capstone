import unittest
import torch
from experiments.repair_priority33a_receipts import server_receipt, recover_server_payload


class ReceiptContract(unittest.TestCase):
    def test_v1_diagnostics_not_server_observable(self):
        q = torch.ones(128)
        receipt = server_receipt(dict(kind="v1", q=q, metadata={"cosine_with_raw": .7}))
        self.assertEqual(set(receipt), {"kind", "q"})
        self.assertIs(receipt["q"], q)
        with self.assertRaises(AssertionError):
            recover_server_payload(None, dict(kind="v1", q=q, metadata={"raw_diagnostics": True}))

    def test_unknown_mode_fails_closed(self):
        with self.assertRaises(ValueError):
            server_receipt(dict(kind="unknown"))
        with self.assertRaises(ValueError):
            recover_server_payload(None, dict(kind="unknown"))


if __name__ == "__main__":
    unittest.main()
