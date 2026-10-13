import unittest
from experiments import priority35b_leakage_publication_contract as p


class Tests(unittest.TestCase):
    def fixtures(self):
        assets={k:'a'*64 for k in p.ASSETS}
        rendered=dict(status='RENDERED_AWAITING_VISUAL_QA',measurement_repeated=False,
            full_priority35b_complete=False,outputs=assets)
        qa=dict(status='VISUAL_QA_PASS',synthetic=False,at='synthetic fixture',reviewer='unit test',
            outputs=dict(assets),checks={k:True for k in p.VISUAL_CHECKS})
        return rendered,qa

    def test_qa_bound_to_measured_assets(self):
        r,q=self.fixtures();self.assertTrue(p.visual_qa(r,q))
        for key,value in (('synthetic',True),('reviewer',''),('status','PENDING')):
            bad=dict(q);bad[key]=value
            with self.assertRaises(ValueError):p.visual_qa(r,bad)
        q['outputs']=dict(q['outputs']);q['outputs'][next(iter(p.ASSETS))]='b'*64
        with self.assertRaises(ValueError):p.visual_qa(r,q)

    def test_missing_visual_check_and_completion_binding(self):
        r,q=self.fixtures();q['checks'].pop(next(iter(p.VISUAL_CHECKS)))
        with self.assertRaises(ValueError):p.visual_qa(r,q)
        receipt=dict(status='PRIORITY35B_VERIFIED_COMPLETE',report_sha256='a'*64,manifest_sha256='b'*64,
            no_live_workers=True,postexit_seal=True,measurement_repeated=False)
        self.assertTrue(p.final_receipt_checks(receipt,'a'*64,'b'*64))
        receipt['postexit_seal']=False
        with self.assertRaises(ValueError):p.final_receipt_checks(receipt,'a'*64,'b'*64)


if __name__=='__main__':unittest.main()
