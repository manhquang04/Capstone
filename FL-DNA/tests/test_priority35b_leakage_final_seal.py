import unittest
from experiments import priority35b_leakage_final_seal as s


class Tests(unittest.TestCase):
    def test_requires_full_append_only_publication(self):
        doc=dict(status='REPORT_APPENDED_PENDING_POSTEXIT_FULL_SEAL',measurement_repeated=False,
                 full_priority35b_complete=False,checks={k:True for k in s.contract.guard.REQUIRED})
        s.require_published(doc)
        for key,value in (('status','PARTIAL'),('measurement_repeated',True),('full_priority35b_complete',True)):
            bad=dict(doc);bad[key]=value
            with self.assertRaises(ValueError):s.require_published(bad)

    def test_missing_independent_check_cannot_be_sealed(self):
        doc=dict(status='REPORT_APPENDED_PENDING_POSTEXIT_FULL_SEAL',measurement_repeated=False,
                 full_priority35b_complete=False,checks={k:True for k in s.contract.guard.REQUIRED})
        doc['checks']['figure_visual_qa']=False
        with self.assertRaises(ValueError):s.require_published(doc)


if __name__=='__main__':unittest.main()
