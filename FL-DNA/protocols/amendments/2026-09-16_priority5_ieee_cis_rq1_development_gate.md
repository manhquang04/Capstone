# Amendment: Priority 5.3 IEEE-CIS RQ1 development gate

**Date:** 2026-09-16  
**Status:** FROZEN FOR DEVELOPMENT SCREENING

Use the already frozen IEEE-CIS transaction-only feature contract from Priority
5.1/5.2. Create a new IEEE-CIS development target at four records/group, one
fraud record/group, and one Adam local step. It must be source-disjoint from
all target artifacts and is development-only. The attacker may use only public
feature ranges/normalization and generic non-negativity/range constraints; it
must not use fraud labels as an oracle and must not invent PaySim balance
constraints.

First run only the raw attacker gate against Prior and Zero-update controls.
If that gate fails, stop Priority 5.3 and report that the present attacker has
not validated at IEEE-CIS's smallest scope. If it passes, conduct one
predeclared eight-record/one-fraud/one-step scope screen. If that also passes,
DP distortion calibration may begin in a separate amendment; no DNA-vs-DP
confirmatory study is authorized by this amendment.

No post-hoc target may be read except source-ID disjointness checks. Do not
alter `torch.set_num_threads(1)` or tune the attacker after an observed gate.
