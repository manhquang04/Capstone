# P34D direction and synthetic initialization preflight — BEFORE any science

User chose option1: batch4 is the recovery batch, not the training minibatch.
Part2 utility training retains P31/33C SGD.1/momentum0,batch256,100epochs.
Utility matching must correspond to the fixed trained checkpoint and unchanged
procedure. Reuse only if provenance proves exact checkpoint/procedure equality;
otherwise NEW same-protocol calibration, never old sigma transfer by assumption.
Existing P33C calibration initializes LeNet-Zhu seed42 from scratch; its C2
trained sensitivity explicitly transfers those sigmas rather than calibrates
from the trained checkpoint. Therefore exact warm-start equality is NOT proved;
new calibration is required for Part2. No change to batch size or epoch budget.

Before launch, rechecked local P34B COMPLETE PASS264, zero failures,
1254bitexact arrays,39600noBN assertions,78paired endpoints and report digest
60c5b2c430c12a25852c6b600f74b7feb78b9c4f0a030ce67b890ab79cfab4bc.
Goal remains paused; no silent resume or scientific rerun of P34B. P34C report
also verified complete. No P34D research job has run at this amendment.

Synthetic preflight failed on duplicate torch.set_num_interop_threads(1), since
the imported P32 runner already sets it. Preserve failure receipt. Change only
the NEW P34D initialization to call this setter if current value is not1;
assert actual inter/intra threads1. No RNG, data, target, optimizer or budget
change. Repeat synthetic tests and compile/diff BEFORE source freeze/launch.
