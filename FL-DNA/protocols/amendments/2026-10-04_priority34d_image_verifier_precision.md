# P34D independent image verifier precision

Before new image targets or confirmatory runs: audit requires IMAGE_COMPLETE,
all scientific workers exited, and valid source/input/output hashes. Reload all
scheduled raw arrays and exact received receipts, verify target IDs against the
frozen manifest and immutable CIFAR test images/labels. Ground truth inversion
of the original float32 normalization compares rtol2e-7/atol2e-7. Independently
recompute Hungarian pairing and all per-image/batch PSNR/SSIM/MSE; metrics
compare rtol1e-6/atol1e-8 and pairing is exact. Mismatch is preserved; never
adjust tolerances based on outcomes or swap target/candidate/metric definitions.

Candidate mode must minimize stored observable objectives with frozen plain tie
rule; optimizer iterations4800/restart1/CPU/thread1 fixed. Recompute the single
client gradient at the same public checkpoint solely to audit received payload
construction (no training, attack or candidate-selection rerun). Raw derivative
and private noise realization remain inside audit boundary, not attacker API.
Recreate single/per-tensor DP receipts bit-exact with frozen private seeds; no
seeds or private gradient arrays are printed. Known-key v1/v2 construction must
match exactly. Private truth bundles remain outside DP release.

Verifier PASS is image-stage audit only, not P34D COMPLETE. Require independent
utility/BN/stats/manifest/report and no-live gates separately.
