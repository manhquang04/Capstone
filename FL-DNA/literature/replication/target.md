# Frozen replication target

This is an independent reimplementation, not a rerun of
`experiments/priority30_native_defenses/`.

Claim measured: on the first five S0u confirmatory target IDs, reconstruct
each known-label, batch-one CIFAR-10 example using the official Inverting
Gradients `GradientReconstructor` configuration after four observations:
undefended gradients; Soteria with the classifier-weight gradient omitted in
E2; 70% per-tensor gradient pruning with a retained-coordinate loss; and a
known-key DNA-v2-style SRHT sketch with sketch-space loss.

Data and conditions: CIFAR-10 test examples identified by the five immutable
S0u records: `(target ID, CIFAR index, label)` = `(0,2310,6)`, `(1,2290,9)`,
`(2,241,1)`, `(3,5402,4)`, `(4,8641,7)`. Model is official
`construct_model('LeNetZhu', seed=42)`, untrained. The attacker has the true
label. The model runs in evaluation mode.

Optimizer target: cosine reconstruction cost, signed Adam, learning rate 0.1,
4,800 iterations, one restart, total variation 0.01, boxed projection,
learning-rate decay, and loss restart score. The frozen reconstruction seed is
`30600 + target_id`, selected before execution to make all four conditions use
the same initial image per target. Two CPU threads are the maximum.

Metrics: raw `[0,1]` MSE, PSNR `-10 log10(MSE)`, and channel-aware SSIM with
`data_range=1`. Comparison targets are the stored S0u undefended values and
available S1c E2 values. A PSNR absolute difference greater than 1 dB is
flagged; direction means whether PSNR is better or worse than the S0u
undefended reconstruction for the same target. S1c lacks all requested
condition/target combinations, which will be reported as unavailable rather
than filled.

Canonical dependency path: pinned local official Inverting Gradients checkout
under `external_defenses/invertinggradients`; interpreter
`external_defenses/.venv/bin/python`. No Priority 30 experiment module is
imported or called.
