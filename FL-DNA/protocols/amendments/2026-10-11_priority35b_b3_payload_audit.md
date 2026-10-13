# B3 additive payload audit — 2026-10-11 (post-launch infrastructure)

Written before inspecting protected-arm quality. This new read-only module
reconstructs the source rows/labels, checks initial public-model tensor bytes,
recomputes the loss gradient with a separately implemented focal/CE loss path,
and checks non-BN upload domains. It independently reproduces joint/per-tensor
clipping, serialization safety factor and seeded float64 Gaussian draws before
float32 serialization. Private OS seed receipts are audit-only, not estimator
inputs. It checks finite CPU tensors, nonnegative BN variance, exact payload
bytes, metadata, observable variant selection and 30-member budget receipts.

Transform application itself reuses its frozen implementation: this is an
exact replay/source-integrity check, not an independent transform algorithm.
No optimizer is rerun, no earlier result changes, and no gate, target, budget,
threshold, Holm family or interpretation changes. Synthetic tests compare the
independent gradient/DP computations against the frozen driver and reject
corruption. This module is not a full finalizer: metric/control/objective audits,
statistical serialization, figure QA, report append and full seals remain.
