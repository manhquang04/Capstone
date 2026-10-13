# P35B B3 additive statistics verification seal

Date2026-10-11, Asia/Ho_Chi_Minh. Post-launch infrastructure note, before
protected submissions or data analysis. No scientific rule or measurement
source changed; the original execution/source-input freeze is preserved.
The two new files below implement the existing fixed144 family only, with
explicit NA p1 reservations and no missing-output substitution. No measured
quality data were read to choose any test or implementation parameter.

- experiments/priority35b_leakage_statistics.py:
  SHA598b6bb3a4675d6749bc3c69621eeb6ed03c6f47547638a2b54e9dd8f2c25033
- tests/test_priority35b_leakage_statistics.py:
  SHA869d8c701ea988cdc8099a8d985179e8f59c4022c546c85dd087cb8f86870d15

Two synthetic tests PASS, including separate SciPy binomtest and NumPy Holm
recomputation, order-statistic ranks13/27 and rejection of missing pairs.
py_compile and git diff --check PASS. These are analysis primitives, not the
full post-exit payload/metric/firewall verifier, figure or final report.
Those deliverables remain pending and must receive their own source/test
seal before confirmatory analysis; no nonexistent --finish is authorized.
