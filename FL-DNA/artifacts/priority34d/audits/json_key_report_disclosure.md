## Infrastructure-only preflight amendment

The initial recovery preparation failed before target creation or attack launch
because calibration grid dictionary keys were numeric in memory and strings
after JSON serialization. The failed PREFLIGHT_FAILURE.json and existing logs
were retained unchanged. After explicit user approval, an additive preparation
adapter normalized JSON keys on both sides and used exact equality, with no
tolerance change. Six unit tests covered integer/string and float/string key
equivalence, genuine value differences, no tolerance, value preservation and
ambiguous-key rejection; five reservation audit tests also passed.

The rerun confirmed16/16 calibration cells matched. No calibration value, sigma,
target list, seed or original frozen scientific source was changed. The273
prespecified reservation IDs matched the independent historical audit. No
training or recovery was replayed. The original frozen recovery launcher and
worker were then used for1092 scheduled jobs. The amendment and adapter were
sealed into the new recovery execution freeze, together with the retained
failure and repair receipt. This infrastructure repair is not a scientific
result or a completion claim.

Recovery freeze SHA256:
ab82ab050520d0cf52009e4fc362434f5865536c98be04b51e6b7e6a855cd350.
Repair receipt SHA256:
2f71a90de5f88f09d5c514220b6590bc1c09d5f4632b081c68a47951e46dcf51.
Include this disclosure in the final P34D report; final independent audits are
still required before verified COMPLETE.
