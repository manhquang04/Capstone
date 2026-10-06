# P34B preflight annex — before experimental execution

The first unit-test import failed because the tabular project loads its
`models` package before P30's image dependency chain expects TabLeak's generic
`models.MetaMonkey`. No scientific dataset training or target creation occurred.
New priority34b_image_imports.py temporarily isolates generic external package
names and restores the original project namespace/search path after importing
the immutable image references. No external_defenses/ or prior file was edited.
All eleven tests then passed. Tests/freeze receipt retain this preflight failure.
The DP clipping/server averaging computes in float64 before casting outputs to
float32; inward clipping rounding protects the declared L2 bound. P34A's
float32 aggregation baseline is reused unchanged; this small arithmetic change
is part of the new DP implementation, not a replay or repair of old evidence.
The frozen scientific settings, n, seeds, recovery budget and accounting remain
those in the main pre-run amendment.

Privacy randomness audit before freeze: public training seeds must not determine
server Gaussian noise. Independent OS-entropy keys per DP job are frozen in
private_noise_seeds.json (mode0600), retained privately for exact resume; only
their file checksum is referenced publicly. The main protocol's344200+i noise
seed is retained solely as a descriptive job namespace, NOT the RNG key used.
Torch's secret-seeded PRNG simulates the ideal Gaussian mechanism; this is not
a certified finite-precision/CSPRNG implementation. Keys, raw local states,
targets and all private audit files must not be published as DP outputs.
Attacker closures receive only the noisy payload, never the secret key or noise
realization. This prevents deterministic publicly predictable noise, without
changing any epsilon, clipping, training seed or reconstruction budget.

Administrative lifecycle: the science report is generated before supervisor
exit, then post-exit verification preserves it byte-identically in reports/archive/
before updating completion wording and appending audit details at the requested
report path. Original and final manifest/checksums are kept separately. No
scientific table/result is overwritten or reselected by this bookkeeping.
