# Workstream B: theory audit of the implemented transforms

**Scope and grading conditions.**  The statements below concern one transmitted
tensor/update and the actual implementations in
`dna_encoder/transform_defense.py` and
`dna_encoder/transform_defense_v2.py`, with the attacker's observation stated
in each result.  They do **not** establish input reconstruction for arbitrary
models, datasets, optimizers, rounds, or server views.  In particular, the
Priority 24--28 evidence is a first-BN-running-mean, PaySim T1 channel where
applicable; it is not evidence for a gradient-only attacker.

## Deliverables / result status

| Item | Precise result | Status |
|---|---|---|
| B1 | Exact known-linear-map and ignore-map reconstruction identities and bounds | proved, conditional on the stated linear observation model |
| B2 | Necessary conditions for a meaningful Gaussian comparator; exact conditional MSE matching equation | proved; no claim that the current v1 or v2 experiment is automatically Gaussian-matched |
| B3 | Counterexamples to universal secrecy/privacy claims from a server key; implementation-specific key/linearity limitations | proved counterexamples; no computational-security proof |
| B4 | Exact fresh-state Adam first-step formula and consequences for gradient information | proved; any general inversion/privacy conclusion is explicitly unproved |

Notation: `\|\cdot\|` is Euclidean norm, `L^\dagger` is the
Moore--Penrose pseudoinverse, `P_{\ker L}` is the orthogonal projector onto
`\ker L`, and `\sigma_+(L)` is the least nonzero singular value of nonzero
`L`.  Error/MSE may be divided by the ambient input dimension if a per-coordinate
MSE convention is wanted.

## B1. Attacker error when a linear map is known, versus ignored

### Proposition B1.1 (known map)

Let `x \in \mathbb R^d` and suppose the received observation is

\[
y=Lx+e,\qquad L\in\mathbb R^{m\times d},
\]

where the attacker knows `L`.  The minimum-norm least-squares estimator
`\hat x_L=L^\dagger y` obeys the **identity**

\[
\hat x_L-x=-P_{\ker L}x+L^\dagger e,\qquad
\|\hat x_L-x\|^2=\|P_{\ker L}x\|^2+\|L^\dagger e\|^2. \tag{B1}
\]

Consequently,

\[
\|P_{\ker L}x\|^2
\le \|\hat x_L-x\|^2
\le \|P_{\ker L}x\|^2+\frac{\|e\|^2}{\sigma_+(L)^2}. \tag{B2}
\]

If `e` is mean-zero conditional on `x`, with covariance `\Sigma_e`, then

\[
\mathbb E[\|\hat x_L-x\|^2\mid x]
=\|P_{\ker L}x\|^2+
\operatorname{tr}(L^\dagger\Sigma_eL^{\dagger T}). \tag{B3}
\]

**Proof.**  The pseudoinverse identities give
`L^\dagger L=P_{(\ker L)^\perp}`.  Hence
`L^\dagger(Lx+e)-x=-P_{\ker L}x+L^\dagger e`.  The two terms are orthogonal
because the range of `L^\dagger` is `(\ker L)^\perp`; Pythagoras proves
(B1).  The operator norm of `L^\dagger` is `1/\sigma_+(L)`, proving (B2).
Taking the conditional expectation, with the cross term zero, proves (B3).

This is a bound on recovery of the transmitted **update** `x`, not on recovery
of a client record.  A record-level result additionally requires a specified,
validated inverse problem from records to updates.

### Proposition B1.2 (ignoring a same-dimensional transform)

If `m=d` and the attacker instead reports `\hat x_I=y` (i.e., treats the
release as untransformed), then exactly

\[
\hat x_I-x=(L-I)x+e,\qquad
\big|\|(L-I)x\|-\|e\|\big|\le\|\hat x_I-x\|
\le\|(L-I)x\|+\|e\|. \tag{B4}
\]

There is no distribution-free ordering between (B1) and (B4): for invertible
`L` and `e=0`, the known-map error is zero; but if `L=I`, both are the same.
For rectangular sketches, the phrase “ignore the transform” must specify a
lift `A:\mathbb R^m\to\mathbb R^d`; its exact error is
`\|(AL-I)x+Ae\|^2`.  Comparing `L^\dagger` to an arbitrary transpose-lift is
therefore not a valid generic attacker comparison.

### Application to v2

Ignoring padding and quantization for clarity, v2 sends `s=Rx`, where

\[
R=\sqrt{n/k}\,PHD,\quad n=\texttt{padded\_size},\quad
k=\texttt{sketch\_size}.
\]

This is exactly the code's normalized FWHT, sign diagonal and sampled rows.
It has `\operatorname{rank}(R)\le k`; if `k<d` it is rank deficient.  For no
padding (`n=d`), its rows satisfy `RR^T=(n/k)I_k`, and so
`R^\dagger=(k/n)R^T`.  Thus the least-squares error is

\[
\|P_{\ker R}x\|^2+(k/n)\|e\|^2. \tag{B5}
\]

The implementation's `reconstruct_update_array_v2` uses `R^Tq`, not the
least-squares lift `(k/n)R^Tq`; it is a server update rule, not automatically
the optimal known-map attacker.  With padding, use (B1)--(B3) for the actual
restricted matrix `R E\in\mathbb R^{k\times d}`, where `E` zero-pads the
original vector; the simple row-orthogonality simplification need not survive
restriction.

For stochastic rounding, each quantization error coordinate has conditional
mean zero and magnitude at most `\delta/2`, hence
`\mathbb E\|e\|^2\le k\delta^2/4`.  In the no-padding case this gives the
quantization contribution `\mathbb E\|R^\dagger e\|^2\le k^2\delta^2/(4n)`.
The implementation sets `\delta=\eta\|x\|/\sqrt d`, so this noise is
heteroscedastic and signal-dependent.

**v1 limitation (mandatory).**  No result above may be applied to actual v1
by calling it an independent keyed linear map.  In v1, every block seed is a
deterministic function of the block's little-endian float32 bytes
(`_dna_block_seed_from_float32_block`, lines 110--142), and the selected
permutation, quantile threshold, and attenuation mask consequently depend on
the unknown input.  The resulting function is piecewise/discontinuously
data-dependent and generally nonlinear.  A fixed-seed linear-map analysis is
therefore a false model of v1.

## B2. Conditions for a utility-matched Gaussian comparison

### Proposition B2.1 (what an exact distortion match requires)

Let a Gaussian comparator release `z=x+w`, where conditional on `x`,
`w\sim N(0,\tau^2 I_d)`, and let the transform release be decoded by a fixed
estimator `a(y)`.  Exact conditional squared-error matching at this `x` is

\[
d\tau^2=\mathbb E[\|a(Y)-x\|^2\mid x]. \tag{B6}
\]

If `a(Y)=L^\dagger(Lx+e)` under B1's assumptions, the right side is (B3),
which includes irrecoverable null-space bias.  Thus a scalar Gaussian variance
can match it at an individual `x` by choosing the right side divided by `d`,
but a *single* `\tau` matches all updates only if that conditional MSE is
constant on the declared population (or if the estimand is explicitly the
same population average).  This proves that “same nominal noise multiplier”
is not a distortion match.

For utility matching, there is no theorem equating (B6) to test utility.
A defensible empirical comparison requires, at minimum:

1. same model initialization, clients/batches, rounds, aggregation weights,
   clipping location/norm, and transmitted/server-visible tensor(s);
2. the same stochasticity protocol and a pre-specified utility endpoint;
3. a Gaussian grid that **brackets** the transform's utility on independent
   runs (rather than selecting only the least damaging Gaussian point);
4. attack access and knowledge that match each branch's actual server view;
5. an explicit release/composition accounting if the comparator is described
   as DP.

The Priority 25b grid did bracket its declared F1 losses: v1 lay between
Gaussian `\sigma=10^{-5}` (mean `\Delta F1=+0.004805`) and `3\times10^{-5}`
(`-0.005682`), and v2 between `3\times10^{-5}` (`-0.005682`) and `10^{-4}`
(`-0.039918`).  This supports a *protocol-specific empirical utility match*,
not equality of distribution, privacy, or attacker MSE.  Priority 26's v2-SB
utility endpoint did not establish both frozen non-inferiority gates (AUC-ROC
CI lower end `-0.006063 < -0.005`), so it should not be called utility-matched
under that frozen criterion.

**Why v1/v2 are not Gaussian mechanisms.**  v1 is data-dependent as stated
above.  v2 has projection bias plus stochastic rounding with
`\delta\propto\|x\|`; even conditional rounding is not generally additive
isotropic Gaussian noise.  A Gaussian comparison is therefore valid only as a
matched comparator defined by the above protocol, not as an identity between
mechanisms.

## B3. Server key, impossibility statements, and counterexamples

### Proposition B3.1 (key possession defeats any claim based only on hiding a known invertible map)

If a server learns a key sufficient to instantiate an invertible deterministic
map `T_K`, and observes noiseless `y=T_K(x)`, it recovers `x=T_K^{-1}(y)`.
Hence key secrecy alone cannot imply a privacy guarantee after key disclosure.
This is immediate by substitution.

### Proposition B3.2 (rank deficiency alone gives no universal privacy lower bound)

Let `L` be any nonzero rank-deficient linear map.  There is a one-dimensional
prior support `S=\{ae_j:a\in\mathbb R\}` on which recovery is exact whenever
`Le_j\ne0`: from `y=L(ae_j)`, choose any row vector `r^T` with
`r^TLe_j=1` and report `\hat a=r^Ty`.  More generally, if a prior support
subspace `S` has `\ker L\cap S=\{0\}`, the restriction `L|_S` is injective.

Therefore a rank-deficient sketch is not, by itself, an information-theoretic
or record-level privacy guarantee.  It can lose update components, as B1
quantifies, while still exactly revealing a structured sensitive attribute or
candidate set.  A finite candidate prior gives an even simpler counterexample:
if all `Lx_i` are distinct and there is no noise, nearest matching identifies
the candidate exactly.

### Implementation audit and status

* v1's base seed is supplied by the caller, but its actual block seed also
  depends on the unknown input bytes and is reduced modulo `2^{31}`.  It is
  neither an independent secret linear map nor a standard encryption scheme.
  No information-theoretic, cryptographic, or DP guarantee is proved here.
* Actual v2 metadata explicitly contains `seed` and `sampled_indices`; the
  derived seed and its RNG are 32-bit (`_derive_seed`, lines 229--239).  Thus
  the ordinary v2 implementation does not support a claim that the server is
  blind to its sketch map merely because a configuration seed was intended as
  a key.  The Priority 26 audit separately measured a bounded 32-bit seed
  space.  This is an implementation fact, not a proof that exhaustive search
  always succeeds for an arbitrary attack criterion.
* v2-SB is a distinct implementation and its server-key claim needs its own
  threat model: the report requires that `K` remains secret from the server,
  no server--client collusion occurs, and a fresh key is used each round.
  Even under these conditions, Proposition B3.2 rules out promoting rank
  deficiency alone to a universal privacy theorem.

**Explicit unproved flag.**  Computational hardness against a server that does
not know a v2-SB key is **unproved** here.  It cannot be inferred from a
failed/ungated empirical attacker or from the rank bound alone.

## B4. Adam first-step information

### Proposition B4.1 (fresh-state Adam is a near-sign map)

For Adam initialized with `m_0=v_0=0`, bias corrections enabled, gradient `g`,
learning rate `\alpha`, and stabilizer `\varepsilon\ge0`, its first parameter
step is coordinatewise

\[
\Delta\theta_{1,i}=-\alpha\frac{g_i}{|g_i|+\varepsilon}. \tag{B7}
\]

For `\varepsilon=0` and `g_i\ne0`, this is exactly
`-\alpha\operatorname{sign}(g_i)`.  **Proof.**  At the first step,
`\hat m_1=g` and `\hat v_1=g\odot g`; substitute them in Adam's update.

Thus with `\varepsilon=0` the map is non-injective on each same-sign ray: any
two positive magnitudes yield the same coordinate step.  With
`\varepsilon>0`, the scalar magnitude map `r\mapsto\alpha r/(r+\varepsilon)`
is formally invertible below `\alpha`, but saturates: its derivative is
`\alpha\varepsilon/(r+\varepsilon)^2`.  Consequently exact noiseless,
infinite-precision observations can retain magnitude information when
`\varepsilon>0`; it is not correct to claim a strict magnitude-erasure theorem
for ordinary Adam.

For later steps, `(m_t,v_t)` retain optimizer history, so (B7) does not apply
without conditioning on that state.  A transmitted model delta can also mix
multiple local steps and other tensors.  No theorem here converts the
first-step sign property into a record-reconstruction lower bound.

### Literature resolution and empirical consistency

Balles and Hennig, ICML 2018, explicitly interpret Adam as a sign direction
times variance adaptation: ignoring `\varepsilon`,
`m_t/\sqrt{v_t}=\operatorname{sign}(m_t)/\sqrt{1+(v_t-m_t^2)/m_t^2}`.
Their claim is an interpretation under moment estimates, not a privacy proof.
Bernstein et al., ICLR 2019, prove signSGD results only under lower boundedness,
coordinatewise smoothness/variance, and unimodal symmetric gradient-noise
assumptions; they also give a bimodal counterexample where stochastic gradient
sign is wrong with probability 0.9 despite positive mean.  Those optimization
results do not imply reconstruction resistance.

The project's fresh-state diagnostic is consistent with (B7): Priority 25
reported mean cosine `0.9920503` between step and `-\alpha\operatorname{sign}(g)`,
unit sign agreement on nonzero coordinates, and magnitude-correlation `0.0473`
(eight targets).  Priority 27 likewise found cosines about `0.9915--0.9921`.
These are measured diagnostics, not a proof beyond their runs; the associated
gradient-only attacks failed their n=8 qualification gates, so a general
gradient-only privacy conclusion remains **unproved**.

## Exact sources and implementation evidence used

1. L. Balles and P. Hennig, “Dissecting Adam: The Sign, Magnitude and Variance
   of Stochastic Gradients,” *Proceedings of the 35th ICML*, PMLR 80,
   2018, pp. 404--413.  Verified full text: https://arxiv.org/abs/1705.07774
   (especially Eqs. 4--9, 10, and the stochastic-quadratic discussion).
2. J. Bernstein, J. Zhao, K. Azizzadenesheli, and A. Anandkumar,
   “signSGD with Majority Vote is Communication Efficient and Fault Tolerant,”
   *ICLR*, 2019.  Verified full text: https://arxiv.org/abs/1810.05291
   (Assumptions 1--4, Theorem 1, Lemma 1 and its bimodal counterexample).
3. Implementation read directly: `dna_encoder/transform_defense.py`, lines
   36--92 and 110--142; `dna_encoder/transform_defense_v2.py`, lines 61--136,
   172--211, and 225--286.
4. Project evidence read directly: `reports/priority24_rq1_valid_instrument_report_20260929.md`,
   `reports/priority25_rq1_extension_report_20260929.md`,
   `reports/priority25b_bracketed_utility_report_20260929.md`,
   `reports/priority26_v2sb_server_blind_report_20260929.md`,
   `reports/priority27_gaps_and_harness_report_20260930.md`, and
   `reports/priority28_defense_audit_report_20260930.md`.

## Verification

Run from repository root:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('literature/theory/theory_report.md')
s = p.read_text(encoding='utf-8')
assert all(x in s for x in ('## B1.', '## B2.', '## B3.', '## B4.',
                            'v1 limitation', 'unproved', 'Balles and Hennig',
                            'Bernstein et al.'))
print(f'{p}: {len(s.splitlines())} lines; required B1--B4/source flags present')
PY
```

This is a structural check of the delivered report; the proofs are algebraic
and were not replaced by a numerical simulation.
