# Literature Work Summary

## Completion status

All requested literature deliverables were written under `literature/` only. Existing repository files, datasets, experiments, artifacts, reports, and `external_defenses/` were treated as read-only. No training or inversion experiment was launched.

| Workstream | Deliverable | Key result |
|---|---|---|
| A: novelty | `novelty/novelty_report.md`, `novelty/novelty.bib`, `novelty/search_log.md` | The planned work is **not** first systematic/auditing/adaptive evaluation. Its defensible novelty is the four-check conjunction, native cross-modal reference controls, explicit key-holding-server analysis, and specified clipped DP comparison at both distortion and utility matches. |
| B: theory | `theory/theory_report.md` | Conditional update-space bounds are proved; general record-reconstruction/privacy claims are explicitly unproved. A server-known invertible key defeats secrecy based only on hiding the map; rank deficiency alone gives no universal privacy bound. |
| C: related work | `related_work/related_work.tex`, `related_work/new_refs.bib` | 849-word IEEE-style section with 17 resolvable citation keys. It positions the paper as a reproducible audit/case study, not first adaptive defense evaluation. |
| D: citation audit | `citations/citation_audit.md` | 52 entries: 45 OK, 5 mismatch, 2 unverified. Concrete page corrections include Dwork--Roth `211--487`, GradInversion `16332--16341`, and Soteria `9307--9315`. |
| E: datasets | `datasets/dataset_notes.md` | PaySim is documented as synthetic CC BY-SA 4.0; Adult as CC BY 4.0; CIFAR-10 explicit source license was not found and is marked unverified. |

## Novelty threat and recommended framing

The strongest novelty threat is prior work already showing the core methodological lessons. Huang et al. evaluate pruning and encoding defenses using strong assumptions; Balunovic et al. formalize defense-aware inference; Yue et al. attack obfuscation defenses adaptively; Li et al. audit several defenses; Hatamizadeh et al. show the importance of transmitted BN state; TabLeak demonstrates feature-level tabular scoring against a marginal baseline; and Du et al. systematize practical configurations and post-processing. Do **not** claim the first systematic defense re-evaluation, first adaptive attacker, first BN leakage result, first DP comparison, or first input-space metric.

Position the paper as a protocol-level, reproducible case study: selected defense claims are accepted only after (a) an undefended positive control passes, (b) the attacker is adapted to the exact server observation/knowledge, including keys, (c) input-space performance beats a data-free baseline, and (d) the stated-clipping DP comparator is evaluated at distortion- and utility-matched operating points. The results concern the named models, data, server views, and reference implementations; they are not a universal ranking of FL defenses.

## Theory highlights

- For a known linear observation `y=Lx+e`, the minimum-norm least-squares update estimator has exact error `-P_{ker L}x+L^\dagger e`; its error is controlled by null-space loss and observation noise, not by a generic “keyed transform” label.
- Actual v1 must not be analyzed as an independent secret linear map: its block seed depends on the input float32 bytes, and thresholding/masking are data-dependent.
- For fresh-state Adam, the first update is coordinatewise `-lr*g/(|g|+eps)`. It is sign-like but does not itself prove a record-reconstruction lower bound; with positive epsilon, noiseless magnitude information is formally invertible below saturation.
- The theory report gives counterexamples to a universal claim that rank deficiency or secret-key transforms prevent reconstruction. It does not claim a computational-security proof for the server-blind v2-SB variant.

## Unverified or limited items

- “LeakProf” could not be resolved to a unique scholarly work; it is marked **UNVERIFIED** in the novelty audit.
- Semantic Scholar Graph API returned HTTP 429 for the documented SoK metadata request; the arXiv paper was read, but the exact venue metadata remains reported from the manuscript bibliography/citation audit rather than that failed API call.
- CIFAR-10’s authoritative page did not supply an explicit license/reuse term.
- The bibliography audit leaves `jamshidi2024obfuscator` and `du2025sok` **UNVERIFIED** under its strict record-resolution rule; this means the audit could not obtain an exact live metadata record, not that the works are nonexistent. The SoK’s arXiv text was nevertheless read for the novelty evidence.
- The PaySim dataset note identifies a source warning against balance fields for fraud detection, while the project uses them. This is a validity/leakage risk requiring a no-balance-feature sensitivity analysis before strong fraud-performance claims.

## Suggested next actions

1. Revise manuscript novelty/contribution language to the three bullets in `novelty/novelty_report.md`; remove any “first systematic evaluation” language.
2. Correct the three unambiguous bibliography page ranges and resolve the remaining two metadata records before submission.
3. Insert `related_work/related_work.tex` only after reconciling any overlap with the current manuscript’s existing related-work section; add `related_work/new_refs.bib` to the build bibliography.
4. Treat `theory/theory_report.md` as a source of conditional propositions and counterexamples, not as a formal privacy proof for either transform.
5. Add the dataset license, synthetic-data, PaySim-balance-feature, and Adult sensitive-attribute limitations to the data/ethics statement.

## Final checks

The final structural check passed: all required report sections were present; `novelty.bib` contained the expected resolved identifiers and no `UNVERIFIED` entry; all 17 related-work citation keys resolve to the source plus supplemental bibliography; and the citation audit contained all 52 expected rows with status totals 45/5/2.
