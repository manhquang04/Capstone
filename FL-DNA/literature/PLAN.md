# Literature Work Plan

Scope guard: all existing project folders are read-only. New files are written only below `literature/`. No training or attack experiments will be run.

| Order | Workstream | Task | Estimate | Status |
|---:|---|---|---:|---|
| 1 | Shared context | Located manuscript/bibliography at `../Latex/`; read specified reports and transforms read-only. | 10 min | Completed |
| 2 | A: novelty | Searched/logged arXiv, OpenAlex, and Semantic-Scholar-targeted queries; read load-bearing papers; wrote quote-level evidence table and strict gap analysis. | 60 min | Completed |
| 3 | B: theory | Stated/proved conditional B1--B4 results or explicit counterexamples/unproved limits; no numerical experiment required. | 50 min | Completed |
| 4 | C: related work | Drafted 849-word IEEE-style LaTeX section; all 17 citation keys resolve against `../Latex/references.bib` plus `new_refs.bib`. | 35 min | Completed |
| 5 | D: citation audit | Audited all 52 bibliography entries; 45 OK, 5 mismatch, 2 unverified. | 45 min | Completed |
| 6 | E: datasets | Documented source/terms/preprocessing/ethics for PaySim, CIFAR-10, and Adult. | 25 min | Completed |
| 7 | Final | Performed structural/citation-key checks and wrote `literature/SUMMARY.md`. | 15 min | Completed |

Decision log: this is a scoping/novelty audit, not an exhaustive systematic review. The date window is 2020--2026. Any unavailable database result is logged as unavailable, never treated as no prior work. For theory, statements are limited to the exact stated transform/observation assumptions; no general impossibility claim will be made without a proof.
