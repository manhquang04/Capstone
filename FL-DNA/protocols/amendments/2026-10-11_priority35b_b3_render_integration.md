# Priority 35B B3: additive post-exit render integration

Date: 2026-10-11. Infrastructure implementation note written before production
analysis/rendering. No scientific rule, measurement, arm, budget, metric,
tolerance, selection rule, matching rule or Holm family changes.

The new render runner accepts only the stable preparation wrapper's receipt.
It requires the scientific processes and preparation wrapper PID to have exited
(PID reuse fails closed), checks the prepared input/output hashes and all 49,920
scientific evidence hashes, and recomputes CSV tests and the figure join before
rendering. Existing frozen sources remain unchanged. No measurement is replayed.

The frozen renderer exclusively creates PDF, PNG and exact data CSV under
artifacts/priority35b/figure/. Poppler renders the PDF to a separate PNG; the
calling agent must emit the PDF artifact-operation marker before invoking this
actual production command and inspect both images. Any failure is preserved and
requires human direction. Existing figure files cannot be overwritten.

Command, only after post-exit preparation succeeds and its process exits:

`.venv-phase1/bin/python -B -m experiments.priority35b_leakage_render_publication --ready <relative READY_FOR_RENDER.json path>`

Success means RENDERED_AWAITING_VISUAL_QA, not COMPLETE. Report append, measured
figure visual QA, tests/compile/diff checks and final post-exit manifest sealing
remain mandatory separate steps. Earlier report bytes are not touched here.
