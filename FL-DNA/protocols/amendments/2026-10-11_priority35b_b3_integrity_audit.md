# Priority 35B B3 — additive infrastructure audit (2026-10-11, post-launch)

This infrastructure note is written while measurements are running, before any
analysis of incomplete protected-arm quality. It does not change measurements,
targets, mechanisms, budgets, gates, matching rules, Holm-144, or interpretations.

The new `priority35b_leakage_integrity.py` is a read-only evidence-integrity
building block. It requires all 9,984 outputs and an external no-live audit
covering the supervisor and every recorded worker PID. It checks the original
execution-freeze digest, every frozen source/input hash, exact three-bundle
inventory, per-result digests, identities, one-thread receipts, finite JSON
scores, assignment permutations, source-ID uniqueness/exclusions, and private
receipt permissions. Any preserved failed attempt rejects completion.

Synthetic tests exercise corruption, missing evidence, non-finite values, path
escape and firewall failures. They do not read protected measurement results.
This module is NOT the complete independent scientific verifier and cannot mark
Priority 35B COMPLETE: independent tensor/mechanism/metric/control/statistical
recomputation, figure QA, report-prefix assertion and final seals remain pending.
The original frozen measurement driver and all earlier evidence remain untouched.
