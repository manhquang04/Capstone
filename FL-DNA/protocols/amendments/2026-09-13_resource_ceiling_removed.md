# Amendment — Machine-hour ceiling removed

**Amendment ID:** `2026-09-13-RESOURCE-CEILING-REMOVED`  
**Effective date:** 2026-09-13  
**Approved by:** Manh Quang (supervisor and research lead)  
**Deadline retained:** 2026-10-04  

## Decision

The `maximum_machine_hours` ceiling is removed for RQ1, RQ2 and RQ3 because
machine resources are available. The removal applies prospectively to the
remaining approved work and does not authorize outcome-dependent reruns,
post-result tuning, or an expansion to a second dataset.

The following ceilings remain frozen and unchanged:

- `maximum_artifact_gb = 500` for each track;
- `maximum_feasible_target_count = 150` for RQ1;
- `maximum_feasible_seed_count = 30` for RQ2.

These retained ceilings are statistical/design constraints rather than
machine-time constraints. A required sample size above a retained ceiling must
still be reported as underpowered or precision-limited; it must not be silently
truncated or used to justify changing alpha, power, margins, endpoints, or the
minimum meaningful effect.

## Scope and sequencing

The approved execution order is RQ3 deployment benchmarking, then the bounded
RQ1 extensions, then independent RQ1/RQ2 replications, and finally the
time-boxed full-client/full-FedAvg attacker-development track. The separate
report and pre-run amendment requirements for each group remain in force.

This amendment does not authorize the optional second-dataset study. The
supervisor reserved that decision.

## Administrative representation

In machine-readable configs, the unchanged field name is retained for audit
compatibility and represented as:

```yaml
maximum_machine_hours:
  approved: null
  status: REMOVED_BY_SUPERVISOR
  amendment: <path to this file>
```

`null` means that this protocol no longer imposes a machine-hour ceiling; it
does not mean that the approval is pending.
