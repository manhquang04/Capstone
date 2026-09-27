# Technical replay amendment: Priority 6 PaySim target provenance

Status: `AUTHORIZED_BEFORE_REPLAY`

Written at: 2026-09-16T16:47:30Z, before any Priority 6 attacker gate and
before the replay target draw.

The frozen PaySim target generator wrote
`paysim_priority6_targets.pt` successfully, then failed while constructing its
provenance matrix because the repository-wide scan encountered an IEEE-CIS
target whose group dictionaries use `source_rows`, not PaySim `source_ids`.
No attack result was observed.  This is a dataset-namespace parsing fault, not
a scientific outcome.

The incomplete file is retained and treated as excluded historical data.  It
must not be used for a gate.  The target loader is corrected to ignore
non-PaySim group dictionaries that do not contain `source_ids`.  One replay
draw is authorized with precommitted seed `2026091665` and output
`paysim_priority6_targets_replay1.pt`.  All other frozen Priority 6 settings
remain unchanged.

