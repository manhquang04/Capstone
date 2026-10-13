# Priority35B — post-hoc label coordinate presentation supplement

Date: 2026-10-12. Within the approved label-only repair, the first renderer
stopped before creating any new figure output because Matplotlib cap coordinates
were an object-dtype array. Preserve the frozen presentation source and error
record. A new additive label-position adapter converts only display-coordinate
arrays to float64 before the display transform; it does not convert measurement
values, points, error bars, model tensors or CSVs. The same layout, artifact
fingerprint and hash checks remain mandatory. This is not a scientific verifier
failure or a measurement replay. The descriptive extraction already passed.
