# B3 scientific table serialization — 2026-10-11, additive infrastructure

Written before protected quality analysis. This pure module prepares complete
9,984 per-target records, 768 arm/cell/metric medians and Holm-144, retaining
NA p=1 reservations and clip-only matching labels. Every scientific input is
required; missing outcomes never become NA. Accuracy is a fraction. MSE and
bounded MSE remain unchanged stored scores. The unit is 39 paired row sets,
not the number of features or records within a batch. Intervals use ranks13/27.

Machine-readable CSV strings encode exact JSON scalar values and quoted nested
structures; no presentation rounding affects signs/ties. Independent recomputation
parses the serialized values and checks NumPy medians/ranks, separate SciPy sign
tests and NumPy Holm from the fixed family. Synthetic tests use complete invented
fixtures only, reject corruption and missing outputs, and write no measurement CSV.

Production calls require the completed post-exit scientific audit first. This
module reads/writes no files and changes no measurement, match, rule or family.
Figure/report publication and full final manifest remain pending.
