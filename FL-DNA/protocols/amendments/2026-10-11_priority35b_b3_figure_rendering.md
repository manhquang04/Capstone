# Priority 35B B3 figure-rendering infrastructure

Date: 2026-10-11. Additive, post-launch presentation infrastructure, frozen
before final protected-data analysis. No scientific source, numeric point,
estimator-selection rule, grid, matching decision or statistic is changed.

The new renderer takes previously audited source-data rows, not job files.
It requires all 32 arms per dataset, one qualified preselected estimator per
panel, n=11 utility seeds, n=39 leakage row sets and ranks 13/27. Numeric
coordinates/error bars are unchanged. Shared x limits include utility gains
without clamping them to zero. Shared accuracy scales retain all data points.
PDF and 300-dpi PNG use 8.8 cm by 18 cm and at least 7-point text.

DP joint curves are separate for C=.01, .1, baseline median/2 and median;
per-tensor C=.01 is separate. Epsilon labels use k=1,000 and M=1,000,000.
Clip-only points are explicitly NO DP. Data-free lines, v1/v2 labels and the
different-configurations/n/interval disclosure are included. Label locations
are selected deterministically from a display grid, with leader lines and
collision rejection; scientific marker coordinates are never jittered,
omitted or altered. Labels that cannot fit fail closed and require review.

Three preserved synthetic layout attempts contain no measured results.
Attempt 1 exposed label overlap; attempt 2 resolved label collision but its
footer margin was insufficient. Attempt 3 passes automated text bounds and
collision checks and visual inspection of the Poppler-rendered PDF. These
are pre-analysis rendering refinements, not experimental failures or reruns.
The final measured figure still requires independent CSV/data checking and
visual QA after scientific verification. Synthetic QA cannot certify its
eventual layout or mark Priority 35B COMPLETE.

PDF and nature-statistics skills guided render inspection, explicit paired
units/interval definitions and careful different-configuration disclosures.
Existing reports, measurement artifacts and frozen sources are untouched.
