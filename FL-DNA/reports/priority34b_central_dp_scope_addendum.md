# Priority34B — central-DP scope label (2026-10-03)

All results in the unchanged reports/priority34b_report.md are labeled:
**central DP; requires secure aggregation to protect against the server**.
The trusted central noise mechanism does not protect individual raw uploads
from an honest-but-curious aggregation server. Secure aggregation must prevent
that server from seeing individual inputs, with a deployment-appropriate secure
noise-release protocol; central noise alone does not provide this.

All prior utility tables and conditional aggregate image-recovery measurements
remain intact, with original hashes. None is a local-DP comparator for individual
client recovery in P34C. The new client-side mechanism/noise-before-upload arm is
separately registered in 2026-10-03_priority34b_local_dp_extension.md, with new
results and report in the local-extension namespace. It is not yet complete.

This addendum corrects labels/interpretation, not historical numeric results.
