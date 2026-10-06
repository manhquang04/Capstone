# Priority34C — authorized local-DP comparator correction

Written before any P34C experimental run; user explicitly chose option2.
Supersedes only the pending release-model alternatives in the original P34C
draft, which is retained unchanged. Honest-but-curious server sees each client's
individual payload. ALL primary arms operate on the SAME individual client
gradient, not an aggregate. The comparator is client-side DP from the separately
frozen P34B local extension: global clip C=.01, joint sensitivity2C, independent
client Gaussian SD2C*4.0156365196350094,50round per-client update-level accounting,
whole-training epsilon10/delta1e-5. Not record-level DP, not central DP.
The unprotected/DNA/DP gradients have identical parameter domain/checkpoint/target
and known-label side information within each instrument cell. No realized noise
is given to the attacker. Gradient query differs from the utility Adam-update
query; report it as a disclosed same-bounded-sensitivity instrument transfer,
not a claim to reproduce full Adam or final-model recovery. No utility matching.

Central P34B utility/recovery results remain historical, labeled central DP;
requires secure aggregation to protect against the server. They are NEVER
substituted as a per-client recovery comparator. Optional aggregate secondary
arm is NOT scheduled; user described it as optional, and avoiding it keeps the
main threat and72test family unchanged.

Distortion-matched comparators are also added on the client before transmission,
with independent noise and development-only matching, separate for v1/v2.
The nine cell/n8then24/n39/source firewall/native30member budget/72test Holm
requirements of the P34C draft remain. Batch1 is the easiest recovery setting.
BN eval-mode/public initial checkpoint is documented; no private learned BN is
uploaded. TabLeak FC/selectedIEEE60 variant remains separate from full P32 ratio
model. No scientific tuning/seed replacement/variance clamp.

Execution sequence: implement/test/freeze/run264-job LOCAL utility extension;
audit/report verified COMPLETE; then freeze the exact P34C seeds, payload-only
ratio/known-key sketch solver and source firewall; run independent qualification,
development matching and only qualified n39 cells; independently audit/report.
No P34C confirmatory run may precede local utility verification. Scientific
failure blocks dependent stages and requires direction, not substitute central
results or noise scales. Both priorities CPU/thread1, detached/resumable,
per-job progress and preserved partial artifacts. Original pending-direction
marker is historical; direction receipt below records the user's resolution.
