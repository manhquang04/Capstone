# P33b utility execution clarification before any utility training

No utility/DP-calibration/confirmatory jobs have run. Qualification remains
unchanged. The registered P32 validation-only baseline quality gate also applies
to the newly disclosed native two-logit FC utility model: all16 required baseline
jobs finite, mean validation AUC>=.70, mean validation F1 greater than mean
predict-all-fraud F1 2*prevalence/(1+prevalence). Failure=>utility arm NA; no
six-config tuning extension in this priority. All50round budgets/lr/CE settings,
16seeds, grids, delta tolerance, source targets and16-test family unchanged.

Comparator implementation calls the existing epsilon_from_rdp signature using
noise_multiplier/sensitivity_ratio/delta/compositions/orders keyword parameters;
one add/remove whole-update release, not per-record DP. Unit/identity gates and
source/input freeze must pass before these new jobs launch. Native qualification
runner/core/hash freeze is not changed by adding these later-stage source files.
