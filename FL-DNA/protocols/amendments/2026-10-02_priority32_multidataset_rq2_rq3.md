# Priority 32: multi-dataset RQ2 replication and RQ3 extension

Date: 2026-10-02. Status: PREPROCESSING / VALIDATION DESIGN FROZEN.
Stages 0–2 authorized; Stage 4 is prohibited until all quality gates pass and
the separate execution-freeze manifest records the selected configurations.
No earlier finding, research question, artifact, or report is replaced.

## Immutable scope and execution

PaySim, IEEE-CIS, BAF; baseline, DNA v1 conservative (block 256, mix .08,
keep .88, shrink .45), DNA v2 (ratio .95, eta .01). CPU float32, one Torch
thread per process, four worker processes (within the authorized 4–6).
No reconstruction attacks. No edits to datasets, Latex, external_defenses,
earlier transforms, loaders, artifacts, or reports. New outputs live under
artifacts/priority32_multidataset. Every launched/skipped/failed/resumed job
is recorded. A validated completed result is never recomputed.

## Stage 0–1: data, splitting, preprocessing

Raw paths: datasets/creditcard.csv; datasets/ieee-fraud-detection/
train_transaction.csv and train_identity.csv; datasets/baf/Base.csv.
Missing files, absent/broken binary labels, duplicate IEEE identity IDs,
nonfinite processed features, intersecting partition source IDs, or IEEE
input dimension above 512 stop execution. Record raw SHA-256, full row count,
fraud rate, selected partition counts/rates, and processed feature dimension.

Split/subsample seed 320032. PaySim follows the unchanged loader's
stratified train_test_split logic: 65/15/20; its feature definitions and
non-IID partition helper are imported without changing the loader. The new
driver fits medians only on train (the legacy helper imputes before splitting,
so it is used only when its numerical input contains no missing values).
IEEE left-joins identity on TransactionID with many-to-one validation, keeps
TransactionID only as provenance, orders by TransactionDT with stable sorting,
and splits first floor(.65*n), next through floor(.80*n), last remainder.
BAF stable-orders by month, uses 0–4/5/6–7, rejects other months.

Cap total at 500,000: proportional largest-remainder allocation over the
three original partitions, stratified selection independently inside each
partition using seed + partition index; selected indices retain original
partition ordering. No cross-partition sampling. Record source IDs and split
hashes. This shared-pipeline PaySim replication does not replace frozen RQ2.

BAF: every exact numeric -1 sentinel becomes NaN and an explicit missing
indicator; other negative numbers stay unchanged. The five named categorical
columns are categorical. IEEE object columns are categorical; drop >90%
missing columns determined on capped train only, recording the complete list.
Numerical columns: train median imputation then train RobustScaler; an entirely
missing numerical feature outside IEEE uses a declared constant zero fill.
Categoricals: <=20 train levels, train one-hot with unknown all-zero; >20,
train empirical frequency encoding with unknown zero. Categorical missing
is a dedicated __MISSING__ level. No label-derived features.

K=3; fraud rows evenly split using the existing loader helper; nonfraud
groups use sorted type / ProductCD / payment_type, primary client index mod 3,
55% to primary and 22.5% to each other. Import the existing helper by passing
the natural category as its 'type' column. Save mappings and client summaries.

## Stage 2: validation-only quality selection, fixed before training

Quality seed 320100; new FraudMLP(input_dim), BatchNorm/dropout unchanged;
BinaryFocalLoss gamma=2; fresh Adam per client per round; one local epoch,
batch1024, lr .001, 50 rounds. Evaluate FINAL round only. Threshold is the
unchanged fraud_fl_common.tune_threshold on validation. No test scores in
quality jobs, no test-based checkpoint/configuration selection.

Gate: validation AUC >= .70 AND validation F1 >= all-fraud validation F1 + .02
(the explicit operational definition of 'clearly above'). Try configurations
in this fixed order, stop selecting at the first passing one, maximum six:
1. rounds50/lr.001/focal alpha.95;
2. rounds50/lr.0003/alpha.95;
3. rounds50/lr.003/alpha.95;
4. rounds100/lr.001/alpha.95;
5. rounds50/lr.001/alpha.90;
6. rounds100/lr.0003/alpha.99.
If no configuration passes for any dataset, do not launch Stage 4; disclose
all validation attempts and ask for direction. This is not outcome tuning of
confirmatory data. Stage 3 records chosen configuration plus preparation and
quality-result hashes in execution_freeze.json BEFORE the first Stage 4 job.

## Frozen Stage 4–5 statistical design

21 paired replicates per dataset, seeds 321000 through 321020 inclusive,
identical across methods; check these seeds against prior registered training
seed lists before execution. Exactly 189 jobs, no post-hoc replicate increase.
Use fixed processed splits, seed-specific client partitions and model/dropout/
batch order. Final-round model; validation-F1 threshold, then test once.
Transmit/transform every floating state_dict delta, including BN buffers,
as the existing RQ2 protocol does. Nonfloating states use existing FedAvg
semantics. V1 calls the existing transform state function; v2 seed contract
derive_seed(derive_seed(seed,'rq2-dna-transform-v2'), 'dna_transform_v2', round,
client), quantization derive_seed(client_seed,'quantization').

Endpoints test F1, ROC-AUC, descriptive PR-AUC. Paired transform-minus-baseline
differences; sample SD(ddof1); two-sided 95% Student-t CI of the paired mean
(df20), strict lower bound > -.02 F1 and > -.005 AUC. Overall PASS iff both.
No multiplicity adjustment for these co-primary non-inferiority intersection
gates. Report absolute baseline means and all per-seed differences. An
independent NumPy/SciPy script recomputes values from job JSON, not summary.

## Stage 6: frozen cost design

Import the optimized existing run_rq3_benchmark.py and
analyze_rq3_benchmark.py, using protocols/config/rq3_v2_benchmark.json.
IEEE and BAF each reproduce 27 cells (RAW, v1, v2 x 3 network profiles x
K=3/5/10), with only input_dim changed to the audited size; 10 warmups,
50 measured repetitions, method_order_seed271828. Retain all Bundle B
criteria, bootstrap10000/seed271828 and CI-upper acceptance rule unchanged:
p95 round overhead25%, encode250ms, server500ms, client/server memory25%,
payload6x, zero failures/timeouts, original absolute memory and transfer caps.
The frozen synthetic named_parameters-only shape contract is retained;
this benchmark does not measure real full-state RQ2 traffic or training.

Descriptive DP cost on all THREE model dimensions in the SAME timed harness:
whole flattened synthetic client update L2 clip C100 and independent Gaussian
sigma .001, serialized as RAW_FLOAT32; clipping/noise included in client time,
server deserialize/aggregate unchanged. This is a cost-only example, NOT a
new calibrated privacy/utility comparator or DP guarantee. Keep all nine
profile/client-count cells per model (27 DP cells); measure new RAW references
for PaySim (9 cells); reuse current-run RAW for IEEE/BAF. Fixed per-round/client
seed contract, no caching of randomness. Cost stages run serially in isolated
processes AFTER training workers finish to avoid mutual timing interference.

## Progress, provenance, checks

checklist.json holds stages 0–7; progress.json/log update after each finished
job, with stage/dataset/method/done/total/failed/start/update/ETA. Parent runs
detached, four CPU workers; per-job atomic JSON with config SHA/seed and finite
metrics; validated-result resume only. Unfinished jobs may restart with exact
config and are disclosed, never retuned. Freeze manifests before execution.
Compile new Python and git diff --check; record final outputs/code/raw hashes
in sha256_manifest.csv. Do not claim completion until all gates and independent
statistics verification pass and no Priority32 workload remains.
