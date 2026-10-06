# Dataset provenance, use terms, and project preprocessing

**Scope.** This note documents the three datasets referenced by this repository's
loaders/experiments: PaySim (stored locally as `datasets/creditcard.csv`),
CIFAR-10, and Adult. It was prepared from live primary/authoritative sources on
2026-09-30. It does not download, inspect, alter, or redistribute any dataset.
License information is a record of the source consulted, not legal advice.

## PaySim (the project's `creditcard.csv`)

### Source, terms, and citation

* **Source:** Lopez-Rojas's Kaggle dataset, *Synthetic Financial Datasets For
  Fraud Detection*, ref `ealaxi/paysim1` ([dataset page](https://www.kaggle.com/datasets/ealaxi/paysim1)).
  The live [Kaggle metadata API](https://www.kaggle.com/api/v1/datasets/view/ealaxi/paysim1)
  identifies Edgar Lopez-Rojas as owner, version 2 (2017-04-03), and identifies
  the dataset as synthetic output from PaySim. Its description says it is a
  one-quarter-scale version made for Kaggle from runs calibrated to aggregated
  private mobile-money logs; it is **not** a public dump of those logs.
* **License:** the live metadata reports **CC BY-SA 4.0**. On sharing the
  material or an adaptation, retain attribution, link to the license, mark
  modifications, and apply a same-element/compatible ShareAlike license to an
  adaptation ([CC BY-SA 4.0 legal code, §3](https://creativecommons.org/licenses/by-sa/4.0/legalcode)).
  The dataset-specific Kaggle page's `/license` endpoint returned HTTP 404, so
  no additional dataset-page terms were verified. Kaggle's
  [Terms of Use](https://www.kaggle.com/terms) endpoint was live but its
  substantive text was client-rendered and was not recoverable in this fetch;
  any additional Kaggle-platform terms are therefore **unverified here**.
* **Cite:** E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, “PaySim: A
  financial mobile money simulator for fraud detection,” *Proceedings of the
  28th European Modeling and Simulation Symposium (EMSS)*, 2016,
  pp. 249–255, ISBN 978-88-97999-76-8. [Publisher PDF](https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf).
  The paper describes PaySim as an agent-based simulator derived from a sample
  of mobile-money-service logs, not as the original transactions.

### Exact project preprocessing

`data/load_creditcard.py` explicitly identifies its expected schema as PaySim
despite the local filename (`data/load_creditcard.py:75-80`). The target is
`isFraud`; `nameOrig`, `nameDest`, and `isFlaggedFraud` are intentionally
excluded (`:22`, `:24-43`, `:75-80`). The loader:

1. reads only `step`, `type`, `amount`, four balance fields, and `isFraud`
   (`:241-260`); if `max_rows`/`MAX_ROWS` is set, it makes a seeded stratified
   sample rather than taking early time steps (`:243-260`);
2. creates `balance_diff_orig = oldbalanceOrg - newbalanceOrig` and
   `balance_diff_dest = newbalanceDest - oldbalanceDest`; coerces every numeric
   feature and fills numeric missing values by that loaded subset's median;
   fills missing `type` as `UNKNOWN` (`:263-276`);
3. makes seeded (`FL_RUN_SEED`, default 42, `:20-23`) stratified splits of
   65% train / 15% validation / 20% test (`:95-109`);
4. fits `RobustScaler` **only on train** for the eight numeric columns, applies
   it to validation/test, fits a train one-hot `type` encoder with unknown
   categories ignored, and concatenates numeric then categorical values as
   `float32` (`:279-310`);
5. creates mild non-IID FL clients: all positive examples are shuffled/split
   evenly; negative examples are split by transaction type with 55% allocated
   to a type's primary client and the remaining 45% spread across other clients
   (`:314-359`). It calculates train-only `negatives / positives` class weight
   (`:143-147`).

### Ethical, privacy, and validity notes

* The file is synthetic, but it models a private financial domain. Do not
  describe it as real individual-level banking data or infer behavior of a
  named provider/country. Synthetic data can preserve population patterns and
  simulation artifacts; it is not a privacy guarantee for its confidential
  calibration source.
* Fraud labels represent the simulator's injected behavior, not adjudicated
  real-world fraud. Report results as performance on PaySim's simulated label
  mechanism, not operational fraud-detection performance.
* **Source/project conflict to disclose:** the live Kaggle description says
  that transactions detected as fraud are cancelled and specifically warns not
  to use `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, or
  `newbalanceDest` for fraud detection. The project does use all four fields
  and two deterministic differences derived from them (`data/load_creditcard.py:24-42,263-270`).
  This can create target leakage/overly optimistic results; evaluate a
  no-balance-feature ablation before making fraud-detection claims.

## CIFAR-10

### Source, terms, and citation

* **Source:** Alex Krizhevsky's authoritative
  [CIFAR-10 and CIFAR-100 datasets page](https://www.cs.toronto.edu/~kriz/cifar.html).
  It describes CIFAR-10 as 60,000 32×32 colour images in 10 classes, with
  50,000 training and 10,000 test images.
* **License/terms:** **unverified.** The authoritative CIFAR page fetched on
  2026-09-30 provides download files, description, and a recommended citation,
  but does not state an explicit dataset license or reuse terms. Do not assume
  that it is CC-licensed; obtain clarification/permission before redistribution
  or a use requiring an explicit license. The source images may also carry
  third-party rights not resolved by this documentation.
* **Cite:** A. Krizhevsky, *Learning Multiple Layers of Features from Tiny
  Images*, technical report, University of Toronto, 2009.
  [Author-hosted report PDF](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).

### Exact project loading/preprocessing

The principal project image experiments use `torchvision.datasets.CIFAR10`
under `datasets/cifar10` and `torchvision.transforms.ToTensor()`:

* training experiment: download enabled, training split, `ToTensor()` only
  (`experiments/run_priority7_image_domain_gate.py:27,246` and
  `experiments/priority23_repaired_cifar_attack.py:25,104`);
* reference reconstruction: test split with `download=False`, `ToTensor()`;
  its untransformed training set is used only to compute a raw per-pixel train
  mean baseline (`external_defenses/reference_image.py:16-18,49-55`);
* the vendored Inverting Gradients data pipeline separately downloads CIFAR-10,
  initially tensors it, then applies channel normalization using fixed
  `[0.4914672375, 0.4822617471, 0.4467701316]` mean and
  `[0.2470322400, 0.2434851378, 0.2615878582]` standard deviation
  (`external_defenses/invertinggradients/inversefed/consts.py:9-10`;
  `.../data/data_processing.py:58-81`). If augmentation is enabled there, its
  training transform adds `RandomCrop(32, padding=4)` and
  `RandomHorizontalFlip()` before normalization (`.../data/data_processing.py:73-81`);
  validation has no augmentation (`:81`).

### Ethical/privacy notes

Images and coarse object labels are not a consent/privacy benchmark. Images may
contain people or identifying context even when the class label is not a person;
models can memorize/reconstruct training examples. Do not use it to support
claims about surveillance, identity, demographic fairness, or performance on
real deployment populations. Its small, historic, web-collected image
distribution and unknown explicit license should constrain redistribution and
generalization claims.

## Adult / Census Income

### Source, terms, and citation

* **Source:** [UCI Machine Learning Repository: Adult](https://archive.ics.uci.edu/dataset/2/adult),
  donated 1996-04-30. UCI describes 48,842 instances and 14 features, with the
  task of predicting whether annual income exceeds $50,000; it says Barry
  Becker extracted records from the 1994 Census database subject to the listed
  age/income/final-weight/hours conditions.
* **License:** **CC BY 4.0**, as stated on the live UCI record. On sharing,
  give appropriate credit, link the license, indicate changes, and do not add
  restrictions that prevent recipients complying with the license
  ([CC BY 4.0 legal code, §3](https://creativecommons.org/licenses/by/4.0/legalcode)).
  No unavailable UCI terms were encountered in this verification.
* **Cite:** B. Becker and R. Kohavi (1996). *Adult* [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5XW20.

### Project reference and exact preprocessing

No top-level `data/` Adult loader was found. The repository contains the
vendored Tableak loader `external_defenses/tableak/datasets/adult.py`, which
expects `datasets/ADULT/adult.data` and `datasets/ADULT/adult.test`
(`:53-54`). It fixes the UCI category vocabulary and calls the latter column
`salary` (`:16-51`). It drops every row containing `?` (thereby also removing
the `Never-worked` category because its occupation is missing; `:59-65`), strips
defined feature dictionary via `to_numeric` (`:71-73`), casts features to
`float32` and labels to integer tensors (`:75-82`), then calculates its
standardization statistics and feature bounds (`:87-91`). This documents the
loader behavior; no Adult data were loaded for this work.

### Ethical/privacy notes

Adult contains sensitive/protected and proxy attributes, including age, sex,
race, marital status, relationship, occupation, and national origin. The
income threshold is an historical, socially contingent label, not a measure of
merit or ability. Dropping missing rows changes group representation and removes
`Never-worked`; report this exclusion and group-level effects. Results should
not be used for employment, credit, benefits, or other high-impact decisions,
and benchmark accuracy does not establish fairness, legal compliance, or causal
validity.

## Live-query log and verification status

All URLs below were fetched live on **2026-09-30**; these are the queries used
to establish the statements above.

| Target | Live query | Outcome |
|---|---|---|
| PaySim metadata/license/description | `https://www.kaggle.com/api/v1/datasets/view/ealaxi/paysim1` | Success; owner, version, CC BY-SA 4.0, description, and requested citation returned. |
| PaySim dataset-page license path | `https://www.kaggle.com/datasets/ealaxi/paysim1/license` | HTTP 404; additional dataset-page terms **unverified**. |
| Kaggle general terms | `https://www.kaggle.com/terms` | Endpoint live, but fetch exposed only client-rendered shell/metadata; substantive terms **unverified** here. |
| PaySim primary paper | `https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf` | Success; full text read (pp. 1–2). |
| CIFAR source | `https://www.cs.toronto.edu/~kriz/cifar.html` | Success; split/size/class description and citation guidance available; explicit license/terms absent, hence **unverified**. |
| CIFAR report | `https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf` | Success; author-hosted report retrieved. |
| Adult source/license/citation | `https://archive.ics.uci.edu/dataset/2/adult` | Success; description, CC BY 4.0, and DOI citation available. |
| CC legal codes | `https://creativecommons.org/licenses/by-sa/4.0/legalcode`; `https://creativecommons.org/licenses/by/4.0/legalcode` | Success; attribution and (where applicable) ShareAlike conditions checked. |

**File-only verification command (no data loading or training):**

```bash
python - <<'PY'
from pathlib import Path
p = Path('literature/datasets/dataset_notes.md')
s = p.read_text(encoding='utf-8')
required = ['PaySim', 'CIFAR-10', 'Adult / Census Income',
            'CC BY-SA 4.0', 'CC BY 4.0', 'data/load_creditcard.py:75-80',
            'unverified', 'Live-query log']
missing = [x for x in required if x not in s]
assert not missing, missing
print(f'OK: {p} ({len(s.splitlines())} lines)')
PY
```
