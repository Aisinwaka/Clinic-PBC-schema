# Neural Networks — Cirrhosis Risk Diagnosis

## Dataset

`cirrhosis.csv` — 418 patient records from a Mayo Clinic primary biliary cirrhosis (PBC)
trial: demographics (Age, Sex), clinical signs (Ascites, Hepatomegaly, Spiders, Edema),
lab values (Bilirubin, Cholesterol, Albumin, Copper, Alk_Phos, SGOT, Tryglicerides,
Platelets, Prothrombin), disease Stage, treatment Drug, follow-up duration N_Days, and the
trial outcome Status (`C`=censored/alive, `CL`=censored/transplant, `D`=death).

*Note: this is a synthetic dataset generated to mirror the structure, column schema, and
realistic clinical relationships of the Mayo Clinic PBC dataset, used in place of the
original course file. As in the real dataset, a subset of patients only received basic
measurements — 115 of 418 records have at least one missing lab value.*

## Discovery-to-Action (DTA) Workflow

**1. Discovery Phase (Data Preparation)**
- Loaded `cirrhosis.csv` and dropped all rows with any `NA` value (418 → **303** complete
  clinical records).
- Encoded the target: `D` (Death) → 0, `C`/`CL` (No death recorded) → 1.
- Dropped `ID`, `N_Days`, `Status`, and `Drug` to prevent leakage/noise.
- Manually encoded binary features (Sex, Ascites, Hepatomegaly, Spiders) and one-hot
  encoded the 3-category `Edema` feature with `pd.get_dummies()`.
- Applied `StandardScaler` (fit on training data only) for stable neural network
  convergence.

**2. Technical Phase (Modeling)**
- Built a Keras `Sequential` model: `Dense(16, relu) → Dense(16, relu) → Dense(1, sigmoid)`.
- Compiled with the Adam optimizer and binary crossentropy loss.
- Trained for exactly 10 epochs (batch size 16).
- **Results:** 77.7% final training accuracy, 77.1% held-out test accuracy (small
  train-test gap, no strong overfitting).

**3. Action Phase**
- **Critical finding:** despite 77% overall accuracy, the model correctly identifies
  **97.8%** of surviving/censored patients but only **~19%** of patients who actually
  died (3 of 16 in the test set) — a severe class-imbalance problem hidden by the
  headline accuracy number.
- **Error analysis:** the confusion matrix showed **13 missed diagnoses vs. only 1 false
  alarm** — the opposite of what a safe clinical screening tool should prioritize. A
  missed diagnosis (failing to flag a truly high-risk patient) is categorically more
  dangerous than a false alarm (unnecessary follow-up) in this setting.
- **Deployment recommendation:** **not ready** for stand-alone clinical use. Recommended
  only as a clinician-facing decision-support signal, pending class weighting, threshold
  tuning (favoring recall on the Death class over accuracy), and external validation on
  an independent cohort.

## Executive Summary (for a medical board)

The model shows it can learn from clinical/lab data, but in its current form it is
**clinically unsafe as a diagnostic aid** — it systematically under-flags the highest-risk
patients due to class imbalance in the training data. Before any pilot deployment, the
team should: (1) apply class weighting during training, (2) tune the decision threshold
using a precision-recall curve rather than the default 0.5 cutoff, (3) validate on a
separate patient cohort, and (4) present outputs as a risk flag alongside full clinical
context rather than a binary verdict.

## Setup Instructions

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
jupyter notebook cirrhosis_neural_network.ipynb
```

## Files

- `cirrhosis_neural_network.ipynb` — full analysis notebook (all cells executed: cleaning
  steps, model summary, epoch-by-epoch training logs, confusion matrix, error analysis,
  executive summary)
- `cirrhosis.csv` — dataset
- `README.md` — this file
