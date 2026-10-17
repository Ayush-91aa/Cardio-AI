# PTB-XL Dataset Notes & Clinical Metadata Analysis

This document records the exact statistical findings, clinical mappings, and scientific assumptions derived from inspecting the official **PhysioNet PTB-XL v1.0.3** clinical metadata.

---

## 1. Dataset Dimensions & Patient Demographics

* **Total 12-Lead ECG Records:** 21,799
* **Total Unique Patients:** 18,869
* **Average Recordings per Patient:** 1.16
* **Patients with Multiple ECGs:** 2,111 patients (11.2% of all patients) have between 2 and 10 serial ECG recordings.
* **Sex Distribution:**
  * Female (`0`): 11,354 recordings (52.1%)
  * Male (`1`): 10,445 recordings (47.9%)
* **Age Distribution:**
  * Mean: 62.8 ± 32.3 years | Median: 62.0 years
  * Age < 30: 1,379 (6.3%)
  * Age 30–49: 3,930 (18.0%)
  * Age 50–69: 9,689 (44.4%)
  * Age 70+: 6,508 (29.9%)
  * **Data Anomaly Note:** The raw dataset includes outlier ages up to 300 (due to recording/anonymization encoding). Our preprocessing pipeline will filter/clip age features for any models using demographic covariates.

---

## 2. Multi-Label Diagnostic Superclasses

PTB-XL contains 71 distinct SCP-ECG statements. Diagnostic statements are aggregated into five mutually non-exclusive superclasses:

| Superclass | Clinical Description | Record Count | Prevalence (%) |
| :--- | :--- | :--- | :--- |
| **NORM** | Normal ECG (no pathological abnormalities) | 9,514 | 43.6% |
| **MI** | **Myocardial Infarction (Our First AI Task)** | **5,469** | **25.1%** |
| **STTC** | ST/T Changes (ischemia, digitalis, repolarization) | 5,235 | 24.0% |
| **CD** | Conduction Disturbance (bundle branch blocks, AV blocks) | 4,898 | 22.5% |
| **HYP** | Hypertrophy (ventricular/atrial enlargement) | 2,649 | 12.2% |
| **OTHER** | Non-diagnostic statements (rhythm/form only) | 411 | 1.9% |

### Multi-Label Complexity
ECGs frequently display co-occurring pathologies:
* **1 Superclass:** 16,244 records (74.5%)
* **2 Superclasses:** 4,068 records (18.7%)
* **3 Superclasses:** 919 records (4.2%)
* **4 Superclasses:** 157 records (0.7%)

*Example:* A patient who suffered a prior inferior myocardial infarction (`MI`) may also present with a complete left bundle branch block (`CD`) and non-specific ST depression (`STTC`).

---

## 3. Deep Dive: Myocardial Infarction (MI) Breakdown

The **5,469 MI records** are categorized anatomically by coronary vascular territories:

### A. Anatomical Subclasses
1. **AMI (Anterior MI — Left Anterior Descending Artery [LAD]):**
   * 3,374 occurrences
   * Specific statements: `ASMI` (Anteroseptal MI - 2,357), `AMI` (Anterior MI - 353), `ALMI` (Anterolateral MI - 288), `INJAS` (Subendocardial injury in anteroseptal leads - 214), `INJAL` (145).
2. **IMI (Inferior MI — Right Coronary Artery [RCA] or LCx):**
   * 3,271 occurrences
   * Specific statements: `IMI` (Inferior MI - 2,676), `ILMI` (Inferolateral MI - 478), `IPLMI` (51), `IPMI` (33), `INJIN` (18), `INJIL` (15).
3. **LMI (Lateral MI — Left Circumflex Artery [LCx]):**
   * 201 occurrences (`LMI` - 201, `INJLA` - 17).
4. **PMI (Posterior MI):**
   * 17 occurrences (`PMI` - 17).

### B. Infarction Stadium & Clinical Safety Reality
* **Unknown / Unspecified:** 3,372 records (61.7%)
* **Stadium III (Chronic / Old Infarction scar):** 949 records (17.4%)
* **Stadium II-III (Subacute to Chronic):** 910 records (16.6%)
* **Stadium I (Acute Infarction):** 146 records (2.7%)
* **Stadium II (Subacute):** 88 records (1.6%)

> **Key Clinical AI Insight:** Only ~2.7% of MI-annotated recordings are confirmed acute Stage I infarctions. Over 34% are explicitly chronic healed infarcts (pathological Q-waves). This proves why CardioAI models must be framed as **retrospective ECG label classifiers**, not emergency diagnostic tools for active heart attacks.

---

## 4. Signal Quality & Noise Annotations (Our Research Focus)

PTB-XL contains direct human annotations of noise and technical artifacts:
* **Static Noise:** 3,260 records (15.0%)
* **Baseline Drift:** 1,598 records (7.3%)
* **Burst Noise:** 613 records (2.8%)
* **Electrode Problems:** 30 records (0.1%)

These annotations provide an objective ground-truth benchmark for evaluating our **Signal Quality Index (SQI)** in later phases.

---

## 5. Evaluation Setup & Leakage-Free Stratification

PTB-XL provides an official `strat_fold` column (values 1–10). Our automated verification proved:
* **0 shared patients across folds** (Total unique patients = 18,869; sum across folds = 18,869).
* MI prevalence is uniformly balanced across all 10 folds ($24.09\% - 25.90\%$).
* **Folds 1–8 (Train & Validation):** 17,418 records (79.9%)
* **Folds 9–10 (Held-Out Test):** 4,381 records (20.1%)
