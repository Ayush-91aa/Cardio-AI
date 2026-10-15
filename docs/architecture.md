# CardioAI System Architecture

This document outlines the modular architectural design for **CardioAI**, explaining the rationale behind each component from a biomedical signal-processing and machine-learning perspective.

---

## 1. High-Level Modular Design

CardioAI is built around five decoupled modules:

```text
[ Raw 12-Lead ECG (WFDB / CSV) ]
               │
               ▼
┌──────────────────────────────┐
│     1. Data & Schema Layer   │  ──> Validates lead count (12), sampling rate (100 Hz),
│                              │      and enforces patient-level stratified splitting.
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  2. Signal Processing Layer  │  ──> 0.5 - 40 Hz bandpass filter, baseline wander removal,
│                              │      NaN/infinite value checks, and z-score normalization.
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      3. Modeling Layer       │  ──> Baseline statistical classifier (Phase 1)
│                              │      followed by evaluated deep-learning models (Phase 2).
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  4. Reliability & Explain    │  ──> Signal quality index (SQI), calibrated confidence,
│                              │      and lead-level temporal attribution.
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   5. Presentation Layer      │  ──> FastAPI REST API + interactive React/TS dashboard
│                              │      (Phase 3).
└──────────────────────────────┘
```

---

## 2. Key Engineering & Biomedical Decisions

### A. Sampling Frequency: 100 Hz vs. 500 Hz
* PTB-XL provides signals at both 100 Hz and 500 Hz.
* **Decision:** We initiate our baseline experiments using **100 Hz**.
* **Rationale:** A 10-second 12-lead ECG at 100 Hz contains $10 \times 100 = 1,000$ timepoints per lead ($1,000 \times 12 = 12,000$ values). At 500 Hz, that increases to 60,000 values. Clinical diagnostics for general MI superclasses primarily depend on low-to-mid frequency waveforms (P-waves, QRS complexes, ST-segment deviations, T-waves, occurring below 40 Hz). According to the Nyquist theorem, a sampling rate of 100 Hz permits faithful frequency representation up to 50 Hz, perfectly suited for our 0.5–40 Hz filtering while dramatically speeding up training on student laptops.

### B. Patient-Wise Stratified Splitting (Preventing Data Leakage)
* In PTB-XL, some patients have multiple recordings taken over days or months.
* **Danger:** If recordings from the same patient appear in both the training set and the test set, the model might simply memorize patient-specific idiosyncrasies rather than learning generalizable cardiac pathology.
* **Decision:** We strictly adhere to PTB-XL's recommended `strat_fold` setup (folds 1–8 for training/validation, folds 9–10 for testing), ensuring all recordings from any given patient are isolated entirely within a single fold.

### C. Standard Lead Ordering
12-lead ECGs consist of:
* **Limb Leads (Frontal Plane):** Lead I, Lead II, Lead III, aVR, aVL, aVF.
* **Precordial Leads (Horizontal Plane):** V1, V2, V3, V4, V5, V6.
CardioAI validates and preserves this exact 12-channel ordering to ensure spatial lead attributions remain physiologically meaningful.
