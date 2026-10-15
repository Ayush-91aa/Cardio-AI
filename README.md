# CardioAI: Explainable and Reliable AI Platform for ECG Analysis

> **A long-term biomedical AI engineering & research project.**
> **Core Principle:** Real data, real code, reproducible evaluation, honest limitations, and no fabricated medical predictions.

---

## 1. Project Vision & Research Context

**CardioAI** is a modular research and educational framework designed to investigate how artificial intelligence can analyze digital 12-lead electrocardiogram (ECG) signals, identify patterns associated with specific cardiac conditions (starting with **Myocardial Infarction [MI]**), and provide clinically relevant explainability and prediction reliability metrics.

### Medical Disclaimer & Ethical Boundary
> **CRITICAL MEDICAL DISCLAIMER:**
> CardioAI is strictly an educational and scientific research prototype. It is **NOT** an approved diagnostic device and must **NEVER** be used to guide clinical decisions, make definitive diagnoses, or replace the clinical judgment of qualified healthcare professionals. A machine-learning model trained on retrospective dataset labels does not establish whether an individual patient is experiencing an acute heart attack.

---

## 2. The Three Research Questions We Investigate

Rather than creating a generic classification model or simple website, CardioAI focuses on three open research challenges in biomedical AI:

1. **ECG Signal Quality Assessment:**
   * Can the system detect poor lead contact, baseline drift, powerline interference, or missing leads before inference?
   * *Goal:* Identify recordings that are too degraded for reliable automated analysis.
2. **Explainability & Attribution:**
   * Can the system highlight which specific leads (e.g., anterior leads $V_1-V_4$ vs. inferior leads $II, III, aVF$) and temporal segments (ST segments, T-waves, Q-waves) most heavily influenced a prediction?
   * *Goal:* Provide transparent, qualified visual insights rather than "black-box" outputs.
3. **Prediction Reliability & Abstention:**
   * Can the system identify ambiguous, low-confidence, or out-of-distribution signals and communicate uncertainty rather than outputting false confidence?
   * *Goal:* Know when the model should decline to make a definitive assertion.

---

## 3. First AI Task: PTB-XL Myocardial Infarction (MI) Baseline

* **Dataset:** PhysioNet PTB-XL v1.0.3 (large clinical 12-lead dataset with 21,837 clinical recordings from 18,885 patients).
* **Task Definition:** Classify ECG recordings exhibiting diagnostic patterns associated with Myocardial Infarction (`MI` superclass vs. Normal `NORM` and other conditions).
* **Baseline Approach:** Start with a reproducible, interpretable baseline (statistical time-domain & frequency-domain feature extraction + Logistic Regression / Linear SVM) before introducing complex 1D convolutional neural networks.
* **Leakage-Free Splitting:** Strict adherence to patient-stratified fold assignments (`strat_fold`) so no patient appears in both training and test sets.

---

## 4. Repository Structure

```text
d:\TDP SEM 3 PROJECT\
├── .gitignore                    # Ensures datasets, weights, and caches are never committed
├── README.md                     # Project master documentation & guide
├── requirements.txt              # Core scientific & ML dependencies
├── requirements-dev.txt          # Testing, linting, and notebook dependencies
├── check_env.py                  # Automated environment verification script
├── configs/                      # Reproducible experiment configurations
│   └── baseline_config.yaml      # Configuration for MI baseline pipeline
├── src/                          # CardioAI core Python package
│   └── cardioai/
│       ├── __init__.py
│       ├── data/                 # PTB-XL schema, metadata parsing, and loader
│       ├── preprocessing/        # Bandpass filters, normalization, lead checks
│       ├── models/               # Baseline ML models & neural architectures
│       ├── evaluation/           # Sensitivity, specificity, AUC, confusion matrix
│       └── utils/                # Logging, validation, and reproducibility
├── tests/                        # Automated test suite (pytest)
│   ├── __init__.py
│   └── test_environment.py       # Structure & sanity tests
├── docs/                         # In-depth architectural & ethical documentation
│   ├── architecture.md           # System design & modular pipelines
│   └── ethics_and_limitations.md # Medical safety & risk management
├── notebooks/                    # Jupyter notebooks for step-by-step learning
└── data/                         # Kept outside Git
    ├── raw/                      # PTB-XL raw records (.hea/.dat)
    ├── processed/                # Normalized numpy arrays & splits
    └── sample/                   # Small synthetic test fixtures for unit tests
```

---

## 5. Environment Setup (Windows PowerShell)

### Step 1: Create a Dedicated Python 3.11 Virtual Environment
Python 3.11 is strongly recommended because all scientific C-extension wheels (`scipy`, `scikit-learn`, `wfdb`) have precompiled, verified binaries.

```powershell
# Open Windows PowerShell in the project directory
cd "d:\TDP SEM 3 PROJECT"

# Create a virtual environment using Python 3.11
py -3.11 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

> **Note on PowerShell Execution Policy:** If you encounter a script execution error when running `Activate.ps1`, run:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Step 2: Install Dependencies
```powershell
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install core and development packages
pip install -r requirements-dev.txt
```

### Step 3: Verify the Environment
```powershell
# Run the automated diagnostic script
python check_env.py
```

### Step 4: Run the Unit Tests
```powershell
# Run the test suite
pytest -v
```

---

## 6. Long-Term B.Tech Roadmap

* **Phase 1 (Current):** Environment verification, PTB-XL metadata inspection, signal-processing basics, patient-stratified split setup, and reproducible baseline model.
* **Phase 2:** Advanced preprocessing, class imbalance handling, baseline vs. 1D-CNN comparisons, signal quality assessment module, and lead-attribution explainability experiments.
* **Phase 3:** Prediction uncertainty & abstention mechanisms, FastAPI backend, and interactive React/TypeScript frontend.
* **Phase 4:** Performance optimization, synthetic testing, expert evaluation, and regulatory/ethical assessments.
