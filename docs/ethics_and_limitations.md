# CardioAI: Ethics, Medical Safety & Research Limitations

## 1. Ethical Purpose & Educational Boundary

**CardioAI** is developed as an academic engineering research project to investigate explainability, signal quality, and prediction reliability in automated ECG analysis.

* **Not a Medical Device:** CardioAI has not undergone clinical trials, FDA/CE clearance, or formal medical device certification.
* **No Diagnostic Authority:** The platform must never be used to establish a diagnosis, rule out acute coronary syndromes, or recommend changes in patient care.
* **Clinical Disclaimer:** Any user experiencing chest discomfort, shortness of breath, palpitations, or lightheadedness must immediately seek emergency medical attention.

---

## 2. Retrospective Dataset Labels vs. Acute Clinical Reality

A fundamental concept in medical machine learning is understanding what dataset labels actually represent:

* **What the PTB-XL `MI` label means:** In PTB-XL, annotations were assigned based on retrospective cardiologist reviews and clinical statements (e.g., `IMI` - Inferior Myocardial Infarction, `AMI` - Anterior Myocardial Infarction). Many recordings reflect prior, healed infarctions (e.g., persistent pathological Q-waves) rather than an acute, evolving ST-elevation myocardial infarction (STEMI) occurring in real time.
* **Clinical Reality:** In emergency medicine, acute myocardial infarction is diagnosed using a combination of the clinical presentation (chest pain, radiation), serial 12-lead ECG changes over time, and cardiac biomarkers (e.g., high-sensitivity troponin assays).
* **Research Position:** CardioAI models will be evaluated strictly as **ECG label classifiers**, not as diagnostic engines for acute heart attacks.

---

## 3. Signal Quality and Noise Limitations

* **Electrode Misplacement:** Swapping precordial or limb leads can completely change ECG morphology (e.g., false indications of axis deviation or infarction). AI algorithms trained without lead-swap detection can produce erroneous predictions.
* **Artifacts & Baseline Drift:** Patient respiration, movement, and muscle tremor (electromyographic noise) introduce high-frequency and low-frequency noise that can distort ST-segment elevation or depression.
* **Abstention Requirement:** A reliable AI system should know its limits. If signal quality is inadequate or if an input is out-of-distribution, the platform must flag uncertainty and decline to predict.

---

## 4. Privacy & Data Protection

* Raw patient ECG files must never be committed to Git or uploaded to public web services.
* Synthetic or de-identified public benchmark records (such as PTB-XL under PhysioNet's open data license) are used exclusively.
* In web inference (Phase 3), uploaded files will be processed in memory and discarded without permanent retention unless explicitly consented for localized research logging.
