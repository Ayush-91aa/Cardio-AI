"""
CardioAI - Environment and Dependency Verification Script
=========================================================
This script verifies that your Python environment, core scientific libraries,
and system capabilities are properly configured for the CardioAI research platform.

Run this script anytime to check your setup:
    python check_env.py
"""

import sys
import os
import platform
import importlib

# Standard 12-Lead ECG standard nomenclature for verification
STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Core required packages for Phase 1
REQUIRED_PACKAGES = [
    ("numpy", "NumPy (Array computing & linear algebra)"),
    ("scipy", "SciPy (Scientific computing & digital signal processing)"),
    ("pandas", "Pandas (ECG metadata & diagnostic statement parsing)"),
    ("sklearn", "Scikit-Learn (Baseline ML models & evaluation metrics)"),
    ("matplotlib", "Matplotlib (Waveform plotting & figures)"),
    ("seaborn", "Seaborn (Statistical visualization)"),
    ("wfdb", "WFDB (PhysioNet Waveform Database reader)"),
    ("pytest", "Pytest (Automated unit & regression testing)"),
]

OPTIONAL_PACKAGES = [
    ("torch", "PyTorch (Deep learning - optional for later phases)"),
    ("fastapi", "FastAPI (REST API - planned for Phase 3)"),
    ("uvicorn", "Uvicorn (ASGI server - planned for Phase 3)"),
]


def print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def check_python_version() -> bool:
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    print(f"[*] Detected Python: {version_str} ({platform.architecture()[0]} on {platform.system()})")
    print(f"[*] Executable Path: {sys.executable}")

    # Check virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print("[+] Running inside a virtual environment (.venv)")
    else:
        print("[!] Warning: NOT currently running inside a virtual environment.")
        print("    Recommendation: Create and activate a virtual environment using:")
        print("    py -3.11 -m venv .venv")
        print("    .\\.venv\\Scripts\\Activate.ps1")

    # Evaluation of Python version
    if v.major == 3 and v.minor in [10, 11, 12]:
        print(f"[+] Python {version_str} is fully supported and recommended.")
        return True
    elif v.major == 3 and v.minor >= 13:
        print(f"[!] Notice: Python {version_str} is very recent.")
        print("    Some biomedical/C-extension packages (wfdb, scipy) may lack binary wheels.")
        print("    If you encounter wheel compilation errors, use Python 3.11 via: py -3.11")
        return True
    else:
        print(f"[-] Python {version_str} is below the minimum recommended version (3.10+).")
        return False


def check_packages(pkg_list, is_required=True):
    missing = []
    installed = []

    for mod_name, description in pkg_list:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            installed.append((mod_name, ver, description))
            print(f"  [+] {mod_name:<14} v{ver:<12} -> {description}")
        except ImportError:
            missing.append((mod_name, description))
            tag = "[-]" if is_required else "[?]"
            status = "MISSING (Required)" if is_required else "NOT INSTALLED (Optional)"
            print(f"  {tag} {mod_name:<14} {status:<15} -> {description}")

    return installed, missing


def test_synthetic_ecg_pipeline() -> bool:
    """Verifies that NumPy, SciPy, and Scikit-learn can execute a minimal ECG processing flow."""
    print_banner("Running Verification Smoke Test (Synthetic 12-Lead ECG)")
    try:
        import numpy as np
        from scipy import signal
        from sklearn.linear_model import LogisticRegression

        # 1. Generate 10 seconds of synthetic 12-lead ECG at 100 Hz (1000 samples x 12 leads)
        duration_sec = 10
        fs = 100
        n_samples = duration_sec * fs
        n_leads = 12

        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        # Synthetic baseline: heart rate ~ 60 bpm (1 Hz fundamental) + harmonics
        synthetic_signal = np.zeros((n_samples, n_leads), dtype=np.float32)
        for lead_idx in range(n_leads):
            phase_shift = lead_idx * 0.1
            lead_wave = (
                0.8 * np.sin(2 * np.pi * 1.0 * t + phase_shift)
                + 0.3 * np.sin(2 * np.pi * 3.0 * t + phase_shift)
                + 0.05 * np.random.normal(0, 1, n_samples)
            )
            synthetic_signal[:, lead_idx] = lead_wave

        # 2. Test DSP: Apply standard digital bandpass filter (0.5 Hz - 40 Hz at fs=100)
        # Note: at 100 Hz, Nyquist is 50 Hz. Highcut must be < 50 Hz.
        nyquist = 0.5 * fs
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        b, a = signal.butter(2, [low, high], btype="bandpass")
        filtered_signal = signal.filtfilt(b, a, synthetic_signal, axis=0)

        # 3. Test Feature Extraction & Baseline Classifier:
        # Extract basic time-domain statistics per lead (mean, std, peak-to-peak)
        features = np.hstack([
            np.mean(filtered_signal, axis=0),
            np.std(filtered_signal, axis=0),
            np.ptp(filtered_signal, axis=0)
        ]).reshape(1, -1)

        # Train a dummy logistic regression to verify sklearn C-extensions
        X_dummy = np.vstack([features, features * 1.1])
        y_dummy = np.array([0, 1])
        clf = LogisticRegression()
        clf.fit(X_dummy, y_dummy)
        pred = clf.predict(features)

        print("[+] Signal Synthesis Check: Generated 10s 12-lead synthetic ECG (1000 x 12).")
        print(f"[+] Signal Processing Check: Butterworth filter applied successfully.")
        print(f"[+] Feature Extraction Check: Extracted {features.shape[1]} statistical features.")
        print(f"[+] Baseline Classifier Check: Scikit-Learn LogisticRegression fit and predicted: {pred[0]}")
        print("[+] Mathematical and DSP stack is 100% operational!")
        return True

    except Exception as exc:
        print(f"[-] Smoke test failed with exception: {exc}")
        return False


def main():
    print_banner("CardioAI - Environment Health Check")
    py_ok = check_python_version()

    print("\n--- Checking Core Required Packages ---")
    _, missing_req = check_packages(REQUIRED_PACKAGES, is_required=True)

    print("\n--- Checking Optional Future Packages ---")
    _, _ = check_packages(OPTIONAL_PACKAGES, is_required=False)

    dsp_ok = False
    if not missing_req:
        dsp_ok = test_synthetic_ecg_pipeline()

    print_banner("Summary & Next Actions")
    if not missing_req and dsp_ok:
        print("[READY] All core dependencies and signal processing tools are functional!")
        print("        You are ready to proceed to Phase 1, Step 2 (Dataset Inspection & Metadata).")
        sys.exit(0)
    else:
        print("[ACTION REQUIRED] Some required packages are missing or not importable.")
        print(f"Missing count: {len(missing_req)} required package(s).")
        print("\nRecommended setup command:")
        print("  1) Activate your virtual environment:")
        print("     .\\.venv\\Scripts\\Activate.ps1")
        print("  2) Install requirements:")
        print("     pip install -r requirements.txt")
        print("     pip install -r requirements-dev.txt")
        print("  3) Re-run this check:")
        print("     python check_env.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
