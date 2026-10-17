"""
CardioAI - PTB-XL Clinical Metadata Inspection Engine
=====================================================
Performs detailed, rigorous statistical inspection of the PTB-XL dataset metadata:
  1. Patient demographics and recording volumes.
  2. Diagnostic superclasses and multi-label co-occurrence.
  3. In-depth Myocardial Infarction (MI) subclass and statement breakdown.
  4. Signal quality annotations (drift, static, burst noise, electrode issues).
  5. Patient-stratified fold balance and leakage-free verification.

Usage:
    python -m cardioai.data.inspect_metadata
    # or
    python src/cardioai/data/inspect_metadata.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure 'src' is in sys.path when running script directly
_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd

from cardioai.data.ptbxl_schema import (
    DIAGNOSTIC_SUPERCLASSES,
    SUPERCLASS_DESCRIPTIONS,
    load_scp_statements,
    parse_scp_codes,
    get_diagnostic_superclass_mapping,
    get_mi_subclass_mapping,
    map_record_superclasses,
    verify_patient_split_isolation,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(f" {title.upper()}")
    print("=" * 78)


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def inspect_ptbxl_dataset(
    data_dir: Path | None = None,
    output_summary_path: Path | None = None,
) -> Dict[str, Any]:
    """
    Executes full clinical metadata inspection on PTB-XL files.
    """
    if data_dir is None:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        data_dir = repo_root / "data" / "raw" / "ptbxl"

    db_path = data_dir / "ptbxl_database.csv"
    scp_path = data_dir / "scp_statements.csv"

    if not db_path.is_file() or not scp_path.is_file():
        raise FileNotFoundError(
            f"PTB-XL metadata files not found in {data_dir}. "
            "Please run downloader.py first to acquire the files."
        )

    print_header("CardioAI - PTB-XL Clinical Metadata & Schema Inspection")
    print(f"[*] Database file: {db_path.name} ({db_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"[*] SCP Statements file: {scp_path.name} ({scp_path.stat().st_size / 1024:.2f} KB)")

    # 1. Load Data
    df = pd.read_csv(db_path, index_col="ecg_id")
    scp_df = load_scp_statements(scp_path)
    code_to_super = get_diagnostic_superclass_mapping(scp_df)
    mi_subclass_map = get_mi_subclass_mapping(scp_df)

    n_records = len(df)
    n_unique_patients = df["patient_id"].nunique()

    # 2. Parse SCP Codes and Superclasses
    parsed_scp = df["scp_codes"].apply(parse_scp_codes)
    record_superclasses = parsed_scp.apply(lambda d: map_record_superclasses(d, code_to_super))

    # Add diagnostic flags to DataFrame
    for sc in DIAGNOSTIC_SUPERCLASSES:
        df[f"diag_{sc}"] = record_superclasses.apply(lambda s: sc in s)

    df["diag_count"] = record_superclasses.apply(len)
    df["has_any_diag"] = df["diag_count"] > 0

    # 3. Demographics & Data Volume
    print_section("1. Dataset Volume & Patient Demographics")
    records_per_patient = df["patient_id"].value_counts()
    patients_with_multi_ecg = (records_per_patient > 1).sum()

    age_series = df["age"].dropna()
    sex_counts = df["sex"].value_counts(dropna=False).to_dict()

    print(f"Total 12-lead ECG Recordings:  {n_records:,}")
    print(f"Total Unique Patients:          {n_unique_patients:,}")
    print(f"Average Recordings per Patient: {n_records / n_unique_patients:.2f}")
    print(f"Patients with Multiple ECGs:   {patients_with_multi_ecg:,} ({patients_with_multi_ecg / n_unique_patients * 100:.1f}%)")
    print(f"Max Recordings for One Patient: {records_per_patient.max()}")

    print(f"\nAge Demographics (Available: {len(age_series):,} | Missing: {df['age'].isna().sum():,}):")
    print(f"  Mean:   {age_series.mean():.1f} ± {age_series.std():.1f} years")
    print(f"  Median: {age_series.median():.1f} years (Range: {age_series.min():.0f} to {age_series.max():.0f})")

    # Age buckets
    age_bins = [0, 30, 50, 70, 120]
    age_labels = ["< 30", "30 - 49", "50 - 69", "70+"]
    age_dist = pd.cut(age_series, bins=age_bins, labels=age_labels, right=False).value_counts().sort_index()
    for lbl, cnt in age_dist.items():
        print(f"  Age {lbl:<10}: {cnt:>6,} ({cnt / len(age_series) * 100:5.1f}%)")

    print(f"\nSex Distribution:")
    # In PTB-XL: 0 represents Female, 1 represents Male
    for k, v in sex_counts.items():
        label = "Female (0)" if k == 0 else ("Male (1)" if k == 1 else f"Missing/Unknown ({k})")
        print(f"  {label:<18}: {v:>6,} ({v / n_records * 100:5.1f}%)")

    # 4. Diagnostic Superclass Distribution
    print_section("2. Diagnostic Superclass Frequencies (PTB-XL Multi-Label)")
    superclass_stats = {}
    for sc in DIAGNOSTIC_SUPERCLASSES:
        cnt = int(df[f"diag_{sc}"].sum())
        pct = (cnt / n_records) * 100
        superclass_stats[sc] = {"count": cnt, "percentage": round(pct, 2), "description": SUPERCLASS_DESCRIPTIONS[sc]}
        print(f"  {sc:<6} : {cnt:>6,} recordings ({pct:5.1f}%) -> {SUPERCLASS_DESCRIPTIONS[sc]}")

    no_diag_count = int((~df["has_any_diag"]).sum())
    print(f"  OTHER  : {no_diag_count:>6,} recordings ({no_diag_count / n_records * 100:5.1f}%) -> Form / rhythm only, no superclass")

    # Multi-label overlap
    print(f"\nMulti-Label Complexity:")
    diag_cnt_dist = df["diag_count"].value_counts().sort_index()
    for num_classes, cnt in diag_cnt_dist.items():
        print(f"  Recordings with {num_classes} superclass labels: {cnt:>6,} ({cnt / n_records * 100:5.1f}%)")

    # 5. Deep Dive: Myocardial Infarction (MI) Breakdown
    print_section("3. Myocardial Infarction (MI) Detailed Breakdown (Our First AI Task)")
    mi_records = df[df["diag_MI"]]
    n_mi = len(mi_records)
    print(f"Total MI-annotated recordings: {n_mi:,} ({n_mi / n_records * 100:.2f}% of dataset)")

    # MI Subclasses (AMI, IMI, LMI, PMI)
    mi_subclass_counts = {sc: 0 for sc in set(mi_subclass_map.values())}
    mi_statement_counts: Dict[str, int] = {}

    for codes in parsed_scp[df["diag_MI"]]:
        for c in codes:
            if c in mi_subclass_map:
                sub = mi_subclass_map[c]
                mi_subclass_counts[sub] = mi_subclass_counts.get(sub, 0) + 1
                mi_statement_counts[c] = mi_statement_counts.get(c, 0) + 1

    print("\nMI Subclasses (Anatomical infarction regions):")
    subclass_names = {
        "AMI": "Anterior / Anteroseptal / Anterolateral MI (LAD coronary artery)",
        "IMI": "Inferior / Inferoposterior / Inferolateral MI (RCA / LCx coronary artery)",
        "LMI": "Lateral MI (Left Circumflex artery)",
        "PMI": "Posterior MI",
    }
    for sub, cnt in sorted(mi_subclass_counts.items(), key=lambda x: x[1], reverse=True):
        desc = subclass_names.get(sub, "Other MI subclass")
        print(f"  {sub:<6} : {cnt:>5,} occurrences -> {desc}")

    print("\nTop Specific MI Diagnostic Statements:")
    for statement, cnt in sorted(mi_statement_counts.items(), key=lambda x: x[1], reverse=True):
        stmt_desc = scp_df.loc[statement, "description"] if statement in scp_df.index else "N/A"
        print(f"  {statement:<7} : {cnt:>5,} ({stmt_desc})")

    # Infarction stadium (acute vs chronic/old)
    if "infarction_stadium1" in df.columns:
        print("\nInfarction Stadium Distribution in MI records (Reported clinical timing):")
        stadium_counts = mi_records["infarction_stadium1"].value_counts(dropna=False)
        for stadium, cnt in stadium_counts.items():
            stadium_name = "Not specified / missing" if pd.isna(stadium) else str(stadium)
            print(f"  {stadium_name:<30}: {cnt:>5,} ({cnt / n_mi * 100:5.1f}%)")

    # 6. Signal Quality & Noise Annotations
    print_section("4. Signal Quality & Noise Annotations (Our Research Focus)")
    noise_cols = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]
    noise_stats = {}
    for col in noise_cols:
        if col in df.columns:
            has_noise = df[col].notna() & (df[col] != "")
            cnt = int(has_noise.sum())
            pct = (cnt / n_records) * 100
            noise_stats[col] = {"count": cnt, "percentage": round(pct, 2)}
            print(f"  {col:<22}: {cnt:>5,} recordings flagged ({pct:4.1f}%)")

    # 7. Stratified Fold Verification & Patient Leakage Check
    print_section("5. Patient-Stratified Evaluation Setup ('strat_fold' Verification)")
    leakage_result = verify_patient_split_isolation(df, "strat_fold")
    print(f"Patient Isolation Status: {'[PASSED] Leakage-Free (0 overlap)' if leakage_result['is_leakage_free'] else '[FAILED] Data Leakage Detected!'}")
    print(f"Total Unique Patients across all 10 folds: {leakage_result['total_unique_patients']:,}")
    print(f"Sum of Patients in individual folds:      {leakage_result['sum_of_fold_patients']:,}")

    print("\nFold-by-Fold Distribution (Official PTB-XL Split):")
    print(f"  {'Fold':<6} | {'Total Records':<14} | {'Unique Patients':<16} | {'MI Positives':<14} | {'MI Prevalence':<14}")
    print("  " + "-" * 72)

    fold_details = []
    for f in sorted(df["strat_fold"].unique()):
        fold_df = df[df["strat_fold"] == f]
        f_rec = len(fold_df)
        f_pat = fold_df["patient_id"].nunique()
        f_mi = int(fold_df["diag_MI"].sum())
        f_prev = (f_mi / f_rec) * 100
        fold_details.append({
            "fold": int(f),
            "records": f_rec,
            "patients": f_pat,
            "mi_count": f_mi,
            "mi_prevalence_pct": round(f_prev, 2),
        })
        print(f"  Fold {f:<2} | {f_rec:>13,} | {f_pat:>15,} | {f_mi:>13,} | {f_prev:>13.2f}%")

    # Recommended evaluation splits:
    # Folds 1-8: Train/Val (17,441 records)
    # Folds 9-10: Test (4,396 records)
    train_val_records = len(df[df["strat_fold"] <= 8])
    test_records = len(df[df["strat_fold"] >= 9])
    print(f"\nRecommended Split Breakdown:")
    print(f"  Folds 1-8  (Train / Validation): {train_val_records:,} records ({train_val_records / n_records * 100:.1f}%)")
    print(f"  Folds 9-10 (Held-Out Test)     : {test_records:,} records ({test_records / n_records * 100:.1f}%)")

    # 8. Save Structured JSON Summary
    summary_data = {
        "dataset_name": "PTB-XL",
        "dataset_version": "1.0.3",
        "total_records": n_records,
        "total_unique_patients": n_unique_patients,
        "records_per_patient_avg": round(float(n_records / n_unique_patients), 2),
        "demographics": {
            "age_mean": round(float(age_series.mean()), 2),
            "age_std": round(float(age_series.std()), 2),
            "age_median": round(float(age_series.median()), 2),
            "age_missing": int(df["age"].isna().sum()),
            "sex_counts": {str(k): int(v) for k, v in sex_counts.items()},
        },
        "superclass_distribution": superclass_stats,
        "mi_breakdown": {
            "total_mi_records": n_mi,
            "mi_prevalence_pct": round(float(n_mi / n_records * 100), 2),
            "subclasses": mi_subclass_counts,
            "statements": mi_statement_counts,
        },
        "noise_annotations": noise_stats,
        "stratification": {
            "is_leakage_free": bool(leakage_result["is_leakage_free"]),
            "folds": fold_details,
            "train_val_record_count": train_val_records,
            "test_record_count": test_records,
        },
    }

    if output_summary_path is None:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        output_summary_path = repo_root / "data" / "processed" / "ptbxl_metadata_summary.json"

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print_section("Summary Export")
    print(f"[+] Structured summary exported to: {output_summary_path}")
    print("=" * 78)

    return summary_data


if __name__ == "__main__":
    try:
        inspect_ptbxl_dataset()
    except Exception as exc:
        print(f"[-] Inspection failed: {exc}", file=sys.stderr)
        sys.exit(1)
