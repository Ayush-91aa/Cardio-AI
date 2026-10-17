"""
CardioAI - PTB-XL Data Schema, Types, and Diagnostic Code Mappings
==================================================================
Defines standards, constants, and utilities for parsing PhysioNet PTB-XL metadata,
mapping SCP-ECG statements to clinical diagnostic superclasses (especially MI),
and verifying patient-stratified fold integrity.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd


# Five Diagnostic Superclasses in PTB-XL
DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Clinical descriptions for student & researcher clarity
SUPERCLASS_DESCRIPTIONS = {
    "NORM": "Normal ECG (no diagnostic abnormalities detected)",
    "MI": "Myocardial Infarction (ischemic heart injury / infarction patterns)",
    "STTC": "ST/T Changes (non-specific repolarization abnormalities, digitalis effects, ischemia)",
    "CD": "Conduction Disturbance (bundle branch blocks, hemiblocks, AV blocks, WPW)",
    "HYP": "Hypertrophy (ventricular or atrial hypertrophy/enlargement)",
}

# Standard 12-lead names in physiological clinical sequence
STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_scp_codes(scp_str: Any) -> Dict[str, float]:
    """
    Safely parses the string representation of the scp_codes dictionary
    stored in ptbxl_database.csv.

    Example input: "{'IMI': 100.0, 'ABQRS': 0.0}"
    Returns: {'IMI': 100.0, 'ABQRS': 0.0}
    """
    if pd.isna(scp_str) or not scp_str:
        return {}
    if isinstance(scp_str, dict):
        return scp_str
    try:
        parsed = ast.literal_eval(str(scp_str))
        if isinstance(parsed, dict):
            # Ensure keys are strings and values are floats
            return {str(k): float(v) for k, v in parsed.items()}
        return {}
    except (ValueError, SyntaxError):
        return {}


def load_scp_statements(scp_path: Path) -> pd.DataFrame:
    """
    Loads scp_statements.csv and indexes it by the statement code.
    Ensures diagnostic_class, diagnostic_subclass, and statement types are clean.
    """
    df = pd.read_csv(scp_path, index_col=0)
    # Fill NaN for categorical columns with empty strings
    for col in ["diagnostic_class", "diagnostic_subclass", "Statement Category"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def get_diagnostic_superclass_mapping(scp_statements_df: pd.DataFrame) -> Dict[str, str]:
    """
    Returns a dictionary mapping individual SCP codes to their diagnostic superclass
    (only for statements marked as diagnostic == 1.0).
    
    Example: 'IMI' -> 'MI', 'NORM' -> 'NORM', 'CLBBB' -> 'CD'
    """
    diag_rows = scp_statements_df[scp_statements_df["diagnostic"] == 1.0]
    return {
        code: row["diagnostic_class"]
        for code, row in diag_rows.iterrows()
        if row["diagnostic_class"] in DIAGNOSTIC_SUPERCLASSES
    }


def get_mi_subclass_mapping(scp_statements_df: pd.DataFrame) -> Dict[str, str]:
    """
    Returns a dictionary mapping MI-specific statement codes to their subclass
    (e.g., 'IMI' -> 'IMI' [Inferior], 'AMI' -> 'AMI' [Anterior], etc.).
    """
    mi_rows = scp_statements_df[
        (scp_statements_df["diagnostic"] == 1.0) &
        (scp_statements_df["diagnostic_class"] == "MI")
    ]
    return {code: row["diagnostic_subclass"] for code, row in mi_rows.iterrows()}


def map_record_superclasses(
    scp_dict: Dict[str, float],
    code_to_superclass: Dict[str, str],
    min_confidence: float = 0.0
) -> Set[str]:
    """
    Given a record's parsed scp_codes dictionary, returns the set of diagnostic
    superclasses present with confidence >= min_confidence.
    """
    superclasses = set()
    for code, conf in scp_dict.items():
        if conf >= min_confidence and code in code_to_superclass:
            superclasses.add(code_to_superclass[code])
    return superclasses


def verify_patient_split_isolation(df: pd.DataFrame, fold_column: str = "strat_fold") -> Dict[str, Any]:
    """
    Verifies that patients in different folds are completely mutually exclusive
    (0 patient leakage).

    Returns a report dictionary.
    """
    if fold_column not in df.columns or "patient_id" not in df.columns:
        raise ValueError(f"Required columns '{fold_column}' and 'patient_id' not found in DataFrame.")

    folds = sorted(df[fold_column].dropna().unique())
    patient_sets = {f: set(df[df[fold_column] == f]["patient_id"].dropna().unique()) for f in folds}

    total_unique_patients = len(df["patient_id"].dropna().unique())
    sum_of_fold_patients = sum(len(s) for s in patient_sets.values())

    overlap_detected = False
    overlaps = []

    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            f1, f2 = folds[i], folds[j]
            intersection = patient_sets[f1].intersection(patient_sets[f2])
            if intersection:
                overlap_detected = True
                overlaps.append({
                    "fold_a": f1,
                    "fold_b": f2,
                    "shared_patient_count": len(intersection),
                    "shared_patient_ids": list(intersection)[:5],
                })

    return {
        "is_leakage_free": not overlap_detected,
        "total_unique_patients": total_unique_patients,
        "sum_of_fold_patients": sum_of_fold_patients,
        "folds_evaluated": list(folds),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
    }
