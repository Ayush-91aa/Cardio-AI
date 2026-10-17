"""
CardioAI Test Suite: PTB-XL Metadata, Schema & Leakage Verification Tests
==========================================================================
Verifies that metadata files, SCP parsing functions, diagnostic class mappings,
and patient-stratified fold isolation conform to scientific and software standards.
"""

from pathlib import Path
import pytest
import pandas as pd

from cardioai.data.ptbxl_schema import (
    DIAGNOSTIC_SUPERCLASSES,
    load_scp_statements,
    parse_scp_codes,
    get_diagnostic_superclass_mapping,
    get_mi_subclass_mapping,
    map_record_superclasses,
    verify_patient_split_isolation,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def scp_statements_path(repo_root) -> Path:
    return repo_root / "data" / "raw" / "ptbxl" / "scp_statements.csv"


@pytest.fixture
def ptbxl_database_path(repo_root) -> Path:
    return repo_root / "data" / "raw" / "ptbxl" / "ptbxl_database.csv"


def test_parse_scp_codes_valid_and_invalid():
    """Verify that stringified SCP dictionaries are parsed safely."""
    # Standard valid string
    raw_str = "{'IMI': 100.0, 'ABQRS': 0.0}"
    parsed = parse_scp_codes(raw_str)
    assert parsed == {"IMI": 100.0, "ABQRS": 0.0}

    # Empty / NaN inputs
    assert parse_scp_codes(None) == {}
    assert parse_scp_codes("") == {}
    assert parse_scp_codes(float("nan")) == {}

    # Malformed string syntax
    assert parse_scp_codes("invalid syntax {{{{") == {}


def test_load_scp_statements(scp_statements_path):
    """Verify SCP statements CSV loading and index structure."""
    if not scp_statements_path.is_file():
        pytest.skip("scp_statements.csv not downloaded yet.")

    df = load_scp_statements(scp_statements_path)
    assert "IMI" in df.index
    assert "NORM" in df.index
    assert "diagnostic_class" in df.columns
    assert df.loc["IMI", "diagnostic_class"] == "MI"
    assert df.loc["NORM", "diagnostic_class"] == "NORM"


def test_diagnostic_superclass_mapping(scp_statements_path):
    """Verify mapping of clinical SCP codes to the 5 diagnostic superclasses."""
    if not scp_statements_path.is_file():
        pytest.skip("scp_statements.csv not downloaded yet.")

    df = load_scp_statements(scp_statements_path)
    code_map = get_diagnostic_superclass_mapping(df)

    # Core MI checks
    assert code_map.get("IMI") == "MI"
    assert code_map.get("AMI") == "MI"
    assert code_map.get("ASMI") == "MI"
    assert code_map.get("ALMI") == "MI"

    # Core NORM check
    assert code_map.get("NORM") == "NORM"

    # Core Conduction Disturbance checks
    assert code_map.get("CLBBB") == "CD"
    assert code_map.get("CRBBB") == "CD"

    # Core ST/T changes
    assert code_map.get("ISCAL") == "STTC"

    # Core Hypertrophy
    assert code_map.get("LVH") == "HYP"


def test_map_record_superclasses(scp_statements_path):
    """Verify that multi-label records correctly identify all present superclasses."""
    if not scp_statements_path.is_file():
        pytest.skip("scp_statements.csv not downloaded yet.")

    df = load_scp_statements(scp_statements_path)
    code_map = get_diagnostic_superclass_mapping(df)

    # Multi-label record: Patient with Inferior MI + Complete Left Bundle Branch Block
    patient_codes = {"IMI": 100.0, "CLBBB": 100.0, "SR": 100.0}
    superclasses = map_record_superclasses(patient_codes, code_map)
    assert superclasses == {"MI", "CD"}  # 'SR' is a rhythm statement, not a diagnostic superclass


def test_ptbxl_stratified_folds_leakage_free(ptbxl_database_path):
    """
    Rigorously verify that patients across folds 1 through 10 have 0 overlap.
    """
    if not ptbxl_database_path.is_file():
        pytest.skip("ptbxl_database.csv not downloaded yet.")

    df = pd.read_csv(ptbxl_database_path, index_col="ecg_id")
    result = verify_patient_split_isolation(df, "strat_fold")

    assert result["is_leakage_free"] is True, "Patient leakage detected across evaluation folds!"
    assert result["total_unique_patients"] == 18869
    assert result["sum_of_fold_patients"] == 18869
    assert result["overlap_count"] == 0
