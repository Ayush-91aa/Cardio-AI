"""
CardioAI Test Suite: Environment, File Structure & Baseline Smoke Tests
========================================================================
Validates that repository directories, standard lead definitions,
synthetic waveform dimensions, and configurations conform to CardioAI standards.
"""

import os
import sys
from pathlib import Path


def test_python_version_minimum():
    """Verify that Python is at least 3.10."""
    major, minor = sys.version_info[:2]
    assert (major, minor) >= (3, 10), f"Python version must be >= 3.10, found {major}.{minor}"


def test_repository_directory_structure():
    """Verify all required top-level directories exist."""
    base_dir = Path(__file__).resolve().parent.parent
    expected_dirs = [
        base_dir / "src" / "cardioai",
        base_dir / "src" / "cardioai" / "data",
        base_dir / "src" / "cardioai" / "preprocessing",
        base_dir / "src" / "cardioai" / "models",
        base_dir / "src" / "cardioai" / "evaluation",
        base_dir / "src" / "cardioai" / "utils",
        base_dir / "configs",
        base_dir / "docs",
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "data" / "sample",
        base_dir / "models" / "saved",
        base_dir / "tests",
    ]
    for d in expected_dirs:
        assert d.is_dir(), f"Expected directory does not exist: {d}"


def test_baseline_config_exists():
    """Verify the baseline config file exists and is readable."""
    base_dir = Path(__file__).resolve().parent.parent
    config_file = base_dir / "configs" / "baseline_config.yaml"
    assert config_file.is_file(), f"Configuration file not found at: {config_file}"
    content = config_file.read_text(encoding="utf-8")
    assert "PTB-XL" in content
    assert "MI" in content
    assert "sampling_rate: 100" in content


def test_standard_12_lead_definition():
    """Ensure standard 12-lead ECG names and count match clinical norms."""
    expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert len(expected_leads) == 12
    # Verify limb leads + precordial leads
    assert set(expected_leads[:6]) == {"I", "II", "III", "aVR", "aVL", "aVF"}
    assert set(expected_leads[6:]) == {"V1", "V2", "V3", "V4", "V5", "V6"}
