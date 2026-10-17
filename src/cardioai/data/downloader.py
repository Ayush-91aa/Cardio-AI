"""
CardioAI - PhysioNet PTB-XL Dataset Downloader
==============================================
Provides automated, verified downloads of PTB-XL metadata files from PhysioNet.
Adheres to the principle of starting with lightweight metadata before downloading
large waveform archives.
"""

import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm


PHYSIONET_PTBXL_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

METADATA_FILES = [
    {
        "filename": "ptbxl_database.csv",
        "description": "Primary clinical metadata, patient demographics, and diagnostic codes",
        "expected_min_bytes": 6_000_000,  # ~6.5 MB
    },
    {
        "filename": "scp_statements.csv",
        "description": "SCP-ECG statement mappings, diagnostic classes, and statement types",
        "expected_min_bytes": 5_000,      # ~9 KB
    },
]


class DownloadProgressBar(tqdm):
    """Integrates tqdm progress bar with urllib.request."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, description: str) -> Path:
    """
    Downloads a single file from a given URL with progress indication.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Downloading: {output_path.name}")
    print(f"    Source: {url}")
    print(f"    Purpose: {description}")

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

    size = output_path.stat().st_size
    print(f"[+] Download complete: {output_path.name} ({size / (1024 * 1024):.2f} MB)")
    return output_path


def download_ptbxl_metadata(target_dir: Path | None = None, force: bool = False) -> dict[str, Path]:
    """
    Downloads PTB-XL metadata files (ptbxl_database.csv, scp_statements.csv).
    
    Args:
        target_dir: Directory to save the files. Defaults to 'data/raw/ptbxl'.
        force: If True, re-downloads even if the file already exists and is non-empty.

    Returns:
        Dictionary mapping filenames to their local Path objects.
    """
    if target_dir is None:
        # Determine repository root relative to this file
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        target_dir = repo_root / "data" / "raw" / "ptbxl"

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths = {}

    for item in METADATA_FILES:
        filename = item["filename"]
        desc = item["description"]
        expected_min = item["expected_min_bytes"]
        file_path = target_dir / filename
        file_url = PHYSIONET_PTBXL_BASE_URL + filename

        if file_path.is_file() and file_path.stat().st_size >= expected_min and not force:
            print(f"[+] Found existing valid file: {file_path.name} ({file_path.stat().st_size / (1024 * 1024):.2f} MB). Skipping download.")
            downloaded_paths[filename] = file_path
        else:
            download_file(file_url, file_path, desc)
            if file_path.stat().st_size < expected_min:
                raise RuntimeError(
                    f"Downloaded file {filename} is smaller than expected ({file_path.stat().st_size} bytes). Download may be corrupted."
                )
            downloaded_paths[filename] = file_path

    return downloaded_paths


if __name__ == "__main__":
    print("=" * 70)
    print(" CardioAI - PTB-XL Metadata Downloader")
    print("=" * 70)
    try:
        paths = download_ptbxl_metadata()
        print("\n[SUCCESS] All PTB-XL metadata files are downloaded and verified.")
        for k, v in paths.items():
            print(f"  * {k}: {v}")
    except Exception as exc:
        print(f"\n[-] Error downloading metadata: {exc}", file=sys.stderr)
        sys.exit(1)
