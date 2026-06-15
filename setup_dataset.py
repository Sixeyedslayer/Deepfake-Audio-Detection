"""
setup_dataset.py — Download and verify the Fake-or-Real (FoR) dataset from Kaggle.

This script handles:
  1. Checking Kaggle API credentials
  2. Downloading the dataset
  3. Extracting the for-norm subset
  4. Verifying directory structure and file counts
  5. Printing dataset statistics

Usage:
    python setup_dataset.py

First-time Kaggle API setup:
    1. Go to kaggle.com → Profile → Settings → API → Create New Token
    2. Place the downloaded kaggle.json at: C:\\Users\\<username>\\.kaggle\\kaggle.json
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict


# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_SLUG = "mohammedabdeldayem/the-fake-or-real-dataset"
DATA_DIR = Path(__file__).parent / "data"
TARGET_DIR = DATA_DIR / "for-norm"

EXPECTED_STRUCTURE = {
    "training": ["real", "fake"],
    "validation": ["real", "fake"],
    "testing": ["real", "fake"],
}

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are properly configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("=" * 60)
        print("  KAGGLE API SETUP REQUIRED")
        print("=" * 60)
        print()
        print("  Your kaggle.json file was not found.")
        print()
        print("  Follow these steps:")
        print()
        print("  1. Go to https://www.kaggle.com")
        print("  2. Sign in (or create a free account)")
        print("  3. Click your profile icon (top-right)")
        print("  4. Click 'Settings'")
        print("  5. Scroll to the 'API' section")
        print("  6. Click 'Create New Token'")
        print("  7. This downloads a kaggle.json file")
        print(f"  8. Move it to: {kaggle_json}")
        print()
        print(f"  The .kaggle directory is at: {kaggle_dir}")
        
        # Create .kaggle directory if it doesn't exist
        kaggle_dir.mkdir(exist_ok=True)
        
        print()
        print("  After placing kaggle.json, run this script again.")
        print("=" * 60)
        return False
    
    # Validate the JSON content
    try:
        with open(kaggle_json, "r") as f:
            creds = json.load(f)
        
        if "username" not in creds or "key" not in creds:
            print("  ERROR: kaggle.json is missing 'username' or 'key' fields.")
            print(f"  File location: {kaggle_json}")
            print("  Expected format: {\"username\": \"...\", \"key\": \"...\"}")
            return False
        
        print(f"  Kaggle credentials found for user: {creds['username']}")
        
        # Set proper permissions (Windows doesn't need chmod, but set env vars)
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]
        
        return True
        
    except json.JSONDecodeError:
        print(f"  ERROR: kaggle.json is not valid JSON. File: {kaggle_json}")
        return False


def download_dataset() -> bool:
    """Download the Fake-or-Real dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("  Installing kaggle package...")
        os.system(f"{sys.executable} -m pip install kaggle --quiet")
        from kaggle.api.kaggle_api_extended import KaggleApi
    
    print(f"\n  Downloading dataset: {DATASET_SLUG}")
    print(f"  Destination: {DATA_DIR}")
    print("  This may take 10-30 minutes depending on your connection...")
    print()
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        api.dataset_download_files(
            DATASET_SLUG,
            path=str(DATA_DIR),
            unzip=True,
            quiet=False
        )
        
        print("\n  Download complete!")
        return True
        
    except Exception as e:
        print(f"\n  ERROR downloading dataset: {e}")
        print("\n  Alternative: Download manually from:")
        print(f"  https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print(f"  Then extract to: {DATA_DIR}")
        return False


def find_and_organize_dataset() -> bool:
    """
    Find the for-norm directory in the downloaded data and organize it.
    Kaggle datasets sometimes have nested folders after extraction.
    """
    if TARGET_DIR.exists() and (TARGET_DIR / "training").exists():
        print(f"  Dataset already organized at: {TARGET_DIR}")
        return True
    
    print("  Searching for for-norm directory in downloaded data...")
    
    # Search for the for-norm directory recursively
    for_norm_candidates = list(DATA_DIR.rglob("for-norm"))
    
    # Also check for alternate names
    if not for_norm_candidates:
        for_norm_candidates = list(DATA_DIR.rglob("for_norm"))
    
    if not for_norm_candidates:
        # Check if training/real exists directly under some subfolder
        training_dirs = list(DATA_DIR.rglob("training"))
        for td in training_dirs:
            if (td / "real").exists() and (td / "fake").exists():
                # Found it — the parent is our target
                source = td.parent
                print(f"  Found dataset at: {source}")
                if source != TARGET_DIR:
                    # Move/copy to expected location
                    TARGET_DIR.mkdir(parents=True, exist_ok=True)
                    for item in source.iterdir():
                        dest = TARGET_DIR / item.name
                        if not dest.exists():
                            shutil.move(str(item), str(dest))
                return True
    
    if for_norm_candidates:
        source = for_norm_candidates[0]
        print(f"  Found for-norm at: {source}")
        
        if source != TARGET_DIR:
            # Check if it has an extra nested folder
            if (source / "for-norm").exists():
                source = source / "for-norm"
            
            # Move to expected location
            if not TARGET_DIR.exists():
                shutil.move(str(source), str(TARGET_DIR))
            else:
                for item in source.iterdir():
                    dest = TARGET_DIR / item.name
                    if not dest.exists():
                        shutil.move(str(item), str(dest))
        
        return True
    
    print("  ERROR: Could not find the for-norm directory.")
    print(f"  Please ensure the dataset is extracted at: {TARGET_DIR}")
    print("  Expected structure:")
    print("    data/for-norm/training/real/*.wav")
    print("    data/for-norm/training/fake/*.wav")
    print("    data/for-norm/validation/real/*.wav")
    print("    data/for-norm/validation/fake/*.wav")
    print("    data/for-norm/testing/real/*.wav")
    print("    data/for-norm/testing/fake/*.wav")
    return False


def verify_dataset() -> dict:
    """
    Verify the dataset structure and count files.
    
    Returns:
        Dictionary with file counts per split and class.
    """
    print("\n  Verifying dataset structure...")
    stats = {}
    all_ok = True
    
    for split, classes in EXPECTED_STRUCTURE.items():
        split_dir = TARGET_DIR / split
        stats[split] = {}
        
        if not split_dir.exists():
            print(f"  [MISSING] {split_dir}")
            all_ok = False
            continue
        
        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"  [MISSING] {cls_dir}")
                all_ok = False
                stats[split][cls] = 0
                continue
            
            # Count audio files
            count = sum(
                1 for f in cls_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            stats[split][cls] = count
            
            status = "[OK]" if count > 0 else "[MISSING]"
            print(f"  {status} {split}/{cls}: {count:,} files")
    
    return stats if all_ok else None


def print_dataset_summary(stats: dict):
    """Print a formatted summary of the dataset."""
    if stats is None:
        return
    
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  {'Split':<15} {'Real':>8} {'Fake':>8} {'Total':>8}")
    print("  " + "-" * 42)
    
    grand_total = 0
    for split in ["training", "validation", "testing"]:
        if split in stats:
            real = stats[split].get("real", 0)
            fake = stats[split].get("fake", 0)
            total = real + fake
            grand_total += total
            print(f"  {split:<15} {real:>8,} {fake:>8,} {total:>8,}")
    
    print("  " + "-" * 42)
    print(f"  {'TOTAL':<15} {'':>8} {'':>8} {grand_total:>8,}")
    print("=" * 60)
    print("\n  Dataset is ready for training! [OK]")
    print(f"  Location: {TARGET_DIR}")
    print()


def main():
    print()
    print("=" * 60)
    print("  DEEPFAKE AUDIO DETECTION - DATASET SETUP")
    print("=" * 60)
    print()
    
    # Step 1: Check if dataset already exists
    stats = verify_dataset()
    if stats:
        print_dataset_summary(stats)
        return
    
    # Step 2: Check Kaggle credentials
    if not check_kaggle_credentials():
        sys.exit(1)
    
    # Step 3: Download dataset
    if not download_dataset():
        sys.exit(1)
    
    # Step 4: Organize dataset
    if not find_and_organize_dataset():
        sys.exit(1)
    
    # Step 5: Verify
    stats = verify_dataset()
    if stats:
        print_dataset_summary(stats)
    else:
        print("\n  ERROR: Dataset verification failed after download.")
        print("  Please check the directory structure manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
