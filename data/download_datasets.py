"""
Dataset Download & Preparation Script
=======================================
Downloads and prepares the benchmark datasets used for
HCPG-GNN training and evaluation.

DATASETS USED:
  1. SmartBugs Curated     — 143 labeled vulnerable Solidity contracts
  2. SolidiFI Benchmark    — 9,369 contracts with injected vulnerabilities
  3. SB Wild (SmartBugs)   — 47,518 unique contracts from Etherscan

HOW DATASETS ARE ACCESSED:
  - This project uses a **synthetic HCPG dataset** generated programmatically
    by `models/train_model.py`.  The synthetic dataset simulates the HCPG
    (Heterogeneous Code Property Graph) feature vectors that would be extracted
    from real contracts using AST/CFG/DFG parsing.
  - The real-world datasets listed above can be cloned for extended training
    and evaluation.  The scripts below automate cloning.
  - The model architecture (HGT with GATv2Conv) operates on *graph features*,
    not raw Solidity text.  Features include node type, visibility, payable
    status, call patterns, state mutation, arithmetic operations, etc.

DOWNLOAD LINKS:
  SmartBugs Curated:  https://github.com/smartbugs/smartbugs-curated
  SolidiFI Benchmark: https://github.com/DependableSystemsLab/SolidiFI-benchmark
  SB Wild:            https://github.com/smartbugs/smartbugs-wild

Run:
    python data/download_datasets.py
"""

import os
import subprocess
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent

DATASETS = {
    "smartbugs-curated": {
        "url": "https://github.com/smartbugs/smartbugs-curated.git",
        "description": "143 labeled vulnerable Solidity contracts across 10 vulnerability categories",
        "download_link": "https://github.com/smartbugs/smartbugs-curated",
        "size": "~5 MB",
        "categories": [
            "Reentrancy", "Access Control", "Arithmetic",
            "Unchecked Low Level Calls", "Denial of Service",
            "Bad Randomness", "Front Running", "Time Manipulation",
            "Short Addresses", "Other"
        ],
    },
    "solidifi-benchmark": {
        "url": "https://github.com/DependableSystemsLab/SolidiFI-benchmark.git",
        "description": "9,369 smart contracts with systematically injected vulnerabilities (7 types)",
        "download_link": "https://github.com/DependableSystemsLab/SolidiFI-benchmark",
        "size": "~200 MB",
        "categories": [
            "Reentrancy", "Timestamp Dependency", "Unhandled Exceptions",
            "integer_overflow", "use_tx-origin", "unchecked_low_level_calls",
            "TOD"
        ],
    },
    "smartbugs-wild": {
        "url": "https://github.com/smartbugs/smartbugs-wild.git",
        "description": "47,518 unique Solidity contracts collected from Etherscan (for in-the-wild evaluation)",
        "download_link": "https://github.com/smartbugs/smartbugs-wild",
        "size": "~1.5 GB",
        "categories": ["Real-world Etherscan contracts (unlabeled)"],
    },
}


def clone_dataset(name: str, info: dict) -> bool:
    """Clone a dataset repository into the data directory."""
    dest = DATA_DIR / name
    if dest.exists():
        print(f"  [SKIP] {name} already exists at {dest}")
        return True

    print(f"  [CLONE] {name}")
    print(f"          URL:  {info['url']}")
    print(f"          Size: {info['size']}")
    print(f"          Desc: {info['description']}")

    try:
        subprocess.run(
            ["git", "clone", "--depth=1", info["url"], str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  [OK]   Cloned to {dest}")
        return True
    except FileNotFoundError:
        print(f"  [ERROR] 'git' command not found. Install Git first.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Clone failed: {e.stderr.strip()}")
        return False


def print_dataset_info():
    """Display information about all datasets."""
    print("=" * 70)
    print("  HCPG-GNN — Dataset Information")
    print("=" * 70)
    print()
    for name, info in DATASETS.items():
        print(f"  {name}")
        print(f"    URL:        {info['download_link']}")
        print(f"    Size:       {info['size']}")
        print(f"    Contracts:  {info['description']}")
        print(f"    Categories: {', '.join(info['categories'][:5])}")
        exists = (DATA_DIR / name).exists()
        print(f"    Status:     {'Downloaded' if exists else 'Not downloaded'}")
        print()

    print("  HOW DATASETS ARE USED:")
    print("  -" * 35)
    print("  The model trains on SYNTHETIC HCPG feature vectors generated")
    print("  by `models/train_model.py`. These feature vectors simulate")
    print("  the graph properties (node types, edge patterns, vulnerability")
    print("  signatures) that would be extracted from real smart contracts.")
    print()
    print("  The real datasets above are used for:")
    print("    1. Extended training with real Solidity parsed via solc")
    print("    2. Evaluation / benchmarking against known vulnerable contracts")
    print("    3. Cross-validation with other tools (Slither, Mythril)")
    print()


def main():
    print_dataset_info()

    if "--download" in sys.argv:
        print("  Downloading datasets...")
        print("  " + "-" * 50)
        for name, info in DATASETS.items():
            if name == "smartbugs-wild" and "--all" not in sys.argv:
                print(f"  [SKIP] {name} (1.5 GB, use --all to include)")
                continue
            clone_dataset(name, info)
        print()
        print("  Done! Datasets saved to:", DATA_DIR)
    else:
        print("  To download datasets, run:")
        print("    python data/download_datasets.py --download")
        print("    python data/download_datasets.py --download --all  (includes 1.5GB wild dataset)")


if __name__ == "__main__":
    main()
