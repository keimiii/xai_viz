#!/usr/bin/env python3
"""
Download WikiChurches dataset from Zenodo (record 5166987).

Features:
- Progress bar for large file downloads
- Resume support for interrupted downloads
- File size verification after download
"""

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Zenodo record ID for WikiChurches dataset
ZENODO_RECORD_ID = "5166987"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

# Dataset files with expected sizes in bytes (from Zenodo API)
DATASET_FILES = {
    "images.zip": 12_389_751_014,  # 11.5 GB
    "models.zip": 262_910_967,  # 250.7 MB
    "image_meta.json": 37_671_550,  # 35.9 MB
    "churches.json": 3_377_060,  # 3.2 MB
    "building_parts.json": 303_897,  # 303.9 kB
    "labels.zip": 252_533,  # 252.5 kB
    "datasheet.pdf": 85_108,  # 85.1 kB
    "LICENSE": 14_759,  # 14.8 kB
    "README.md": 13_391,  # 13.4 kB
    "style_names.txt": 3_460,  # 3.5 kB
    "parent_child_rel.txt": 925,  # 925 B
}


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_file(filename: str, output_dir: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from Zenodo with progress bar and resume support.

    Args:
        filename: Name of the file to download
        output_dir: Directory to save the file
        chunk_size: Size of chunks to download at a time

    Returns:
        True if download successful, False otherwise
    """
    url = f"{BASE_URL}/{filename}?download=1"
    output_path = output_dir / filename
    expected_size = DATASET_FILES.get(filename)

    # Check if file already exists and is complete
    if output_path.exists():
        existing_size = output_path.stat().st_size
        if expected_size and existing_size == expected_size:
            print(f"✓ {filename} already downloaded ({format_size(existing_size)})")
            return True
        elif existing_size > 0:
            print(f"  Resuming {filename} from {format_size(existing_size)}...")
    else:
        existing_size = 0

    # Set up headers for resume support
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        # Handle response codes
        if response.status_code == 416:  # Range not satisfiable - file complete
            print(f"✓ {filename} already complete")
            return True
        elif response.status_code not in (200, 206):
            print(f"✗ Failed to download {filename}: HTTP {response.status_code}")
            return False

        # Get total size from Content-Range or Content-Length
        if response.status_code == 206:  # Partial content
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                total_size = int(content_range.split("/")[-1])
            else:
                total_size = existing_size + int(
                    response.headers.get("Content-Length", 0)
                )
        else:
            total_size = int(response.headers.get("Content-Length", 0))
            existing_size = 0  # Server doesn't support resume, start fresh

        # Open file in append mode if resuming, write mode otherwise
        mode = "ab" if existing_size > 0 and response.status_code == 206 else "wb"

        with open(output_path, mode) as f, tqdm(
            total=total_size,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename,
            ncols=80,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading {filename}: {e}")
        return False


def verify_downloads(
    output_dir: Path, files_to_check: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """
    Verify downloaded files against expected sizes.

    Args:
        output_dir: Directory containing downloaded files
        files_to_check: List of files to verify (defaults to all files)

    Returns:
        Tuple of (verified_files, failed_files)
    """
    verified = []
    failed = []

    if files_to_check is None:
        files_to_check = list(DATASET_FILES.keys())

    for filename in files_to_check:
        expected_size = DATASET_FILES[filename]
        filepath = output_dir / filename
        if not filepath.exists():
            failed.append(f"{filename} (missing)")
        else:
            actual_size = filepath.stat().st_size
            # Allow 5% tolerance for size differences
            if abs(actual_size - expected_size) / expected_size < 0.05:
                verified.append(filename)
            else:
                failed.append(
                    f"{filename} (size mismatch: {format_size(actual_size)} vs expected {format_size(expected_size)})"
                )

    return verified, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download WikiChurches dataset from Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_wikichurches.py                    # Download all files
  python download_wikichurches.py -o ./data          # Download to ./data directory
  python download_wikichurches.py --files churches.json image_meta.json
  python download_wikichurches.py --exclude images.zip models.zip
        """,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        choices=list(DATASET_FILES.keys()),
        help="Download only specific files",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        choices=list(DATASET_FILES.keys()),
        help="Exclude specific files from download",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads without downloading",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available files and exit",
    )

    args = parser.parse_args()

    # List files and exit
    if args.list:
        print("WikiChurches Dataset Files:")
        print("-" * 50)
        total_size = 0
        for filename, size in DATASET_FILES.items():
            print(f"  {filename:<25} {format_size(size):>10}")
            total_size += size
        print("-" * 50)
        print(f"  {'Total':<25} {format_size(total_size):>10}")
        return 0

    # Determine which files to download
    files_to_download = args.files or list(DATASET_FILES.keys())

    if args.exclude:
        files_to_download = [f for f in files_to_download if f not in args.exclude]

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Verify only mode
    if args.verify_only:
        print(f"Verifying downloads in {args.output}...")
        verified, failed = verify_downloads(args.output)
        print(f"\n✓ Verified: {len(verified)}/{len(DATASET_FILES)} files")
        if failed:
            print("✗ Failed verification:")
            for f in failed:
                print(f"  - {f}")
            return 1
        return 0

    # Calculate total download size
    total_size = sum(DATASET_FILES[f] for f in files_to_download)
    print("WikiChurches Dataset Download")
    print(f"Output directory: {args.output.absolute()}")
    print(f"Files to download: {len(files_to_download)}")
    print(f"Total size: {format_size(total_size)}")
    print("-" * 50)

    # Download files
    success_count = 0
    for filename in files_to_download:
        if download_file(filename, args.output):
            success_count += 1

    print("-" * 50)
    print(f"Downloaded: {success_count}/{len(files_to_download)} files")

    # Verify downloads (only check files we attempted to download)
    print("\nVerifying downloads...")
    verified, failed = verify_downloads(args.output, files_to_download)

    if failed:
        print("✗ Some files failed verification:")
        for f in failed:
            print(f"  - {f}")
        return 1

    print(f"✓ All {len(verified)} files verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
