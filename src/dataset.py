"""
Dataset Management Module for Music Genre Classification (FMA Small)

This module handles:
1. Downloading fma_metadata.zip and fma_small.zip with resume support and progress bars.
2. Extracting archives into data/raw/.
3. Inspecting metadata, genre distributions, and verifying file integrity.
"""

import os
import sys
import zipfile
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Constants and URLs
METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
AUDIO_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"

RAW_DATA_DIR = Path("data/raw")
METADATA_DIR = RAW_DATA_DIR / "fma_metadata"
AUDIO_DIR = RAW_DATA_DIR / "fma_small"


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024, max_retries: int = 25) -> Path:
    """
    Downloads a large file with a progress bar, HTTP Range resume support,
    and automatic reconnection if network drops.
    """
    import time
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    
    retries = 0
    while retries < max_retries:
        try:
            existing_size = temp_path.stat().st_size if temp_path.exists() else 0
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
                print(f"\nResuming download from {existing_size / (1024*1024):.1f} MB (byte {existing_size:,})...")

            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # 416 means Range Not Satisfiable (already fully downloaded)
            if response.status_code == 416:
                print("File is already complete on server range check.")
                break

            response.raise_for_status()

            if response.status_code == 206:
                total_size = existing_size + int(response.headers.get("content-length", 0))
                mode = "ab"
            else:
                existing_size = 0
                total_size = int(response.headers.get("content-length", 0))
                mode = "wb"

            filename = dest_path.name
            print(f"Downloading {filename} ({existing_size/(1024*1024):.1f} / {total_size/(1024*1024):.1f} MB)...")
            
            with open(temp_path, mode) as f, tqdm(
                total=total_size,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
                miniters=1
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Download finished successfully without exception
            break

        except (requests.exceptions.RequestException, ConnectionResetError, Exception) as e:
            retries += 1
            print(f"\n[Auto-Resume] Connection interrupted ({type(e).__name__}). Reconnecting ({retries}/{max_retries}) in 3s...")
            time.sleep(3)

    if temp_path.exists():
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
        
    print(f"\nDownload complete: {dest_path}")
    return dest_path


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """
    Extracts a zip file with progress tracking.
    """
    print(f"\nExtracting {zip_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.infolist()
        for member in tqdm(members, desc=f"Unzipping {zip_path.name}", unit="file"):
            zf.extract(member, extract_to)
            
    print(f"Extracted to {extract_to}")
    return extract_to


def get_audio_path(audio_dir: Path, track_id: int) -> Path:
    """
    Returns the file path for a track ID in the FMA dataset structure.
    Example: track_id 2 -> audio_dir / '000' / '000002.mp3'
    """
    tid_str = f"{track_id:06d}"
    folder = tid_str[:3]
    return audio_dir / folder / f"{tid_str}.mp3"


def inspect_metadata(metadata_dir: Path = METADATA_DIR, audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    """
    Inspects FMA tracks metadata:
    - Filters to the 'small' subset (~8,000 tracks).
    - Checks genre distribution across the 8 genres.
    - Validates that corresponding MP3 files exist on disk.
    
    Returns:
        DataFrame containing track_id, genre, and valid audio file paths.
    """
    tracks_csv = metadata_dir / "tracks.csv"
    if not tracks_csv.exists():
        # Sometimes unzipping puts it in metadata_dir / fma_metadata / tracks.csv
        alt_path = metadata_dir / "fma_metadata" / "tracks.csv"
        if alt_path.exists():
            tracks_csv = alt_path
        else:
            raise FileNotFoundError(f"tracks.csv not found at {tracks_csv}")

    print("\nReading FMA metadata (tracks.csv)...")
    # FMA tracks.csv has 3 header rows: column category, subfield, and metric
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    print(f"Total tracks in full metadata: {len(tracks):,}")

    # Filter to FMA Small subset
    small_mask = tracks[('set', 'subset')] == 'small'
    small_tracks = tracks[small_mask].copy()
    print(f"Tracks in 'small' subset: {len(small_tracks):,}")

    # Extract genre labels
    genres = small_tracks[('track', 'genre_top')]
    
    # Check for missing labels
    missing_genres = genres.isna().sum()
    print(f"Tracks with missing genre labels: {missing_genres}")

    # Inspect Genre Distribution
    genre_counts = genres.value_counts()
    print("\n" + "=" * 45)
    print(" Genre Distribution in FMA Small:")
    print("=" * 45)
    for genre, count in genre_counts.items():
        print(f"  {genre:<16}: {count:>5} tracks")
    print("=" * 45)
    print(f"  Total           : {len(genres):>5} tracks across {len(genre_counts)} genres\n")

    # Cross-reference with physical audio files on disk
    print("Verifying audio files on disk...")
    # Handle possible nested folder (audio_dir / fma_small / ...)
    actual_audio_dir = audio_dir
    if not (actual_audio_dir / "000").exists() and (actual_audio_dir / "fma_small" / "000").exists():
        actual_audio_dir = actual_audio_dir / "fma_small"

    matched_records = []
    missing_files = 0

    for track_id, genre in genres.items():
        audio_file = get_audio_path(actual_audio_dir, track_id)
        if audio_file.exists():
            matched_records.append({
                "track_id": track_id,
                "genre": genre,
                "file_path": str(audio_file)
            })
        else:
            missing_files += 1

    df_matched = pd.DataFrame(matched_records)
    print(f"Successfully matched audio files : {len(df_matched):,}")
    print(f"Missing or unfound audio files   : {missing_files}")

    return df_matched


def setup_and_prepare_dataset(download_audio: bool = True):
    """
    Master pipeline for Step 2:
    1. Download & extract metadata.
    2. Download & extract audio files (if requested).
    3. Inspect metadata and verify matching audio tracks.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Handle Metadata (~342 MB)
    metadata_zip = RAW_DATA_DIR / "fma_metadata.zip"
    tracks_csv = METADATA_DIR / "tracks.csv"
    alt_tracks_csv = METADATA_DIR / "fma_metadata" / "tracks.csv"
    
    if not tracks_csv.exists() and not alt_tracks_csv.exists():
        if not metadata_zip.exists():
            download_file(METADATA_URL, metadata_zip)
        extract_zip(metadata_zip, METADATA_DIR)
    else:
        print(f"Metadata already exists at {METADATA_DIR}.")

    # 2. Handle Audio Files (~7.2 GB)
    audio_zip = RAW_DATA_DIR / "fma_small.zip"
    if download_audio:
        # Check if audio folders already exist (e.g., folder '000')
        audio_extracted = (AUDIO_DIR / "000").exists() or (AUDIO_DIR / "fma_small" / "000").exists() or (RAW_DATA_DIR / "fma_small" / "000").exists()
        if not audio_extracted:
            if not audio_zip.exists():
                download_file(AUDIO_URL, audio_zip)
            # fma_small.zip contains a root folder named 'fma_small', so extract into RAW_DATA_DIR
            extract_zip(audio_zip, RAW_DATA_DIR)
        else:
            print(f"Audio files already exist in {RAW_DATA_DIR}.")

    # 3. Inspect dataset
    df_matched = inspect_metadata()
    return df_matched


if __name__ == "__main__":
    setup_and_prepare_dataset()
