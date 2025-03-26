#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script downloads the Qdrant binary for Windows and extracts it.
"""

import os
import sys
import requests
import zipfile
import platform
import subprocess
from pathlib import Path
from tqdm import tqdm

# Latest release as of this script's creation
QDRANT_VERSION = "v1.11.0"

def get_download_url():
    """Get the appropriate download URL based on the platform."""
    system = platform.system().lower()
    
    if system == "windows":
        return f"https://github.com/qdrant/qdrant/releases/download/{QDRANT_VERSION}/qdrant-{QDRANT_VERSION}-windows-x86_64.zip"
    elif system == "darwin":  # macOS
        return f"https://github.com/qdrant/qdrant/releases/download/{QDRANT_VERSION}/qdrant-{QDRANT_VERSION}-darwin-x86_64.tar.gz"
    elif system == "linux":
        return f"https://github.com/qdrant/qdrant/releases/download/{QDRANT_VERSION}/qdrant-{QDRANT_VERSION}-linux-x86_64.tar.gz"
    else:
        print(f"Unsupported platform: {system}")
        print("Please download Qdrant manually from: https://github.com/qdrant/qdrant/releases")
        sys.exit(1)

def download_file(url, destination):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"Downloading Qdrant from {url}")
    print(f"Saving to {destination}")
    
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
    
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Downloaded size does not match expected size")
        return False
    
    return True

def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"Extracting to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete")

def main():
    # Setup paths
    qdrant_dir = Path("./qdrant_binary")
    os.makedirs(qdrant_dir, exist_ok=True)
    
    # Download URL
    url = get_download_url()
    filename = url.split("/")[-1]
    download_path = qdrant_dir / filename
    
    # Download file
    if not download_file(url, download_path):
        print("Download failed. Please try again or download manually.")
        return
    
    # Extract file
    if filename.endswith(".zip"):
        extract_zip(download_path, qdrant_dir)
    else:
        # For tar.gz files, we'd need to use tarfile module
        print(f"Please extract {download_path} manually.")
        return
    
    # For Windows, try to find the executable
    if platform.system().lower() == "windows":
        exe_paths = list(qdrant_dir.glob("**/qdrant.exe"))
        if exe_paths:
            qdrant_exe = exe_paths[0]
            print(f"\nQdrant executable found at: {qdrant_exe}")
            print("\nTo run Qdrant:")
            print(f"  {qdrant_exe} --storage-path ./qdrant_storage")
            
            # Create a batch file for easy starting
            batch_file = qdrant_dir / "run_qdrant.bat"
            with open(batch_file, "w") as f:
                f.write(f'@echo off\n')
                f.write(f'echo Starting Qdrant...\n')
                f.write(f'"{qdrant_exe}" --storage-path ./qdrant_storage\n')
            
            print(f"\nOr simply run the batch file: {batch_file}")
        else:
            print("\nCouldn't find qdrant.exe in the extracted files.")
            print(f"Please check the contents of {qdrant_dir} manually.")
    
    print("\nAfter starting Qdrant, you can connect to it using:")
    print("  client = QdrantClient(url=\"http://localhost:6333\")")

if __name__ == "__main__":
    main() 