#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script sets up and runs a local Qdrant instance without Docker.
It uses the qdrant-client package's embedded server functionality.

To use:
1. Run this script: python run_local_qdrant.py
2. Connect to Qdrant from your application using: 
   client = QdrantClient(url="http://localhost:6333")
"""

import os
import sys
import time
import subprocess
import signal
import platform
import argparse
from pathlib import Path

def check_qdrant_server_installed():
    """Check if qdrant-server is installed via pip."""
    try:
        subprocess.check_output(["pip", "show", "qdrant-server"])
        return True
    except subprocess.CalledProcessError:
        return False

def install_qdrant_server():
    """Install qdrant-server via pip."""
    print("Installing qdrant-server package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qdrant-server"])
    print("qdrant-server installed successfully.")

def run_qdrant_server(config_path=None, storage_path=None):
    """Run the Qdrant server with the specified configuration."""
    cmd = ["qdrant"]
    
    if config_path and os.path.exists(config_path):
        cmd.extend(["--config-path", config_path])
    elif storage_path:
        # Create storage path if it doesn't exist
        storage_dir = Path(storage_path)
        storage_dir.mkdir(exist_ok=True, parents=True)
        cmd.extend(["--storage-path", storage_path])
    
    print(f"Starting Qdrant server with command: {' '.join(cmd)}")
    
    # Run as a subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Setup signal handling to properly shut down the server
    def signal_handler(sig, frame):
        print("\nShutting down Qdrant server...")
        process.terminate()
        try:
            process.wait(timeout=5)  # Wait for clean shutdown
        except subprocess.TimeoutExpired:
            process.kill()  # Force kill if it doesn't terminate
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Monitor the process output
    try:
        for line in process.stdout:
            print(line, end='')
            
            # Check if server has started
            if "Ready to accept connections" in line:
                print("\nQdrant server is running and ready to accept connections!")
                print("Connect using: client = QdrantClient(url=\"http://localhost:6333\")")
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    
    return process

def check_qdrant_running():
    """Check if Qdrant is already running on port 6333."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', 6333)) == 0

def main():
    parser = argparse.ArgumentParser(description='Run a local Qdrant server')
    parser.add_argument('--config', help='Path to the Qdrant config file')
    parser.add_argument('--storage', default='./qdrant_storage', 
                        help='Path to the storage directory (default: ./qdrant_storage)')
    args = parser.parse_args()
    
    # Check if Qdrant is already running
    if check_qdrant_running():
        print("Qdrant is already running on port 6333.")
        print("If you want to start a new instance, please stop the existing one first.")
        return

    # Check and install qdrant-server if needed
    if not check_qdrant_server_installed():
        print("qdrant-server is not installed. Installing now...")
        install_qdrant_server()
    
    # Run the server
    process = run_qdrant_server(config_path=args.config, storage_path=args.storage)
    
    # Keep the script running
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down Qdrant server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

if __name__ == "__main__":
    main() 