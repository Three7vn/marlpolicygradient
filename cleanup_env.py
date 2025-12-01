#!/usr/bin/env python3
"""Cleanup script to close any running Unity environments"""

import socket
import subprocess
import time

print("Cleaning up Unity Tennis environments...")

# Method 1: Kill any Tennis.app processes
try:
    result = subprocess.run(
        ["pkill", "-f", "Tennis.app"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ Killed existing Tennis.app processes")
        time.sleep(2)
    else:
        print(" No Tennis.app processes found")
except Exception as e:
    print(f"Note: {e}")

# Method 2: Check if port is free
port = 5004
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    result = s.connect_ex(('localhost', port))
    if result == 0:
        print(f" Port {port} is still in use")
        print("   Waiting 3 seconds...")
        time.sleep(3)
    else:
        print(f"✓ Port {port} is free")
    s.close()
except Exception as e:
    print(f"Port check: {e}")

print("\nCleanup complete. You can now restart the environment.")
