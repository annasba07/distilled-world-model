#!/usr/bin/env python3
"""
Quick script to adjust memory settings for high baseline RAM systems
"""

import sys
from pathlib import Path

def adjust_settings():
    """Adjust memory thresholds for systems with high baseline RAM usage"""

    # Adjust enhanced_server.py threshold
    server_file = Path("src/api/enhanced_server.py")
    if server_file.exists():
        content = server_file.read_text()
        # Change threshold from 0.8 to 0.95 (95% RAM)
        content = content.replace("if pressure > 0.8:", "if pressure > 0.95:")
        server_file.write_text(content)
        print("[OK] Adjusted enhanced_server.py memory threshold to 95%")

    # Adjust memory_manager.py calculation
    manager_file = Path("src/utils/memory_manager.py")
    if manager_file.exists():
        content = manager_file.read_text()
        # Adjust pressure calculation to be more lenient
        content = content.replace(
            "return min(memory_percent / 100",
            "return min(max(0, (memory_percent - 80) / 20)"  # Map 80-100% to 0-1
        )
        manager_file.write_text(content)
        print("[OK] Adjusted memory_manager.py pressure calculation")

    print("\nMemory settings adjusted for high baseline systems!")
    print("Restart the server to apply changes.")

if __name__ == "__main__":
    adjust_settings()