#!/usr/bin/env python3
"""
Run enhanced server with memory protection override for development
"""

import os
import sys

# Override memory protection for development/testing
os.environ['DISABLE_MEMORY_PROTECTION'] = '1'
os.environ['MAX_MEMORY_THRESHOLD'] = '95'

# Run the enhanced server
sys.path.insert(0, 'src')
from api.enhanced_server import app
import uvicorn

if __name__ == "__main__":
    print("Starting enhanced server with memory override...")
    print("Memory protection: DISABLED for development")
    print("This allows testing on high-baseline RAM systems")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )