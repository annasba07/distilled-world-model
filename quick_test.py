#!/usr/bin/env python3
"""
Quick test of toy world fallback only
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_toy_world():
    """Test toy world fallback"""
    print("Testing Toy World Fallback...")

    try:
        from src.inference.toy_world import ToyWorldSimulator

        # Create toy world
        toy_world = ToyWorldSimulator()
        print("  [OK] Toy world created")

        # Create deterministic world
        state = toy_world.create_state("demo world", 42)
        print(f"  [OK] Initial state: Player {state.player}, Goal {state.goal}")

        # Render initial frame
        frame = toy_world.render(state)
        print(f"  [OK] Frame rendered: {frame.shape} pixels")

        # Test step
        new_state, new_frame, metrics = toy_world.step(state, 1)  # Move up
        print(f"  [OK] Step completed: moved={metrics['moved']}")

        return True

    except Exception as e:
        print(f"  [FAIL] Toy world error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_toy_world()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)