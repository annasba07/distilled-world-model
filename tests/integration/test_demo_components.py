#!/usr/bin/env python3
"""
Quick test of individual demo components without interactive input
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from demo_enhanced_features import EnhancedSystemDemo

async def test_individual_components():
    """Test individual demo components"""
    demo = EnhancedSystemDemo()

    print("Testing Enhanced System Demo Components")
    print("=" * 50)

    components = [
        ("Memory Management", demo.demo_memory_management),
        ("Health Monitoring", demo.demo_health_monitoring),
        ("Error Recovery", demo.demo_error_recovery),
        ("Progressive Loading", demo.demo_progressive_loading),
        ("Toy World Fallback", demo.demo_toy_world_fallback),
    ]

    results = {}

    for name, component in components:
        print(f"\n* Testing {name}...")
        try:
            success = await component()
            results[name] = success
            print(f"  [{'OK' if success else 'FAIL'}] {name} completed")
        except Exception as e:
            print(f"  [FAIL] {name} failed: {e}")
            results[name] = False

    print("\n" + "=" * 50)
    print("Demo Component Results:")

    passed = 0
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} components passed ({passed/total*100:.1f}%)")
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(test_individual_components())
    sys.exit(0 if success else 1)