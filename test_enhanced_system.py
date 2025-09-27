#!/usr/bin/env python3
"""
Integration Test Script for Enhanced World Model System

This script tests the enhanced system components step-by-step to identify
and help fix integration issues before running the full system.
"""

import sys
import asyncio
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all enhanced modules can be imported"""
    print("* Testing imports...")

    try:
        # Test basic imports
        import torch
        import numpy as np
        import psutil
        print("  [OK] Basic dependencies (torch, numpy, psutil)")

        # Test original modules
        from src.inference.engine import BatchedInferenceEngine
        print("  [OK] Original inference engine")

        from src.models.dynamics import DynamicsModel
        print("  [OK] Original dynamics model")

        # Test enhanced modules
        from src.utils.memory_manager import MemoryAwareSessionManager
        print("  [OK] Memory manager")

        from src.utils.adaptive_quality import HealthMonitor
        print("  [OK] Health monitor")

        from src.utils.error_recovery import ErrorRecoveryManager
        print("  [OK] Error recovery")

        from src.inference.predictive_engine import PredictiveInferenceEngine
        print("  [OK] Predictive engine")

        from src.api.model_loading import ProgressiveModelLoader
        print("  [OK] Progressive loader")

        return True

    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        print(f"     Traceback: {traceback.format_exc()}")
        return False
    except Exception as e:
        print(f"  [FAIL] Unexpected error: {e}")
        return False


def test_memory_manager():
    """Test memory management system"""
    print("\n* Testing memory manager...")

    try:
        from src.utils.memory_manager import MemoryAwareSessionManager

        # Create memory manager
        manager = MemoryAwareSessionManager(max_sessions=5, ttl_seconds=30)
        print("  [OK] Memory manager created")

        # Test session creation
        test_session = {"frames_generated": 1, "created_at": "2024-01-01"}
        success = manager.create_session("test_session", test_session)
        print(f"  [OK] Session creation: {success}")

        # Test memory report
        report = manager.get_memory_report()
        print(f"  [OK] Memory report: {report['sessions']['session_count']} sessions")

        return True

    except Exception as e:
        print(f"  [FAIL] Memory manager error: {e}")
        return False


def test_health_monitor():
    """Test health monitoring system"""
    print("\n* Testing health monitor...")

    try:
        from src.utils.adaptive_quality import HealthMonitor, QualityLevel

        # Create health monitor
        monitor = HealthMonitor(check_interval=1.0)
        print("  [OK] Health monitor created")

        # Test metrics recording
        monitor.record_inference(0.1, 25.0)  # 100ms inference, 25 FPS
        print("  [OK] Metrics recorded")

        # Test status
        status = monitor.get_current_status()
        print(f"  [OK] Health status: {status.get('health_status', 'unknown')}")

        # Test quality settings
        from src.utils.adaptive_quality import QualitySettings
        quality = QualitySettings.from_level(QualityLevel.HIGH)
        print(f"  [OK] Quality settings: {quality.resolution}px, attention={quality.enable_attention}")

        return True

    except Exception as e:
        print(f"  [FAIL] Health monitor error: {e}")
        return False


def test_error_recovery():
    """Test error recovery system"""
    print("\n* Testing error recovery...")

    try:
        from src.utils.error_recovery import ErrorRecoveryManager, ErrorCategory, ErrorSeverity

        # Create recovery manager
        recovery = ErrorRecoveryManager()
        print("  [OK] Error recovery manager created")

        # Test error classification
        test_error = RuntimeError("CUDA out of memory")
        category, severity = recovery.classifier.classify_error(test_error)
        print(f"  [OK] Error classification: {category.value}, {severity.value}")

        # Test statistics
        stats = recovery.get_error_statistics()
        print(f"  [OK] Error stats: {stats['total_errors']} total errors")

        return True

    except Exception as e:
        print(f"  [FAIL] Error recovery error: {e}")
        return False


async def test_progressive_loader():
    """Test progressive model loading"""
    print("\n* Testing progressive loader...")

    try:
        from src.api.model_loading import ProgressiveModelLoader

        # Create loader
        loader = ProgressiveModelLoader(device='cpu')  # Use CPU for testing
        print("  [OK] Progressive loader created")

        # Test job management
        jobs = loader.list_jobs()
        print(f"  [OK] Job listing: {len(jobs['active'])} active, {len(jobs['completed'])} completed")

        return True

    except Exception as e:
        print(f"  [FAIL] Progressive loader error: {e}")
        return False


def test_toy_world_fallback():
    """Test toy world simulator as fallback"""
    print("\n* Testing toy world fallback...")

    try:
        from src.inference.toy_world import ToyWorldSimulator

        # Create toy world
        toy_world = ToyWorldSimulator()
        print("  [OK] Toy world created")

        # Create initial state
        state = toy_world.create_state("test prompt", 42)
        print(f"  [OK] Initial state: {state.player} player position")

        # Render frame
        frame = toy_world.render(state)
        print(f"  [OK] Frame rendered: {frame.shape} shape")

        # Test step
        new_state, new_frame, metrics = toy_world.step(state, 1)  # Move up
        print(f"  [OK] Step completed: moved={metrics['moved']}")

        return True

    except Exception as e:
        print(f"  [FAIL] Toy world error: {e}")
        return False


async def test_enhanced_server_imports():
    """Test enhanced server can be imported"""
    print("\n* Testing enhanced server...")

    try:
        # This will test if the enhanced server can at least be imported
        from src.api.enhanced_server import app
        print("  [OK] Enhanced server imported successfully")

        # Test app creation
        print(f"  [OK] FastAPI app: {app.title}")

        return True

    except Exception as e:
        print(f"  [FAIL] Enhanced server error: {e}")
        print(f"     This is expected if there are integration issues")
        return False


async def main():
    """Run all integration tests"""
    print("Enhanced World Model Integration Test")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Memory Manager", test_memory_manager),
        ("Health Monitor", test_health_monitor),
        ("Error Recovery", test_error_recovery),
        ("Progressive Loader", test_progressive_loader),
        ("Toy World Fallback", test_toy_world_fallback),
        ("Enhanced Server", test_enhanced_server_imports),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n[FAIL] {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\nAll tests passed! The enhanced system should work.")
        print("Next steps:")
        print("   1. pip install -r requirements.txt")
        print("   2. python -m src.api.enhanced_server")
        print("   3. Visit http://localhost:8000/health")
    else:
        print("\nSome tests failed. Check the errors above.")
        print("Common fixes:")
        print("   1. Install missing dependencies: pip install psutil")
        print("   2. Check Python path and import issues")
        print("   3. Ensure original codebase is working")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)