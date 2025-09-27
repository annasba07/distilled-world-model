#!/usr/bin/env python3
"""
Demo Script for Enhanced World Model Features

This script demonstrates the key enhancements in a simple, interactive way
without requiring trained models or complex setup.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class EnhancedSystemDemo:
    """Interactive demo of enhanced features"""

    def __init__(self):
        self.demo_running = True

    async def demo_memory_management(self):
        """Demo memory-aware session management"""
        print("\n* DEMO: Memory-Aware Session Management")
        print("-" * 50)

        from src.utils.memory_manager import MemoryAwareSessionManager

        # Create memory manager
        manager = MemoryAwareSessionManager(max_sessions=3, ttl_seconds=5)

        print("Starting memory monitoring...")
        await manager.start_monitoring()

        # Create several sessions to show management
        for i in range(5):
            session_data = {
                "frames_generated": i * 10,
                "created_at": time.time(),
                "prompt": f"Demo session {i}"
            }

            success = manager.create_session(f"demo_session_{i}", session_data)
            print(f"  Session {i}: {'[OK] Created' if success else '[FAIL] Rejected'}")

            # Show memory report
            report = manager.get_memory_report()
            print(f"    Sessions: {report['sessions']['session_count']}")
            print(f"    Memory pressure: {report['pressure']:.2f}")

            await asyncio.sleep(0.5)

        print("\nWaiting for TTL cleanup (5 seconds)...")
        await asyncio.sleep(6)

        final_report = manager.get_memory_report()
        print(f"After cleanup: {final_report['sessions']['session_count']} sessions remaining")

        await manager.stop_monitoring()
        return True

    async def demo_health_monitoring(self):
        """Demo adaptive health monitoring"""
        print("\n* DEMO: Health Monitoring & Adaptive Quality")
        print("-" * 50)

        from src.utils.adaptive_quality import HealthMonitor, QualityLevel

        monitor = HealthMonitor(check_interval=1.0)

        print("Starting health monitoring...")
        await monitor.start_monitoring()

        # Simulate different performance scenarios
        scenarios = [
            ("Good Performance", [(0.05, 60.0), (0.04, 65.0), (0.06, 58.0)]),
            ("Degraded Performance", [(0.15, 25.0), (0.18, 20.0), (0.16, 22.0)]),
            ("Poor Performance", [(0.3, 10.0), (0.4, 8.0), (0.35, 9.0)]),
            ("Recovery", [(0.08, 45.0), (0.06, 55.0), (0.05, 60.0)])
        ]

        for scenario_name, metrics in scenarios:
            print(f"\nSimulating: {scenario_name}")

            for inference_time, fps in metrics:
                monitor.record_inference(inference_time, fps)
                await asyncio.sleep(0.2)

            status = monitor.get_current_status()
            print(f"  Health: {status['health_status']}")
            print(f"  Quality: {status['quality_level']}")
            print(f"  Score: {status['performance_score']:.2f}")

        await monitor.stop_monitoring()
        return True

    async def demo_error_recovery(self):
        """Demo error recovery system"""
        print("\n* DEMO: Error Recovery System")
        print("-" * 50)

        from src.utils.error_recovery import ErrorRecoveryManager

        recovery = ErrorRecoveryManager()

        # Simulate different types of errors
        test_errors = [
            RuntimeError("CUDA out of memory"),
            FileNotFoundError("Model checkpoint not found"),
            asyncio.TimeoutError("Request timeout"),
            ValueError("Invalid input shape"),
            MemoryError("System out of memory")
        ]

        for i, error in enumerate(test_errors):
            print(f"\nError {i+1}: {type(error).__name__}: {error}")

            # Attempt recovery
            recovery_result = await recovery.handle_error(
                error,
                session_id=f"demo_session_{i}",
                operation="demo_operation"
            )

            if recovery_result:
                action = recovery_result.get('action', 'unknown')
                status = recovery_result.get('status', 'unknown')
                print(f"  [OK] Recovery: {status} -> {action}")
            else:
                print(f"  [FAIL] Recovery failed")

        # Show recovery statistics
        stats = recovery.get_error_statistics()
        print(f"\nRecovery Stats:")
        print(f"  Total errors: {stats['total_errors']}")
        print(f"  Recovery rate: {stats['recovery_rate']:.1%}")
        print(f"  Categories: {stats['category_breakdown']}")

        return True

    async def demo_progressive_loading(self):
        """Demo progressive model loading"""
        print("\n* DEMO: Progressive Model Loading")
        print("-" * 50)

        from src.api.model_loading import ProgressiveModelLoader

        loader = ProgressiveModelLoader(device='cpu')

        # Simulate loading progress
        print("Simulating model loading with progress updates...")

        class MockProgressCallback:
            def __init__(self):
                self.last_progress = 0.0

            def __call__(self, stage: str, progress: float):
                if progress - self.last_progress >= 0.1 or progress == 1.0:
                    bar_length = 30
                    filled = int(bar_length * progress)
                    bar = "#" * filled + "." * (bar_length - filled)
                    print(f"  {stage:<25} [{bar}] {progress*100:5.1f}%")
                    self.last_progress = progress

        callback = MockProgressCallback()

        # Simulate loading stages
        stages = [
            ("Validating checkpoint", 0.1),
            ("Loading checkpoint data", 0.2),
            ("Parsing checkpoint format", 0.3),
            ("Loading VQ-VAE weights", 0.5),
            ("Loading dynamics weights", 0.7),
            ("Optimizing model", 0.9),
            ("Loading complete", 1.0)
        ]

        for stage, progress in stages:
            callback(stage, progress)
            await asyncio.sleep(0.3)

        print("[OK] Progressive loading demo complete!")
        return True

    async def demo_toy_world_fallback(self):
        """Demo toy world as ultimate fallback"""
        print("\n* DEMO: Toy World Fallback System")
        print("-" * 50)

        from src.inference.toy_world import ToyWorldSimulator

        toy_world = ToyWorldSimulator()

        # Create deterministic world
        print("Creating toy world with seed=42...")
        state = toy_world.create_state("demo world", 42)
        print(f"  Player position: {state.player}")
        print(f"  Goal position: {state.goal}")

        # Render initial frame
        frame = toy_world.render(state)
        print(f"  Rendered frame: {frame.shape} pixels")

        # Simulate game actions
        actions = [
            (0, "stay"),
            (1, "up"),
            (4, "right"),
            (4, "right"),
            (2, "down"),
            (3, "left")
        ]

        print("\nSimulating gameplay:")
        for action_id, action_name in actions:
            new_state, new_frame, metrics = toy_world.step(state, action_id)

            print(f"  Action: {action_name:<5} -> Player: {new_state.player}, "
                  f"Moved: {metrics['moved']}, Goal reached: {metrics['reached_goal']}")

            state = new_state

        print("Toy world provides reliable fallback experience!")
        return True

    async def demo_smart_batching(self):
        """Demo smart batching system"""
        print("\n* DEMO: Smart Batching System")
        print("-" * 50)

        from src.inference.smart_batching import SmartBatcher, RequestPriority

        # Create batcher
        batcher = SmartBatcher(max_batch_size=4, max_wait_time=100.0)

        # Mock batch processor
        async def mock_processor(batch):
            await asyncio.sleep(0.1)  # Simulate processing
            return [f"result_{req.request_id}" for req in batch]

        batcher.set_batch_processor(mock_processor)
        await batcher.start_processing()

        print("Submitting requests with different priorities...")

        # Submit requests with different priorities
        request_tasks = []
        for i in range(8):
            priority = [RequestPriority.LOW, RequestPriority.NORMAL,
                       RequestPriority.HIGH, RequestPriority.URGENT][i % 4]

            task = asyncio.create_task(
                batcher.submit_request(
                    session_id=f"session_{i}",
                    data={"operation": "test", "value": i},
                    priority=priority
                )
            )
            request_tasks.append((i, priority, task))
            print(f"  Request {i}: {priority.name} priority")

            await asyncio.sleep(0.02)  # Small delay between requests

        # Wait for all results
        print("\nProcessing batched requests...")
        for i, priority, task in request_tasks:
            try:
                result = await task
                print(f"  [OK] Request {i} ({priority.name}): {result}")
            except Exception as e:
                print(f"  [FAIL] Request {i} failed: {e}")

        # Show batching statistics
        stats = batcher.get_statistics()
        print(f"\nBatching Stats:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Avg batch size: {stats['avg_batch_size']:.1f}")
        print(f"  Avg processing time: {stats['avg_processing_time']*1000:.1f}ms")

        await batcher.stop_processing()
        return True

    async def run_interactive_demo(self):
        """Run interactive demo menu"""
        demos = [
            ("Memory Management", self.demo_memory_management),
            ("Health Monitoring", self.demo_health_monitoring),
            ("Error Recovery", self.demo_error_recovery),
            ("Progressive Loading", self.demo_progressive_loading),
            ("Toy World Fallback", self.demo_toy_world_fallback),
            ("Smart Batching", self.demo_smart_batching),
        ]

        while self.demo_running:
            print("\n* Enhanced World Model Feature Demo")
            print("=" * 50)
            print("Choose a demo to run:")

            for i, (name, _) in enumerate(demos, 1):
                print(f"  {i}. {name}")

            print(f"  {len(demos)+1}. Run All Demos")
            print(f"  0. Exit")

            try:
                choice = input("\nEnter your choice: ").strip()

                if choice == "0":
                    print("Thanks for trying the enhanced features!")
                    break
                elif choice == str(len(demos)+1):
                    print("\nRunning all demos...")
                    for name, demo_func in demos:
                        print(f"\nStarting: {name}")
                        try:
                            success = await demo_func()
                            if success:
                                print(f"[OK] {name} completed successfully!")
                            else:
                                print(f"[FAIL] {name} encountered issues")
                        except Exception as e:
                            print(f"[FAIL] {name} failed: {e}")

                        input("\nPress Enter to continue...")
                elif choice.isdigit() and 1 <= int(choice) <= len(demos):
                    name, demo_func = demos[int(choice)-1]
                    print(f"\nRunning: {name}")
                    try:
                        success = await demo_func()
                        if success:
                            print(f"\n[OK] {name} completed successfully!")
                        else:
                            print(f"\n[FAIL] {name} encountered issues")
                    except Exception as e:
                        print(f"\n[FAIL] {name} failed: {e}")

                    input("\nPress Enter to continue...")
                else:
                    print("Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\n\nDemo interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\nDemo ended. Goodbye!")
                break


async def main():
    """Main demo function"""
    print("Welcome to the Enhanced World Model Demo!")
    print("\nThis demo shows the new features without requiring trained models.")
    print("All demos use mock data and show the enhanced functionality.")

    demo = EnhancedSystemDemo()
    await demo.run_interactive_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()