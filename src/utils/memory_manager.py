"""
Memory-Aware Session Management for Reliable Performance

This module implements intelligent memory management to prevent OOM crashes
and ensure stable long-running sessions through automatic cleanup and
resource monitoring.
"""

import asyncio
import gc
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import torch
import psutil
from datetime import datetime, timedelta


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_max_allocated_gb: float
    system_ram_gb: float
    system_ram_percent: float
    session_count: int
    timestamp: datetime


@dataclass
class SessionMetrics:
    """Per-session resource usage metrics"""
    session_id: str
    created_at: datetime
    last_accessed: datetime
    frames_generated: int
    memory_usage_mb: float
    avg_inference_time: float
    priority_score: float


class MemoryTracker:
    """Tracks memory usage patterns and predicts resource needs"""

    def __init__(self, history_size: int = 100):
        self.history: List[MemoryStats] = []
        self.history_size = history_size
        self.lock = threading.Lock()

    def get_current_stats(self, session_count: int = 0) -> MemoryStats:
        """Get current memory statistics"""
        gpu_stats = self._get_gpu_memory_stats()
        system_stats = self._get_system_memory_stats()

        return MemoryStats(
            gpu_allocated_gb=gpu_stats['allocated'],
            gpu_reserved_gb=gpu_stats['reserved'],
            gpu_max_allocated_gb=gpu_stats['max_allocated'],
            system_ram_gb=system_stats['used'],
            system_ram_percent=system_stats['percent'],
            session_count=session_count,
            timestamp=datetime.now()
        )

    def _get_gpu_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics in GB"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9
            }
        return {'allocated': 0.0, 'reserved': 0.0, 'max_allocated': 0.0}

    def _get_system_memory_stats(self) -> Dict[str, float]:
        """Get system RAM statistics"""
        memory = psutil.virtual_memory()
        return {
            'used': memory.used / 1e9,
            'percent': memory.percent
        }

    def record_stats(self, session_count: int):
        """Record current memory statistics"""
        with self.lock:
            stats = self.get_current_stats(session_count)
            self.history.append(stats)

            # Maintain history size
            if len(self.history) > self.history_size:
                self.history.pop(0)

    def predict_memory_pressure(self) -> float:
        """Predict memory pressure (0.0 = low, 1.0 = critical)"""
        if not self.history:
            return 0.0

        latest = self.history[-1]

        # GPU memory pressure
        gpu_pressure = min(latest.gpu_allocated_gb / 4.0, 1.0)  # 4GB target

        # System memory pressure
        ram_pressure = latest.system_ram_percent / 100.0

        # Trend analysis (is memory usage increasing?)
        if len(self.history) >= 5:
            recent_gpu = [s.gpu_allocated_gb for s in self.history[-5:]]
            trend_pressure = max(0.0, (recent_gpu[-1] - recent_gpu[0]) / 2.0)
        else:
            trend_pressure = 0.0

        # Combined pressure score
        return min(max(gpu_pressure, ram_pressure) + trend_pressure, 1.0)

    def get_memory_report(self) -> Dict[str, Any]:
        """Get detailed memory usage report"""
        if not self.history:
            return {'status': 'no_data'}

        latest = self.history[-1]
        pressure = self.predict_memory_pressure()

        return {
            'current_gpu_gb': latest.gpu_allocated_gb,
            'current_ram_percent': latest.system_ram_percent,
            'pressure_score': pressure,
            'status': self._get_status_from_pressure(pressure),
            'session_count': latest.session_count,
            'recommendations': self._get_recommendations(pressure)
        }

    def _get_status_from_pressure(self, pressure: float) -> str:
        """Convert pressure score to status"""
        if pressure < 0.3:
            return 'healthy'
        elif pressure < 0.6:
            return 'moderate'
        elif pressure < 0.8:
            return 'high'
        else:
            return 'critical'

    def _get_recommendations(self, pressure: float) -> List[str]:
        """Get recommendations based on pressure"""
        recommendations = []

        if pressure > 0.6:
            recommendations.append('Consider cleaning up old sessions')
        if pressure > 0.7:
            recommendations.append('Reduce batch size or resolution')
        if pressure > 0.8:
            recommendations.append('Stop new session creation')
        if pressure > 0.9:
            recommendations.append('Emergency cleanup required')

        return recommendations


class LRUSessionCache:
    """LRU cache for session management with automatic cleanup"""

    def __init__(self, max_sessions: int = 50, ttl_seconds: int = 3600):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.sessions: OrderedDict[str, Any] = OrderedDict()
        self.session_metrics: Dict[str, SessionMetrics] = {}
        self.lock = threading.Lock()
        self.cleanup_callbacks: List[Callable[[str], None]] = []

    def add_cleanup_callback(self, callback: Callable[[str], None]):
        """Add callback to be called when session is cleaned up"""
        self.cleanup_callbacks.append(callback)

    def put(self, session_id: str, session_data: Any, memory_usage_mb: float = 0.0):
        """Add or update session in cache"""
        with self.lock:
            # Remove if exists (for LRU reordering)
            if session_id in self.sessions:
                del self.sessions[session_id]

            # Add to end (most recent)
            self.sessions[session_id] = session_data

            # Update metrics
            now = datetime.now()
            if session_id in self.session_metrics:
                metrics = self.session_metrics[session_id]
                metrics.last_accessed = now
                metrics.frames_generated += 1
                metrics.memory_usage_mb = memory_usage_mb
            else:
                self.session_metrics[session_id] = SessionMetrics(
                    session_id=session_id,
                    created_at=now,
                    last_accessed=now,
                    frames_generated=1,
                    memory_usage_mb=memory_usage_mb,
                    avg_inference_time=0.0,
                    priority_score=1.0
                )

            # Cleanup if necessary
            self._cleanup_if_needed()

    def get(self, session_id: str) -> Optional[Any]:
        """Get session from cache (updates LRU order)"""
        with self.lock:
            if session_id not in self.sessions:
                return None

            # Move to end (most recent)
            session_data = self.sessions[session_id]
            del self.sessions[session_id]
            self.sessions[session_id] = session_data

            # Update access time
            if session_id in self.session_metrics:
                self.session_metrics[session_id].last_accessed = datetime.now()

            return session_data

    def remove(self, session_id: str) -> bool:
        """Remove session from cache"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                self.session_metrics.pop(session_id, None)

                # Call cleanup callbacks
                for callback in self.cleanup_callbacks:
                    try:
                        callback(session_id)
                    except Exception as e:
                        print(f"Cleanup callback error for {session_id}: {e}")

                return True
            return False

    def _cleanup_if_needed(self):
        """Cleanup old or excess sessions"""
        now = datetime.now()

        # Remove expired sessions (TTL)
        expired_sessions = []
        for session_id, metrics in self.session_metrics.items():
            if (now - metrics.last_accessed).total_seconds() > self.ttl_seconds:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.remove(session_id)

        # Remove excess sessions (LRU)
        while len(self.sessions) > self.max_sessions:
            # Remove least recently used (first in OrderedDict)
            oldest_session_id = next(iter(self.sessions))
            self.remove(oldest_session_id)

    def cleanup_by_memory_pressure(self, pressure: float) -> int:
        """Cleanup sessions based on memory pressure"""
        if pressure < 0.7:
            return 0  # No cleanup needed

        # Calculate how many sessions to remove
        current_count = len(self.sessions)
        if pressure > 0.9:
            target_removal = max(1, current_count // 2)  # Remove half
        elif pressure > 0.8:
            target_removal = max(1, current_count // 3)  # Remove third
        else:
            target_removal = max(1, current_count // 5)  # Remove fifth

        # Sort sessions by priority (remove low priority first)
        sessions_by_priority = sorted(
            self.session_metrics.items(),
            key=lambda x: (x[1].priority_score, x[1].last_accessed)
        )

        removed_count = 0
        for session_id, _ in sessions_by_priority[:target_removal]:
            if self.remove(session_id):
                removed_count += 1

        return removed_count

    def update_session_priority(self, session_id: str, priority: float):
        """Update session priority (higher = more important)"""
        if session_id in self.session_metrics:
            self.session_metrics[session_id].priority_score = priority

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'session_count': len(self.sessions),
                'max_sessions': self.max_sessions,
                'ttl_seconds': self.ttl_seconds,
                'oldest_session_age': self._get_oldest_session_age(),
                'memory_usage_total': sum(m.memory_usage_mb for m in self.session_metrics.values())
            }

    def _get_oldest_session_age(self) -> float:
        """Get age of oldest session in seconds"""
        if not self.session_metrics:
            return 0.0

        now = datetime.now()
        oldest = min(metrics.created_at for metrics in self.session_metrics.values())
        return (now - oldest).total_seconds()


class MemoryAwareSessionManager:
    """Intelligent session manager with memory awareness"""

    def __init__(self, max_sessions: int = 50, ttl_seconds: int = 3600,
                 memory_check_interval: int = 30, max_memory_gb: float = 3.5):
        self.cache = LRUSessionCache(max_sessions, ttl_seconds)
        self.memory_tracker = MemoryTracker()
        self.max_memory_gb = max_memory_gb
        self.memory_check_interval = memory_check_interval

        # Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

        # Setup cleanup callback
        self.cache.add_cleanup_callback(self._cleanup_session_resources)

    async def start_monitoring(self):
        """Start background memory monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._memory_monitor_loop())

    async def stop_monitoring(self):
        """Stop background memory monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

    async def _memory_monitor_loop(self):
        """Background memory monitoring and cleanup"""
        while self.is_monitoring:
            try:
                # Record current stats
                self.memory_tracker.record_stats(len(self.cache.sessions))

                # Check memory pressure
                pressure = self.memory_tracker.predict_memory_pressure()

                if pressure > 0.7:
                    print(f"High memory pressure detected: {pressure:.2f}")
                    # Perform cleanup
                    removed = self.cache.cleanup_by_memory_pressure(pressure)
                    if removed > 0:
                        print(f"Cleaned up {removed} sessions due to memory pressure")
                        # Force garbage collection
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                await asyncio.sleep(self.memory_check_interval)

            except Exception as e:
                print(f"Memory monitoring error: {e}")
                await asyncio.sleep(self.memory_check_interval)

    def create_session(self, session_id: str, session_data: Any) -> bool:
        """Create new session with memory checks"""
        # Check if we should accept new sessions
        pressure = self.memory_tracker.predict_memory_pressure()

        if pressure > 0.8:
            print(f"Rejecting new session {session_id} due to high memory pressure: {pressure:.2f}")
            return False

        # Add to cache
        self.cache.put(session_id, session_data)
        return True

    def get_session(self, session_id: str) -> Optional[Any]:
        """Get session data"""
        return self.cache.get(session_id)

    def update_session(self, session_id: str, session_data: Any,
                      memory_usage_mb: float = 0.0, priority: float = 1.0):
        """Update session with resource usage info"""
        self.cache.put(session_id, session_data, memory_usage_mb)
        self.cache.update_session_priority(session_id, priority)

    def remove_session(self, session_id: str) -> bool:
        """Remove session manually"""
        return self.cache.remove(session_id)

    def _cleanup_session_resources(self, session_id: str):
        """Cleanup resources for a session"""
        try:
            # Clean up any session-specific GPU tensors
            if torch.cuda.is_available():
                # Clear any cached tensors for this session
                torch.cuda.empty_cache()

            print(f"Cleaned up resources for session {session_id}")
        except Exception as e:
            print(f"Error cleaning up session {session_id}: {e}")

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory and session report"""
        memory_report = self.memory_tracker.get_memory_report()
        cache_stats = self.cache.get_stats()

        return {
            'memory': memory_report,
            'sessions': cache_stats,
            'pressure': self.memory_tracker.predict_memory_pressure(),
            'monitoring_active': self.is_monitoring
        }

    def force_cleanup(self, target_sessions: int = 10) -> int:
        """Force cleanup to target number of sessions"""
        current_count = len(self.cache.sessions)
        if current_count <= target_sessions:
            return 0

        sessions_to_remove = current_count - target_sessions

        # Sort by priority and remove lowest priority sessions
        sessions_by_priority = sorted(
            self.cache.session_metrics.items(),
            key=lambda x: (x[1].priority_score, x[1].last_accessed)
        )

        removed_count = 0
        for session_id, _ in sessions_by_priority[:sessions_to_remove]:
            if self.cache.remove(session_id):
                removed_count += 1

        # Force garbage collection after cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return removed_count


# Helper function for easy integration
def create_memory_manager(**kwargs) -> MemoryAwareSessionManager:
    """Factory function to create memory manager with sensible defaults"""
    return MemoryAwareSessionManager(**kwargs)