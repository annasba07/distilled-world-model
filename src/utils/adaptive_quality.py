"""
Health Monitoring and Adaptive Quality System

This module implements real-time system health monitoring and automatically
adjusts quality settings to maintain optimal user experience under varying
system loads and resource constraints.
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import torch
import psutil
from datetime import datetime, timedelta
from collections import deque

from ..utils.logging import get_logger


logger = get_logger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    EXCELLENT = "excellent"  # >90% performance
    GOOD = "good"           # 70-90% performance
    FAIR = "fair"           # 50-70% performance
    POOR = "poor"           # 30-50% performance
    CRITICAL = "critical"   # <30% performance


class QualityLevel(Enum):
    """Quality level presets"""
    ULTRA = "ultra"         # Maximum quality
    HIGH = "high"           # High quality
    MEDIUM = "medium"       # Balanced quality/performance
    LOW = "low"             # Performance-optimized
    MINIMAL = "minimal"     # Emergency fallback


@dataclass
class QualitySettings:
    """Quality configuration settings"""
    resolution: int = 256
    enable_attention: bool = True
    enable_tensorrt: bool = True
    use_fp16: bool = True
    batch_size: int = 4
    prediction_horizon: int = 3
    max_context_length: int = 32
    enable_fast_scan: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'resolution': self.resolution,
            'enable_attention': self.enable_attention,
            'enable_tensorrt': self.enable_tensorrt,
            'use_fp16': self.use_fp16,
            'batch_size': self.batch_size,
            'prediction_horizon': self.prediction_horizon,
            'max_context_length': self.max_context_length,
            'enable_fast_scan': self.enable_fast_scan
        }

    @classmethod
    def from_level(cls, level: QualityLevel) -> 'QualitySettings':
        """Create quality settings from preset level"""
        presets = {
            QualityLevel.ULTRA: cls(
                resolution=256,
                enable_attention=True,
                enable_tensorrt=True,
                use_fp16=True,
                batch_size=4,
                prediction_horizon=3,
                max_context_length=32,
                enable_fast_scan=True
            ),
            QualityLevel.HIGH: cls(
                resolution=256,
                enable_attention=True,
                enable_tensorrt=True,
                use_fp16=True,
                batch_size=3,
                prediction_horizon=2,
                max_context_length=24,
                enable_fast_scan=True
            ),
            QualityLevel.MEDIUM: cls(
                resolution=192,
                enable_attention=True,
                enable_tensorrt=True,
                use_fp16=True,
                batch_size=2,
                prediction_horizon=2,
                max_context_length=16,
                enable_fast_scan=True
            ),
            QualityLevel.LOW: cls(
                resolution=128,
                enable_attention=False,
                enable_tensorrt=True,
                use_fp16=True,
                batch_size=1,
                prediction_horizon=1,
                max_context_length=8,
                enable_fast_scan=True
            ),
            QualityLevel.MINIMAL: cls(
                resolution=64,
                enable_attention=False,
                enable_tensorrt=False,
                use_fp16=False,
                batch_size=1,
                prediction_horizon=1,
                max_context_length=4,
                enable_fast_scan=False
            )
        }
        return presets.get(level, presets[QualityLevel.MEDIUM])


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)

    # Latency metrics (milliseconds)
    avg_inference_time: float = 0.0
    p95_inference_time: float = 0.0
    p99_inference_time: float = 0.0

    # Throughput metrics
    frames_per_second: float = 0.0
    sessions_per_second: float = 0.0

    # Resource utilization (0.0 to 1.0)
    gpu_memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    cpu_usage: float = 0.0
    ram_usage: float = 0.0

    # Error rates
    error_rate: float = 0.0
    timeout_rate: float = 0.0

    # Queue metrics
    pending_requests: int = 0
    queue_wait_time: float = 0.0

    def get_overall_score(self) -> float:
        """Calculate overall performance score (0.0 to 1.0)"""
        # Latency score (lower is better)
        target_latency = 100.0  # 100ms target
        latency_score = max(0.0, 1.0 - (self.avg_inference_time / target_latency))

        # Resource score (lower usage is better)
        resource_score = 1.0 - max(self.gpu_memory_usage, self.cpu_usage, self.ram_usage)

        # Throughput score
        target_fps = 30.0
        throughput_score = min(1.0, self.frames_per_second / target_fps)

        # Error score (lower is better)
        error_score = max(0.0, 1.0 - (self.error_rate + self.timeout_rate))

        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # latency, resource, throughput, error
        scores = [latency_score, resource_score, throughput_score, error_score]

        return sum(w * s for w, s in zip(weights, scores))


class MetricsCollector:
    """Collects and aggregates performance metrics"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.fps_measurements = deque(maxlen=window_size)
        self.error_count = 0
        self.timeout_count = 0
        self.request_count = 0
        self.start_time = time.time()

    def record_inference(self, inference_time: float):
        """Record inference time"""
        self.inference_times.append(inference_time)

    def record_fps(self, fps: float):
        """Record FPS measurement"""
        self.fps_measurements.append(fps)

    def record_error(self):
        """Record an error occurrence"""
        self.error_count += 1

    def record_timeout(self):
        """Record a timeout occurrence"""
        self.timeout_count += 1

    def record_request(self):
        """Record a request"""
        self.request_count += 1

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        metrics = PerformanceMetrics()

        # Latency metrics
        if self.inference_times:
            times = list(self.inference_times)
            metrics.avg_inference_time = np.mean(times) * 1000  # Convert to ms
            metrics.p95_inference_time = np.percentile(times, 95) * 1000
            metrics.p99_inference_time = np.percentile(times, 99) * 1000

        # FPS metrics
        if self.fps_measurements:
            metrics.frames_per_second = np.mean(list(self.fps_measurements))

        # Resource metrics
        try:
            if torch.cuda.is_available():
                metrics.gpu_memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                # GPU utilization would require nvidia-ml-py

            memory = psutil.virtual_memory()
            metrics.ram_usage = memory.percent / 100.0
            metrics.cpu_usage = psutil.cpu_percent() / 100.0

        except Exception as e:
            logger.warning(f"Failed to collect resource metrics: {e}")

        # Error rates
        if self.request_count > 0:
            metrics.error_rate = self.error_count / self.request_count
            metrics.timeout_rate = self.timeout_count / self.request_count

        return metrics

    def reset(self):
        """Reset all metrics"""
        self.inference_times.clear()
        self.fps_measurements.clear()
        self.error_count = 0
        self.timeout_count = 0
        self.request_count = 0
        self.start_time = time.time()


class AdaptiveQualityController:
    """Controls adaptive quality based on performance metrics"""

    def __init__(self, initial_level: QualityLevel = QualityLevel.HIGH):
        self.current_level = initial_level
        self.current_settings = QualitySettings.from_level(initial_level)
        self.target_fps = 30.0
        self.target_latency = 100.0  # milliseconds
        self.adjustment_history = deque(maxlen=10)
        self.last_adjustment = time.time()
        self.min_adjustment_interval = 10.0  # seconds

    def should_adjust_quality(self, metrics: PerformanceMetrics) -> bool:
        """Determine if quality should be adjusted"""
        # Don't adjust too frequently
        if time.time() - self.last_adjustment < self.min_adjustment_interval:
            return False

        # Check if performance is significantly off target
        fps_ratio = metrics.frames_per_second / self.target_fps
        latency_ratio = metrics.avg_inference_time / self.target_latency

        # Adjust if FPS is too low or latency too high
        if fps_ratio < 0.7 or latency_ratio > 1.5:
            return True

        # Adjust if resources are overloaded
        if metrics.gpu_memory_usage > 0.9 or metrics.cpu_usage > 0.9:
            return True

        # Adjust if error rate is high
        if metrics.error_rate > 0.1:
            return True

        return False

    def calculate_target_level(self, metrics: PerformanceMetrics) -> QualityLevel:
        """Calculate target quality level based on metrics"""
        score = metrics.get_overall_score()

        # Map score to quality level
        if score >= 0.9:
            return QualityLevel.ULTRA
        elif score >= 0.7:
            return QualityLevel.HIGH
        elif score >= 0.5:
            return QualityLevel.MEDIUM
        elif score >= 0.3:
            return QualityLevel.LOW
        else:
            return QualityLevel.MINIMAL

    def adjust_quality(self, metrics: PerformanceMetrics) -> bool:
        """Adjust quality based on performance metrics"""
        if not self.should_adjust_quality(metrics):
            return False

        target_level = self.calculate_target_level(metrics)

        if target_level != self.current_level:
            old_level = self.current_level
            self.current_level = target_level
            self.current_settings = QualitySettings.from_level(target_level)
            self.last_adjustment = time.time()

            self.adjustment_history.append({
                'timestamp': datetime.now(),
                'from_level': old_level.value,
                'to_level': target_level.value,
                'score': metrics.get_overall_score(),
                'reason': self._get_adjustment_reason(metrics)
            })

            logger.info(f"Quality adjusted from {old_level.value} to {target_level.value}")
            return True

        return False

    def _get_adjustment_reason(self, metrics: PerformanceMetrics) -> str:
        """Get human-readable reason for quality adjustment"""
        reasons = []

        if metrics.avg_inference_time > self.target_latency * 1.5:
            reasons.append("high_latency")
        if metrics.frames_per_second < self.target_fps * 0.7:
            reasons.append("low_fps")
        if metrics.gpu_memory_usage > 0.9:
            reasons.append("high_gpu_memory")
        if metrics.cpu_usage > 0.9:
            reasons.append("high_cpu")
        if metrics.error_rate > 0.1:
            reasons.append("high_error_rate")

        return ", ".join(reasons) if reasons else "optimization"

    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """Get history of quality adjustments"""
        return list(self.adjustment_history)


class HealthMonitor:
    """Real-time system health monitoring"""

    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.metrics_collector = MetricsCollector()
        self.quality_controller = AdaptiveQualityController()
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        self.health_callbacks: List[Callable] = []
        self.quality_callbacks: List[Callable] = []
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.health_history = deque(maxlen=100)

    def add_health_callback(self, callback: Callable[[HealthStatus, PerformanceMetrics], None]):
        """Add callback for health status changes"""
        self.health_callbacks.append(callback)

    def add_quality_callback(self, callback: Callable[[QualitySettings], None]):
        """Add callback for quality setting changes"""
        self.quality_callbacks.append(callback)

    async def start_monitoring(self):
        """Start health monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")

    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect current metrics
                metrics = self.metrics_collector.get_current_metrics()
                self.current_metrics = metrics

                # Determine health status
                health_status = self._calculate_health_status(metrics)

                # Record in history
                self.health_history.append({
                    'timestamp': datetime.now(),
                    'status': health_status,
                    'score': metrics.get_overall_score()
                })

                # Check for quality adjustments
                quality_adjusted = self.quality_controller.adjust_quality(metrics)

                # Notify callbacks
                for callback in self.health_callbacks:
                    try:
                        callback(health_status, metrics)
                    except Exception as e:
                        logger.error(f"Health callback error: {e}")

                if quality_adjusted:
                    for callback in self.quality_callbacks:
                        try:
                            callback(self.quality_controller.current_settings)
                        except Exception as e:
                            logger.error(f"Quality callback error: {e}")

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    def _calculate_health_status(self, metrics: PerformanceMetrics) -> HealthStatus:
        """Calculate health status from metrics"""
        score = metrics.get_overall_score()

        if score >= 0.9:
            return HealthStatus.EXCELLENT
        elif score >= 0.7:
            return HealthStatus.GOOD
        elif score >= 0.5:
            return HealthStatus.FAIR
        elif score >= 0.3:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def record_inference(self, inference_time: float, fps: float):
        """Record inference metrics"""
        self.metrics_collector.record_inference(inference_time)
        self.metrics_collector.record_fps(fps)
        self.metrics_collector.record_request()

    def record_error(self):
        """Record error occurrence"""
        self.metrics_collector.record_error()

    def record_timeout(self):
        """Record timeout occurrence"""
        self.metrics_collector.record_timeout()

    def get_current_status(self) -> Dict[str, Any]:
        """Get current health status"""
        if not self.current_metrics:
            return {"status": "not_available"}

        health_status = self._calculate_health_status(self.current_metrics)

        return {
            "health_status": health_status.value,
            "quality_level": self.quality_controller.current_level.value,
            "performance_score": self.current_metrics.get_overall_score(),
            "metrics": {
                "avg_inference_time": self.current_metrics.avg_inference_time,
                "frames_per_second": self.current_metrics.frames_per_second,
                "gpu_memory_usage": self.current_metrics.gpu_memory_usage,
                "cpu_usage": self.current_metrics.cpu_usage,
                "error_rate": self.current_metrics.error_rate
            },
            "quality_settings": self.quality_controller.current_settings.to_dict(),
            "is_monitoring": self.is_monitoring
        }

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        status = self.get_current_status()

        # Add historical data
        if self.health_history:
            recent_scores = [h['score'] for h in list(self.health_history)[-10:]]
            status['trend'] = {
                'recent_avg_score': np.mean(recent_scores),
                'score_trend': 'improving' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else 'stable'
            }

        # Add adjustment history
        status['quality_adjustments'] = self.quality_controller.get_adjustment_history()

        return status

    def force_quality_level(self, level: QualityLevel):
        """Force specific quality level (override adaptive control)"""
        self.quality_controller.current_level = level
        self.quality_controller.current_settings = QualitySettings.from_level(level)

        # Notify callbacks
        for callback in self.quality_controller.quality_callbacks:
            try:
                callback(self.quality_controller.current_settings)
            except Exception as e:
                logger.error(f"Quality callback error: {e}")

        logger.info(f"Quality level forced to {level.value}")


# Global instance
global_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance"""
    return global_health_monitor