"""
Smart Batching System for Multi-User Performance Optimization

This module implements intelligent request batching to maximize throughput
for multiple concurrent users while maintaining low latency for individual requests.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np
import torch
from datetime import datetime

from ..utils.logging import get_logger


logger = get_logger(__name__)


class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class BatchRequest:
    """Individual request in a batch"""
    request_id: str
    session_id: str
    timestamp: float
    priority: RequestPriority
    data: Dict[str, Any]
    response_future: asyncio.Future
    timeout: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if request has expired"""
        if self.timeout is None:
            return False
        return time.time() - self.timestamp > self.timeout

    def get_age(self) -> float:
        """Get request age in seconds"""
        return time.time() - self.timestamp


@dataclass
class BatchStats:
    """Statistics for batch processing"""
    batch_id: str
    request_count: int
    processing_time: float
    queue_wait_time: float
    avg_priority: float
    timestamp: datetime = field(default_factory=datetime.now)


class BatchingStrategy(Enum):
    """Batching strategies"""
    TIME_BASED = "time_based"           # Wait for time window
    SIZE_BASED = "size_based"           # Wait for batch size
    ADAPTIVE = "adaptive"               # Adapt based on load
    PRIORITY_AWARE = "priority_aware"   # Consider request priorities


class SmartBatcher:
    """Intelligent request batcher with multiple strategies"""

    def __init__(self,
                 max_batch_size: int = 8,
                 max_wait_time: float = 50.0,  # milliseconds
                 min_batch_size: int = 1,
                 strategy: BatchingStrategy = BatchingStrategy.ADAPTIVE):

        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time / 1000.0  # Convert to seconds
        self.min_batch_size = min_batch_size
        self.strategy = strategy

        # Request queues
        self.pending_requests: Dict[RequestPriority, deque] = {
            priority: deque() for priority in RequestPriority
        }
        self.processing_batches: Dict[str, List[BatchRequest]] = {}

        # Processing state
        self.is_processing = False
        self.batch_processor: Optional[Callable] = None
        self.batch_counter = 0

        # Performance tracking
        self.batch_stats: deque = deque(maxlen=100)
        self.total_requests = 0
        self.total_batches = 0

        # Adaptive parameters
        self.current_load = 0.0
        self.avg_processing_time = 0.1
        self.target_latency = 0.1  # 100ms target

        # Background tasks
        self.batch_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None

    def set_batch_processor(self, processor: Callable[[List[BatchRequest]], Any]):
        """Set the function that processes batches"""
        self.batch_processor = processor

    async def submit_request(self, session_id: str, data: Dict[str, Any],
                           priority: RequestPriority = RequestPriority.NORMAL,
                           timeout: Optional[float] = None) -> Any:
        """Submit a request for batched processing"""
        request_id = f"{session_id}_{int(time.time() * 1000000)}"

        # Create request
        request = BatchRequest(
            request_id=request_id,
            session_id=session_id,
            timestamp=time.time(),
            priority=priority,
            data=data,
            response_future=asyncio.Future(),
            timeout=timeout
        )

        # Add to appropriate queue
        self.pending_requests[priority].append(request)
        self.total_requests += 1

        # Trigger batch processing if needed
        if not self.is_processing:
            asyncio.create_task(self._process_batches())

        # Wait for result
        try:
            result = await request.response_future
            return result
        except asyncio.CancelledError:
            # Remove from queue if cancelled
            self._remove_request(request)
            raise

    def _remove_request(self, request: BatchRequest):
        """Remove a request from pending queues"""
        for queue in self.pending_requests.values():
            try:
                queue.remove(request)
            except ValueError:
                pass  # Not in this queue

    async def start_processing(self):
        """Start background batch processing"""
        if self.batch_task is not None:
            return

        self.batch_task = asyncio.create_task(self._batch_processing_loop())
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Smart batching started")

    async def stop_processing(self):
        """Stop background batch processing"""
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Smart batching stopped")

    async def _batch_processing_loop(self):
        """Main batch processing loop"""
        while True:
            try:
                await self._process_batches()
                await asyncio.sleep(0.001)  # 1ms sleep to prevent busy waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.01)

    async def _process_batches(self):
        """Process pending requests in batches"""
        if self.is_processing:
            return

        batch = self._form_batch()
        if not batch:
            return

        self.is_processing = True
        try:
            await self._execute_batch(batch)
        finally:
            self.is_processing = False

    def _form_batch(self) -> List[BatchRequest]:
        """Form a batch based on current strategy"""
        if self.strategy == BatchingStrategy.PRIORITY_AWARE:
            return self._form_priority_batch()
        elif self.strategy == BatchingStrategy.ADAPTIVE:
            return self._form_adaptive_batch()
        elif self.strategy == BatchingStrategy.SIZE_BASED:
            return self._form_size_batch()
        else:  # TIME_BASED
            return self._form_time_batch()

    def _form_priority_batch(self) -> List[BatchRequest]:
        """Form batch considering request priorities"""
        batch = []

        # First, add urgent requests
        while (len(batch) < self.max_batch_size and
               self.pending_requests[RequestPriority.URGENT]):
            batch.append(self.pending_requests[RequestPriority.URGENT].popleft())

        # Then high priority requests
        while (len(batch) < self.max_batch_size and
               self.pending_requests[RequestPriority.HIGH]):
            batch.append(self.pending_requests[RequestPriority.HIGH].popleft())

        # Fill with normal priority if space available
        while (len(batch) < self.max_batch_size and
               self.pending_requests[RequestPriority.NORMAL]):
            batch.append(self.pending_requests[RequestPriority.NORMAL].popleft())

        # Add low priority only if no other requests
        if len(batch) == 0:
            while (len(batch) < self.max_batch_size and
                   self.pending_requests[RequestPriority.LOW]):
                batch.append(self.pending_requests[RequestPriority.LOW].popleft())

        return batch

    def _form_adaptive_batch(self) -> List[BatchRequest]:
        """Form batch adaptively based on current load"""
        # Calculate current load
        total_pending = sum(len(queue) for queue in self.pending_requests.values())
        self.current_load = total_pending / self.max_batch_size

        # Adaptive parameters
        if self.current_load > 2.0:  # High load
            max_wait = self.max_wait_time * 0.5
            target_size = self.max_batch_size
        elif self.current_load > 1.0:  # Medium load
            max_wait = self.max_wait_time * 0.75
            target_size = max(self.min_batch_size, self.max_batch_size // 2)
        else:  # Low load
            max_wait = self.max_wait_time
            target_size = self.min_batch_size

        return self._form_batch_with_params(target_size, max_wait)

    def _form_size_batch(self) -> List[BatchRequest]:
        """Form batch based on size threshold"""
        total_pending = sum(len(queue) for queue in self.pending_requests.values())
        if total_pending >= self.max_batch_size:
            return self._form_batch_with_params(self.max_batch_size, 0)
        return []

    def _form_time_batch(self) -> List[BatchRequest]:
        """Form batch based on time threshold"""
        # Check if any request is old enough
        oldest_request = None
        oldest_age = 0

        for queue in self.pending_requests.values():
            if queue:
                request = queue[0]  # Oldest in queue
                age = request.get_age()
                if age > oldest_age:
                    oldest_age = age
                    oldest_request = request

        if oldest_request and oldest_age >= self.max_wait_time:
            return self._form_batch_with_params(self.max_batch_size, 0)
        return []

    def _form_batch_with_params(self, target_size: int, max_wait: float) -> List[BatchRequest]:
        """Form batch with specific parameters"""
        batch = []

        # Use priority-aware selection
        for priority in [RequestPriority.URGENT, RequestPriority.HIGH,
                        RequestPriority.NORMAL, RequestPriority.LOW]:
            queue = self.pending_requests[priority]
            while len(batch) < target_size and queue:
                request = queue.popleft()
                if not request.is_expired():
                    batch.append(request)
                else:
                    # Mark expired request as failed
                    if not request.response_future.done():
                        request.response_future.set_exception(
                            asyncio.TimeoutError("Request expired in queue")
                        )

        return batch

    async def _execute_batch(self, batch: List[BatchRequest]):
        """Execute a batch of requests"""
        if not batch or not self.batch_processor:
            return

        batch_id = f"batch_{self.batch_counter}"
        self.batch_counter += 1

        start_time = time.time()
        queue_wait_time = max(req.get_age() for req in batch)

        try:
            # Process the batch
            results = await self._safe_batch_execution(batch)

            processing_time = time.time() - start_time

            # Distribute results
            for request, result in zip(batch, results):
                if not request.response_future.done():
                    if isinstance(result, Exception):
                        request.response_future.set_exception(result)
                    else:
                        request.response_future.set_result(result)

            # Record statistics
            stats = BatchStats(
                batch_id=batch_id,
                request_count=len(batch),
                processing_time=processing_time,
                queue_wait_time=queue_wait_time,
                avg_priority=sum(req.priority.value for req in batch) / len(batch)
            )
            self.batch_stats.append(stats)
            self.total_batches += 1

            # Update adaptive parameters
            self.avg_processing_time = (
                self.avg_processing_time * 0.9 + processing_time * 0.1
            )

            logger.debug(f"Processed batch {batch_id}: {len(batch)} requests in {processing_time:.3f}s")

        except Exception as e:
            logger.error(f"Batch execution error: {e}")
            # Mark all requests as failed
            for request in batch:
                if not request.response_future.done():
                    request.response_future.set_exception(e)

    async def _safe_batch_execution(self, batch: List[BatchRequest]) -> List[Any]:
        """Safely execute batch with timeout protection"""
        try:
            # Execute with timeout
            timeout = max(req.timeout or 10.0 for req in batch)
            results = await asyncio.wait_for(
                self.batch_processor(batch),
                timeout=timeout
            )
            return results
        except asyncio.TimeoutError:
            logger.warning(f"Batch processing timeout after {timeout}s")
            return [asyncio.TimeoutError("Batch processing timeout") for _ in batch]
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [e for _ in batch]

    async def _monitoring_loop(self):
        """Monitor and adjust batching parameters"""
        while True:
            try:
                await asyncio.sleep(10.0)  # Monitor every 10 seconds
                self._adjust_parameters()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

    def _adjust_parameters(self):
        """Adjust batching parameters based on performance"""
        if len(self.batch_stats) < 5:
            return

        recent_stats = list(self.batch_stats)[-10:]
        avg_processing_time = np.mean([s.processing_time for s in recent_stats])
        avg_wait_time = np.mean([s.queue_wait_time for s in recent_stats])

        # Adjust max_wait_time based on performance
        if avg_wait_time > self.target_latency:
            self.max_wait_time = max(0.01, self.max_wait_time * 0.9)
        elif avg_processing_time < self.target_latency:
            self.max_wait_time = min(0.2, self.max_wait_time * 1.1)

        logger.debug(f"Adjusted max_wait_time to {self.max_wait_time:.3f}s")

    def get_statistics(self) -> Dict[str, Any]:
        """Get batching statistics"""
        if not self.batch_stats:
            return {"status": "no_data"}

        recent_stats = list(self.batch_stats)[-20:]

        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": np.mean([s.request_count for s in recent_stats]),
            "avg_processing_time": np.mean([s.processing_time for s in recent_stats]),
            "avg_queue_wait_time": np.mean([s.queue_wait_time for s in recent_stats]),
            "current_load": self.current_load,
            "pending_requests": {
                priority.name: len(queue)
                for priority, queue in self.pending_requests.items()
            },
            "strategy": self.strategy.value,
            "parameters": {
                "max_batch_size": self.max_batch_size,
                "max_wait_time": self.max_wait_time,
                "min_batch_size": self.min_batch_size
            }
        }

    def clear_statistics(self):
        """Clear all statistics"""
        self.batch_stats.clear()
        self.total_requests = 0
        self.total_batches = 0


class MultiUserInferenceManager:
    """Manages multi-user inference with smart batching"""

    def __init__(self, base_engine, max_batch_size: int = 8):
        self.base_engine = base_engine
        self.batcher = SmartBatcher(
            max_batch_size=max_batch_size,
            strategy=BatchingStrategy.ADAPTIVE
        )

        # Set up batch processor
        self.batcher.set_batch_processor(self._process_inference_batch)

    async def start(self):
        """Start multi-user inference manager"""
        await self.batcher.start_processing()

    async def stop(self):
        """Stop multi-user inference manager"""
        await self.batcher.stop_processing()

    async def step_session_batched(self, session_id: str, action: int,
                                 priority: RequestPriority = RequestPriority.NORMAL) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Step session with batched processing"""
        request_data = {
            "action": action,
            "operation": "step_session"
        }

        result = await self.batcher.submit_request(
            session_id=session_id,
            data=request_data,
            priority=priority,
            timeout=5.0
        )

        return result

    async def _process_inference_batch(self, batch: List[BatchRequest]) -> List[Any]:
        """Process a batch of inference requests"""
        results = []

        try:
            # Group requests by operation type
            step_requests = [req for req in batch if req.data.get("operation") == "step_session"]

            # Process step requests in parallel (if possible) or sequentially
            for request in step_requests:
                try:
                    session_id = request.session_id
                    action = request.data["action"]

                    # Use the base engine to process
                    frame, metrics = self.base_engine.step_session(session_id, action)
                    results.append((frame, metrics))

                except Exception as e:
                    logger.error(f"Failed to process request {request.request_id}: {e}")
                    results.append(e)

            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [e for _ in batch]

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "batching_stats": self.batcher.get_statistics(),
            "engine_stats": getattr(self.base_engine, 'get_performance_stats', lambda: {})()
        }


# Factory function
def create_multi_user_manager(base_engine, **kwargs):
    """Create multi-user inference manager"""
    return MultiUserInferenceManager(base_engine, **kwargs)