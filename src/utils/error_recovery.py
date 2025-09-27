"""
Comprehensive Error Recovery System for Reliable User Experience

This module implements a robust error recovery system with multiple fallback
strategies to ensure users always get a response, even when primary systems fail.
"""

import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np
import torch
from datetime import datetime, timedelta

from ..utils.logging import get_logger


logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Minor issues, no user impact
    MEDIUM = "medium"     # Some degradation, fallback used
    HIGH = "high"         # Significant issues, limited functionality
    CRITICAL = "critical" # Major failure, emergency fallback only


class ErrorCategory(Enum):
    """Categories of errors"""
    MEMORY = "memory"           # GPU/CPU memory issues
    MODEL = "model"             # Model loading/inference errors
    NETWORK = "network"         # Network connectivity issues
    HARDWARE = "hardware"       # GPU/hardware failures
    VALIDATION = "validation"   # Input validation errors
    TIMEOUT = "timeout"         # Operation timeouts
    UNKNOWN = "unknown"         # Unclassified errors


@dataclass
class ErrorContext:
    """Context information for an error"""
    error_type: type
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    session_id: Optional[str] = None
    operation: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


class ErrorClassifier:
    """Classifies errors by type and severity"""

    def __init__(self):
        self.classification_rules = {
            # Memory-related errors
            (RuntimeError, "out of memory"): (ErrorCategory.MEMORY, ErrorSeverity.HIGH),
            (RuntimeError, "CUDA out of memory"): (ErrorCategory.MEMORY, ErrorSeverity.HIGH),
            (MemoryError, ""): (ErrorCategory.MEMORY, ErrorSeverity.CRITICAL),

            # Model-related errors
            (FileNotFoundError, "checkpoint"): (ErrorCategory.MODEL, ErrorSeverity.HIGH),
            (RuntimeError, "model"): (ErrorCategory.MODEL, ErrorSeverity.MEDIUM),
            (KeyError, "state_dict"): (ErrorCategory.MODEL, ErrorSeverity.MEDIUM),

            # Hardware-related errors
            (RuntimeError, "CUDA"): (ErrorCategory.HARDWARE, ErrorSeverity.HIGH),
            (RuntimeError, "device"): (ErrorCategory.HARDWARE, ErrorSeverity.MEDIUM),

            # Validation errors
            (ValueError, ""): (ErrorCategory.VALIDATION, ErrorSeverity.LOW),
            (TypeError, ""): (ErrorCategory.VALIDATION, ErrorSeverity.LOW),

            # Timeout errors
            (asyncio.TimeoutError, ""): (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
            (TimeoutError, ""): (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
        }

    def classify_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify an error by category and severity"""
        error_type = type(error)
        error_message = str(error).lower()

        # Check specific rules first
        for (rule_type, rule_message), (category, severity) in self.classification_rules.items():
            if issubclass(error_type, rule_type) and rule_message in error_message:
                return category, severity

        # Default classification based on error type
        if issubclass(error_type, (MemoryError, RuntimeError)):
            return ErrorCategory.MEMORY, ErrorSeverity.HIGH
        elif issubclass(error_type, (FileNotFoundError, ImportError)):
            return ErrorCategory.MODEL, ErrorSeverity.MEDIUM
        elif issubclass(error_type, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION, ErrorSeverity.LOW
        else:
            return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM


class RecoveryStrategy(ABC):
    """Abstract base class for recovery strategies"""

    def __init__(self, name: str, priority: int = 1):
        self.name = name
        self.priority = priority  # Lower number = higher priority
        self.success_count = 0
        self.failure_count = 0
        self.last_used = None

    @abstractmethod
    async def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this strategy can handle the given error"""
        pass

    @abstractmethod
    async def recover(self, error_context: ErrorContext, original_args: Tuple, original_kwargs: Dict) -> Any:
        """Attempt to recover from the error"""
        pass

    def get_success_rate(self) -> float:
        """Get success rate of this strategy"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def record_success(self):
        """Record successful recovery"""
        self.success_count += 1
        self.last_used = datetime.now()

    def record_failure(self):
        """Record failed recovery"""
        self.failure_count += 1
        self.last_used = datetime.now()


class MemoryRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for memory-related errors"""

    def __init__(self):
        super().__init__("memory_recovery", priority=1)

    async def can_handle(self, error_context: ErrorContext) -> bool:
        return error_context.category == ErrorCategory.MEMORY

    async def recover(self, error_context: ErrorContext, original_args: Tuple, original_kwargs: Dict) -> Any:
        """Recover from memory errors by clearing cache and reducing batch size"""
        logger.warning(f"Attempting memory recovery for session {error_context.session_id}")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        import gc
        gc.collect()

        # Reduce batch size if applicable
        if 'batch_size' in original_kwargs:
            original_batch_size = original_kwargs['batch_size']
            new_batch_size = max(1, original_batch_size // 2)
            original_kwargs['batch_size'] = new_batch_size
            logger.info(f"Reduced batch size from {original_batch_size} to {new_batch_size}")

        # Wait a moment for memory to stabilize
        await asyncio.sleep(0.5)

        return {"status": "memory_cleared", "action": "retry_with_reduced_resources"}


class ModelRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for model-related errors"""

    def __init__(self, fallback_model_creator: Optional[Callable] = None):
        super().__init__("model_recovery", priority=2)
        self.fallback_model_creator = fallback_model_creator
        self.fallback_model = None

    async def can_handle(self, error_context: ErrorContext) -> bool:
        return error_context.category == ErrorCategory.MODEL

    async def recover(self, error_context: ErrorContext, original_args: Tuple, original_kwargs: Dict) -> Any:
        """Recover from model errors using fallback models"""
        logger.warning(f"Attempting model recovery for session {error_context.session_id}")

        if self.fallback_model_creator and not self.fallback_model:
            try:
                self.fallback_model = self.fallback_model_creator()
                logger.info("Created fallback model")
            except Exception as e:
                logger.error(f"Failed to create fallback model: {e}")
                return None

        return {
            "status": "fallback_model_ready",
            "model": self.fallback_model,
            "action": "use_fallback_model"
        }


class ToyWorldFallbackStrategy(RecoveryStrategy):
    """Ultimate fallback using toy world simulator"""

    def __init__(self):
        super().__init__("toy_world_fallback", priority=10)  # Lowest priority (last resort)

    async def can_handle(self, error_context: ErrorContext) -> bool:
        # Can handle any error as ultimate fallback
        return True

    async def recover(self, error_context: ErrorContext, original_args: Tuple, original_kwargs: Dict) -> Any:
        """Use toy world as ultimate fallback"""
        logger.warning(f"Using toy world fallback for session {error_context.session_id}")

        from ..inference.toy_world import ToyWorldSimulator

        toy_world = ToyWorldSimulator()

        # Create a deterministic fallback frame
        fallback_frame = np.ones((256, 256, 3), dtype=np.uint8) * 128

        return {
            "status": "toy_world_fallback",
            "simulator": toy_world,
            "frame": fallback_frame,
            "action": "use_toy_world"
        }


class TimeoutRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for timeout errors"""

    def __init__(self):
        super().__init__("timeout_recovery", priority=3)

    async def can_handle(self, error_context: ErrorContext) -> bool:
        return error_context.category == ErrorCategory.TIMEOUT

    async def recover(self, error_context: ErrorContext, original_args: Tuple, original_kwargs: Dict) -> Any:
        """Recover from timeouts by extending timeout and simplifying operation"""
        logger.warning(f"Attempting timeout recovery for session {error_context.session_id}")

        # Extend timeout if present
        if 'timeout' in original_kwargs:
            original_timeout = original_kwargs['timeout']
            new_timeout = original_timeout * 2
            original_kwargs['timeout'] = new_timeout
            logger.info(f"Extended timeout from {original_timeout}s to {new_timeout}s")

        # Add retry delay
        await asyncio.sleep(1.0)

        return {"status": "timeout_extended", "action": "retry_with_longer_timeout"}


class ErrorRecoveryManager:
    """Manages error recovery with multiple strategies"""

    def __init__(self):
        self.classifier = ErrorClassifier()
        self.strategies: List[RecoveryStrategy] = []
        self.error_history: List[ErrorContext] = []
        self.max_history = 1000
        self.max_recovery_attempts = 3

        # Initialize default strategies
        self._initialize_default_strategies()

    def _initialize_default_strategies(self):
        """Initialize default recovery strategies"""
        self.strategies = [
            MemoryRecoveryStrategy(),
            ModelRecoveryStrategy(),
            TimeoutRecoveryStrategy(),
            ToyWorldFallbackStrategy()  # Always last
        ]

        # Sort by priority
        self.strategies.sort(key=lambda s: s.priority)

    def add_strategy(self, strategy: RecoveryStrategy):
        """Add a custom recovery strategy"""
        self.strategies.append(strategy)
        self.strategies.sort(key=lambda s: s.priority)

    async def handle_error(self, error: Exception, session_id: Optional[str] = None,
                          operation: Optional[str] = None, original_args: Tuple = (),
                          original_kwargs: Dict = None) -> Optional[Any]:
        """Handle an error with appropriate recovery strategy"""
        if original_kwargs is None:
            original_kwargs = {}

        # Classify the error
        category, severity = self.classifier.classify_error(error)

        # Create error context
        error_context = ErrorContext(
            error_type=type(error),
            error_message=str(error),
            category=category,
            severity=severity,
            timestamp=datetime.now(),
            session_id=session_id,
            operation=operation,
            stack_trace=traceback.format_exc(),
            recovery_attempted=False,
            recovery_successful=False
        )

        # Record error
        self._record_error(error_context)

        # Log error
        logger.error(f"Error in operation '{operation}' for session '{session_id}': {error}")

        # Check if we should attempt recovery
        if not self._should_attempt_recovery(error_context):
            logger.info("Skipping recovery attempt")
            return None

        # Try recovery strategies
        for strategy in self.strategies:
            try:
                if await strategy.can_handle(error_context):
                    logger.info(f"Attempting recovery with strategy: {strategy.name}")

                    error_context.recovery_attempted = True
                    recovery_result = await strategy.recover(error_context, original_args, original_kwargs)

                    if recovery_result:
                        strategy.record_success()
                        error_context.recovery_successful = True
                        logger.info(f"Recovery successful with strategy: {strategy.name}")
                        return recovery_result
                    else:
                        strategy.record_failure()

            except Exception as recovery_error:
                strategy.record_failure()
                logger.error(f"Recovery strategy '{strategy.name}' failed: {recovery_error}")

        # No recovery strategy worked
        logger.error(f"All recovery strategies failed for error: {error}")
        return None

    def _record_error(self, error_context: ErrorContext):
        """Record error in history"""
        self.error_history.append(error_context)

        # Maintain history size
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

    def _should_attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Determine if recovery should be attempted"""
        # Always attempt recovery for non-critical errors
        if error_context.severity != ErrorSeverity.CRITICAL:
            return True

        # For critical errors, check recent failure rate
        recent_errors = [
            e for e in self.error_history[-10:]  # Last 10 errors
            if e.severity == ErrorSeverity.CRITICAL
            and (datetime.now() - e.timestamp).total_seconds() < 300  # Last 5 minutes
        ]

        # Don't attempt recovery if too many recent critical errors
        if len(recent_errors) > 5:
            logger.warning("Too many recent critical errors, skipping recovery")
            return False

        return True

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        if not self.error_history:
            return {"total_errors": 0}

        # Count by category and severity
        category_counts = {}
        severity_counts = {}

        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1

        # Recovery statistics
        recovery_attempts = len([e for e in self.error_history if e.recovery_attempted])
        recovery_successes = len([e for e in self.error_history if e.recovery_successful])
        recovery_rate = recovery_successes / recovery_attempts if recovery_attempts > 0 else 0.0

        # Strategy statistics
        strategy_stats = []
        for strategy in self.strategies:
            strategy_stats.append({
                "name": strategy.name,
                "priority": strategy.priority,
                "success_count": strategy.success_count,
                "failure_count": strategy.failure_count,
                "success_rate": strategy.get_success_rate(),
                "last_used": strategy.last_used.isoformat() if strategy.last_used else None
            })

        return {
            "total_errors": len(self.error_history),
            "recovery_attempts": recovery_attempts,
            "recovery_successes": recovery_successes,
            "recovery_rate": recovery_rate,
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "strategy_performance": strategy_stats,
            "recent_errors": len([
                e for e in self.error_history
                if (datetime.now() - e.timestamp).total_seconds() < 3600
            ])
        }

    def clear_error_history(self):
        """Clear error history (useful for testing)"""
        self.error_history.clear()
        for strategy in self.strategies:
            strategy.success_count = 0
            strategy.failure_count = 0
            strategy.last_used = None


# Decorator for automatic error recovery
def with_error_recovery(recovery_manager: ErrorRecoveryManager, operation_name: str = None):
    """Decorator to add automatic error recovery to functions"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            session_id = kwargs.get('session_id')
            op_name = operation_name or func.__name__

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                recovery_result = await recovery_manager.handle_error(
                    e, session_id, op_name, args, kwargs
                )

                if recovery_result:
                    # If recovery provides specific action
                    action = recovery_result.get('action')
                    if action == 'retry_with_reduced_resources':
                        return await func(*args, **kwargs)
                    elif action == 'use_fallback_model':
                        # Modify kwargs to use fallback model
                        kwargs['fallback_model'] = recovery_result.get('model')
                        return await func(*args, **kwargs)
                    elif action == 'use_toy_world':
                        # Return toy world result
                        return recovery_result.get('frame')

                # Re-raise if no recovery possible
                raise e

        def sync_wrapper(*args, **kwargs):
            session_id = kwargs.get('session_id')
            op_name = operation_name or func.__name__

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, just log and re-raise
                logger.error(f"Error in {op_name}: {e}")
                raise e

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Global instance
global_recovery_manager = ErrorRecoveryManager()


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get global recovery manager instance"""
    return global_recovery_manager