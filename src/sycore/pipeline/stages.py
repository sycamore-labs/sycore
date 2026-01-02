"""Pipeline stage base class."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sycore.pipeline.context import BasePipelineConfig, BasePipelineContext, PipelineLogger

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Base class for pipeline stages.

    All pipeline stages must inherit from this class and implement
    the `run()` method. Stages can optionally override `should_run()`
    to conditionally skip execution.

    Attributes:
        name: Human-readable name for the stage.
        is_critical: If True, pipeline stops on failure.
        context: The pipeline context (shared state).
        config: The pipeline configuration.
        logger: Pipeline logger for output.

    Example:
        class MyStage(PipelineStage):
            name = "My Stage"
            is_critical = True

            def should_run(self) -> bool:
                return self.config.some_feature_enabled

            def run(self) -> None:
                self.logger.info("Processing...")
                # Do work and update self.context
    """

    name: str = "base"
    is_critical: bool = False  # If True, pipeline stops on failure

    def __init__(
        self,
        context: "BasePipelineContext",
        config: "BasePipelineConfig",
        pipeline_logger: "PipelineLogger",
    ):
        self.context = context
        self.config = config
        self.logger = pipeline_logger

    @abstractmethod
    def run(self) -> None:
        """Execute this stage.

        Implementations should:
        1. Log progress using self.logger
        2. Update self.context with results
        3. Raise exceptions for unrecoverable errors

        Returns:
            None (results are stored in context)

        Raises:
            Exception: On unrecoverable errors (stops pipeline if is_critical)
        """
        pass

    def should_run(self) -> bool:
        """Check if this stage should run.

        Override this method to add conditional execution logic.
        Return False to skip the stage entirely.

        Returns:
            True if the stage should execute, False to skip.
        """
        return True
