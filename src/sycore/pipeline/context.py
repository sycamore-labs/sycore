"""Pipeline context and configuration base classes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BasePipelineContext:
    """Base context shared across all pipeline stages.

    This provides generic pipeline state management. Domain-specific
    implementations should extend this class to add their own fields.

    Attributes:
        source_content: The source content being processed.
        source_file: Optional path to the source file.
        output_path: Path where output will be written.
        start_time: When the pipeline started.
        current_stage: Name of the currently executing stage.
        stage_timings: Duration of each completed stage.
        on_progress: Callback for progress reporting.
    """

    source_content: str
    source_file: Path | None = None
    output_path: Path | None = None

    # Pipeline state
    start_time: datetime = field(default_factory=datetime.now)
    current_stage: str = ""
    stage_timings: dict[str, float] = field(default_factory=dict)

    # Progress callback
    on_progress: Callable[[str], None] | None = None

    def report_progress(self, message: str) -> None:
        """Report progress to callback if set."""
        if self.on_progress:
            self.on_progress(message)

    def record_stage_time(self, stage_name: str, duration: float) -> None:
        """Record how long a stage took."""
        self.stage_timings[stage_name] = duration

    @property
    def total_duration(self) -> float:
        """Get total pipeline duration in seconds."""
        return (datetime.now() - self.start_time).total_seconds()


@dataclass
class BasePipelineConfig:
    """Base configuration for pipeline execution.

    This provides generic pipeline configuration. Domain-specific
    implementations should extend this class with their own settings.

    Attributes:
        primary_provider: LLM provider to use.
        primary_api_key: API key for the primary provider.
        primary_model: Model name (optional, uses provider default).
    """

    primary_provider: str = "anthropic"
    primary_api_key: str = ""
    primary_model: str | None = None


@dataclass
class StageResult:
    """Result from a pipeline stage."""

    stage_name: str
    success: bool
    duration_seconds: float
    error: str | None = None
    output: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)


class PipelineLogger:
    """Logger for pipeline execution with progress reporting."""

    def __init__(
        self,
        on_progress: Callable[[str], None] | None = None,
        verbose: bool = False,
    ):
        self.on_progress = on_progress
        self.verbose = verbose
        self.logs: list[dict[str, Any]] = []

    def info(self, message: str, stage: str | None = None) -> None:
        """Log info message."""
        self._log("INFO", message, stage)
        if self.on_progress:
            self.on_progress(message)

    def debug(self, message: str, stage: str | None = None) -> None:
        """Log debug message (only if verbose)."""
        if self.verbose:
            self._log("DEBUG", message, stage)

    def warning(self, message: str, stage: str | None = None) -> None:
        """Log warning message."""
        self._log("WARNING", message, stage)

    def error(self, message: str, stage: str | None = None) -> None:
        """Log error message."""
        self._log("ERROR", message, stage)

    def _log(self, level: str, message: str, stage: str | None) -> None:
        """Internal log method."""
        self.logs.append(
            {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "stage": stage,
                "message": message,
            }
        )

    def get_logs(self, level: str | None = None) -> list[dict[str, Any]]:
        """Get logs, optionally filtered by level."""
        if level is None:
            return self.logs
        return [log for log in self.logs if log["level"] == level]
