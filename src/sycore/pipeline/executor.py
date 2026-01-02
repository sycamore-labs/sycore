"""Pipeline executor orchestrates multi-stage execution."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from sycore.pipeline.context import BasePipelineConfig, BasePipelineContext, PipelineLogger, StageResult

if TYPE_CHECKING:
    from sycore.pipeline.loader import PipelineDefinition
    from sycore.pipeline.stages import PipelineStage

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes a dynamically configured pipeline.

    Unlike domain-specific executors, this generic executor has NO hardcoded stages.
    All stages must be provided via:
    1. YAML pipeline definition (pipeline_name + stage_registry)
    2. Explicit stage list passed to constructor
    3. Domain-specific subclass that calls _set_stages()

    Example usage with explicit stages:
        executor = PipelineExecutor(context, config)
        executor._set_stages([
            MyStage1(context, config, logger),
            MyStage2(context, config, logger),
        ])
        output_path = executor.execute()

    Example usage with YAML pipeline:
        executor = PipelineExecutor(
            context, config,
            pipeline_name="my_pipeline",
            stage_registry={"MyStage1": MyStage1, "MyStage2": MyStage2},
        )
        output_path = executor.execute()
    """

    def __init__(
        self,
        context: BasePipelineContext,
        config: BasePipelineConfig,
        on_progress: Callable[[str], None] | None = None,
        verbose: bool = False,
        pipeline_name: str | None = None,
        stages: list[PipelineStage] | None = None,
        stage_registry: dict[str, type[PipelineStage]] | None = None,
    ):
        """Initialize the pipeline executor.

        Args:
            context: Pipeline context (shared state).
            config: Pipeline configuration.
            on_progress: Callback for progress reporting.
            verbose: Enable verbose logging.
            pipeline_name: Name of YAML pipeline to load.
            stages: Explicit list of stage instances.
            stage_registry: Mapping of stage class names to classes (for YAML loading).
        """
        self.context = context
        self.config = config
        self.logger = PipelineLogger(on_progress=on_progress, verbose=verbose)
        self.stages: list[PipelineStage] = stages or []
        self.results: list[StageResult] = []
        self.pipeline_name = pipeline_name
        self.pipeline_definition: PipelineDefinition | None = None
        self._stage_registry = stage_registry or {}

        # Initialize from YAML if pipeline_name specified and no stages provided
        if pipeline_name and not stages:
            try:
                self._initialize_from_yaml()
            except Exception as e:
                logger.warning(
                    f"Failed to load pipeline '{pipeline_name}': {e}. "
                    "No stages configured."
                )

    def register_stage_class(self, name: str, stage_class: type[PipelineStage]) -> None:
        """Register a stage class for YAML pipeline loading.

        Args:
            name: Class name as it appears in YAML (e.g., "PlanningStage").
            stage_class: The stage class to register.
        """
        self._stage_registry[name] = stage_class

    def _set_stages(self, stages: list[PipelineStage]) -> None:
        """Set stages programmatically.

        This is the primary way for subclasses to configure stages.

        Args:
            stages: List of stage instances to execute.
        """
        self.stages = stages

    def _initialize_from_yaml(self) -> None:
        """Initialize stages from YAML pipeline definition."""
        from sycore.pipeline.loader import get_pipeline_loader

        loader = get_pipeline_loader()
        definition = loader.load(self.pipeline_name or "default")
        self.pipeline_definition = definition

        self.stages = []
        for stage_def in definition.stages:
            if not stage_def.enabled:
                logger.debug(f"Skipping disabled stage: {stage_def.name}")
                continue

            stage_cls = self._stage_registry.get(stage_def.stage_class)
            if stage_cls is None:
                logger.warning(
                    f"Unknown stage class: {stage_def.stage_class}. "
                    f"Register it with register_stage_class() first."
                )
                continue

            stage = stage_cls(self.context, self.config, self.logger)
            self.stages.append(stage)

        logger.info(
            f"Loaded pipeline '{definition.name}' with {len(self.stages)} stages"
        )

    def execute(self) -> Path:
        """Execute the full pipeline and return output path.

        Returns:
            Path to the generated output file.

        Raises:
            PipelineError: If a critical stage fails or no output path is set.
        """
        if not self.stages:
            raise PipelineError("No stages configured. Add stages before executing.")

        self.logger.info("Starting pipeline execution")

        for stage in self.stages:
            if not stage.should_run():
                self.logger.debug(f"Skipping stage: {stage.name}", stage=stage.name)
                continue

            self.logger.info(f"Running stage: {stage.name}", stage=stage.name)
            self.context.current_stage = stage.name

            start_time = time.time()
            try:
                stage.run()
                duration = time.time() - start_time

                result = StageResult(
                    stage_name=stage.name,
                    success=True,
                    duration_seconds=duration,
                )
                self.results.append(result)
                self.context.record_stage_time(stage.name, duration)

                self.logger.info(
                    f"Completed stage: {stage.name} ({duration:.1f}s)",
                    stage=stage.name,
                )

            except Exception as e:
                duration = time.time() - start_time
                result = StageResult(
                    stage_name=stage.name,
                    success=False,
                    duration_seconds=duration,
                    error=str(e),
                )
                self.results.append(result)

                self.logger.error(f"Stage failed: {stage.name} - {e}", stage=stage.name)

                # Check if stage is critical
                if stage.is_critical:
                    raise PipelineError(f"Critical stage '{stage.name}' failed: {e}") from e

        self.logger.info(f"Pipeline complete in {self.context.total_duration:.1f}s")

        if self.context.output_path is None:
            raise PipelineError("No output path set after pipeline execution")

        return self.context.output_path

    def execute_until(self, stage_name: str) -> BasePipelineContext:
        """Execute pipeline up to and including a specific stage.

        Args:
            stage_name: Name of the stage to stop at.

        Returns:
            The pipeline context with intermediate results.
        """
        for stage in self.stages:
            if not stage.should_run():
                continue

            self.logger.info(f"Running stage: {stage.name}", stage=stage.name)
            self.context.current_stage = stage.name

            start_time = time.time()
            stage.run()
            duration = time.time() - start_time
            self.context.record_stage_time(stage.name, duration)

            if stage.name == stage_name:
                break

        return self.context

    def get_stage_by_name(self, name: str) -> PipelineStage | None:
        """Get a stage by its name."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def get_results_summary(self) -> dict:
        """Get a summary of pipeline execution."""
        return {
            "total_duration": self.context.total_duration,
            "stages_run": len([r for r in self.results if r.success]),
            "stages_failed": len([r for r in self.results if not r.success]),
            "stage_timings": self.context.stage_timings,
            "results": [
                {
                    "stage": r.stage_name,
                    "success": r.success,
                    "duration": r.duration_seconds,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


class PipelineError(Exception):
    """Exception raised when pipeline execution fails."""

    pass
