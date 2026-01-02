"""Pipeline execution framework for sycore."""

from sycore.pipeline.context import (
    BasePipelineConfig,
    BasePipelineContext,
    PipelineLogger,
    StageResult,
)
from sycore.pipeline.executor import PipelineError, PipelineExecutor
from sycore.pipeline.loader import (
    AgentConfig,
    PipelineDefinition,
    PipelineLoader,
    StageDefinition,
    get_pipeline_loader,
    reset_pipeline_loader,
)
from sycore.pipeline.registry import StageRegistry, get_stage_registry, reset_stage_registry
from sycore.pipeline.stages import PipelineStage

__all__ = [
    # Executor
    "PipelineExecutor",
    "PipelineError",
    # Stages
    "PipelineStage",
    "StageRegistry",
    "get_stage_registry",
    "reset_stage_registry",
    # Loader
    "PipelineLoader",
    "PipelineDefinition",
    "StageDefinition",
    "AgentConfig",
    "get_pipeline_loader",
    "reset_pipeline_loader",
    # Context
    "BasePipelineContext",
    "BasePipelineConfig",
    "PipelineLogger",
    "StageResult",
]
