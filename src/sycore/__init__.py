"""sycore - Generic agent/pipeline execution framework.

This is the core of the Sycamore AI operating system driver.
"""

__version__ = "0.1.0"

from sycore.agents.base import Agent, AgentContext, AgentTool
from sycore.pipeline.context import BasePipelineConfig, BasePipelineContext, StageResult
from sycore.pipeline.executor import PipelineError, PipelineExecutor
from sycore.pipeline.stages import PipelineStage

__all__ = [
    # Agents
    "Agent",
    "AgentContext",
    "AgentTool",
    # Pipeline
    "PipelineExecutor",
    "PipelineError",
    "PipelineStage",
    "BasePipelineContext",
    "BasePipelineConfig",
    "StageResult",
]
