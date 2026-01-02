"""Stage registry with entry point discovery."""

from __future__ import annotations

import importlib
import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sycore.pipeline.stages import PipelineStage

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "sycore.stages"


class StageRegistry:
    """Registry for pipeline stages with entry point discovery.

    Stages can be registered via:
    1. Entry points (pyproject.toml) - discovered at runtime
    2. Explicit registration - for testing or dynamic addition
    3. Module path - "package.module:ClassName" format

    Example pyproject.toml entry:
        [project.entry-points."sycore.stages"]
        planning = "sydeck.pipeline.stages:PlanningStage"
        rendering = "sydeck.pipeline.stages:RenderingStage"

    Usage:
        registry = get_stage_registry()
        stage_class = registry.get("planning")  # From entry point
        stage_class = registry.get("sydeck.pipeline.stages:PlanningStage")  # From path
    """

    def __init__(self, discover_entry_points: bool = True):
        """Initialize the stage registry.

        Args:
            discover_entry_points: Whether to discover stages from entry points.
        """
        self._stages: dict[str, type[PipelineStage]] = {}
        self._discovered = False

        if discover_entry_points:
            self.discover()

    def discover(self) -> None:
        """Discover stages from entry points."""
        if self._discovered:
            return

        try:
            eps = entry_points(group=ENTRY_POINT_GROUP)
        except TypeError:
            # Python 3.9 fallback
            all_eps = entry_points()
            eps = all_eps.get(ENTRY_POINT_GROUP, [])

        for ep in eps:
            try:
                stage_class = ep.load()
                self._stages[ep.name] = stage_class
                logger.debug(f"Discovered stage: {ep.name} -> {ep.value}")
            except Exception as e:
                logger.warning(f"Failed to load stage '{ep.name}': {e}")

        self._discovered = True
        if self._stages:
            logger.info(f"Discovered {len(self._stages)} stages from entry points")

    def register(self, name: str, stage_class: type[PipelineStage]) -> None:
        """Register a stage class explicitly.

        Args:
            name: Stage name (used in pipeline YAML).
            stage_class: The stage class.
        """
        self._stages[name] = stage_class
        logger.debug(f"Registered stage: {name}")

    def get(self, name: str) -> type[PipelineStage] | None:
        """Get a stage class by name.

        Supports:
        - Simple names: "planning" (from registry)
        - Module paths: "sydeck.pipeline.stages:PlanningStage"

        Args:
            name: Stage name or module path.

        Returns:
            Stage class or None if not found.
        """
        # Check registry first
        if name in self._stages:
            return self._stages[name]

        # Try module path format (package.module:ClassName)
        if ":" in name:
            return self._load_from_path(name)

        return None

    def _load_from_path(self, path: str) -> type[PipelineStage] | None:
        """Load stage class from module path.

        Args:
            path: Module path in format "package.module:ClassName"

        Returns:
            Stage class or None.
        """
        try:
            module_path, class_name = path.rsplit(":", 1)
            module = importlib.import_module(module_path)
            stage_class = getattr(module, class_name)
            # Cache for future lookups
            self._stages[path] = stage_class
            return stage_class
        except Exception as e:
            logger.warning(f"Failed to load stage from path '{path}': {e}")
            return None

    def list_stages(self) -> list[str]:
        """List all registered stage names."""
        return sorted(self._stages.keys())

    def has(self, name: str) -> bool:
        """Check if a stage is registered.

        Args:
            name: Stage name.

        Returns:
            True if stage exists in registry.
        """
        return name in self._stages

    def to_executor_registry(self) -> dict[str, type[PipelineStage]]:
        """Convert to executor-compatible stage registry dict.

        Returns:
            Dict mapping stage names to stage classes.
        """
        return dict(self._stages)

    def clear(self) -> None:
        """Clear all registered stages."""
        self._stages.clear()
        self._discovered = False


# Global registry instance
_registry: StageRegistry | None = None


def get_stage_registry() -> StageRegistry:
    """Get the global stage registry instance.

    Returns:
        StageRegistry instance with discovered stages.
    """
    global _registry
    if _registry is None:
        _registry = StageRegistry()
    return _registry


def reset_stage_registry() -> None:
    """Reset the global stage registry (for testing)."""
    global _registry
    _registry = None
