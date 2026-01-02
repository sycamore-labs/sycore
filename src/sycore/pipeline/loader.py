"""Pipeline loader for loading pipeline definitions from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StageDefinition:
    """Definition of a pipeline stage."""

    name: str
    stage_class: str
    enabled: bool = True
    required: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineDefinition:
    """Definition of a pipeline from YAML configuration."""

    name: str
    description: str
    version: str = "1.0"
    stages: list[StageDefinition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineLoader:
    """Loads pipeline definitions from YAML files.

    Supports layered configuration with customizable search paths.
    Domain-specific implementations can provide stage class mappings.
    """

    def __init__(
        self,
        search_paths: list[Path] | None = None,
        app_name: str = "sycore",
        project_dir: Path | None = None,
        stage_class_mapping: dict[str, str] | None = None,
    ):
        """Initialize the pipeline loader.

        Args:
            search_paths: List of directories to search for pipelines.
                         If None, uses layered config paths based on app_name.
            app_name: Application name for default path construction.
            project_dir: Project directory for .sy/{app_name}/ lookup.
            stage_class_mapping: Optional mapping of short names to class names.
        """
        if search_paths is None:
            project_root = project_dir or Path.cwd()

            # Search in order: project, user (first match wins)
            search_paths = [
                project_root / ".sy" / app_name / "pipelines",  # Project overrides
                Path.home() / ".sy" / app_name / "pipelines",  # User overrides
            ]

        self._search_paths = search_paths
        self._stage_class_mapping = stage_class_mapping or {}
        self._pipelines: dict[str, PipelineDefinition] = {}
        self._discover()

    def add_search_path(self, path: Path, priority: int = 0) -> None:
        """Add a search path for pipeline definitions.

        Args:
            path: Directory to search for pipeline YAML files.
            priority: 0 = highest priority (searched first), -1 = append to end.
        """
        if priority == 0:
            self._search_paths.insert(0, path)
        else:
            self._search_paths.append(path)
        self._discover()

    def set_stage_class_mapping(self, mapping: dict[str, str]) -> None:
        """Set the stage class mapping.

        Args:
            mapping: Dict mapping short names to class names.
        """
        self._stage_class_mapping = mapping

    def _discover(self) -> None:
        """Discover and load all pipeline definitions from search paths."""
        self._pipelines.clear()
        for search_path in self._search_paths:
            if not search_path.exists():
                continue

            for pipeline_file in search_path.glob("*.yaml"):
                self._load_definition(pipeline_file)

    def _load_definition(self, pipeline_path: Path) -> None:
        """Load a pipeline definition from a YAML file.

        Args:
            pipeline_path: Path to the pipeline YAML file.
        """
        try:
            with open(pipeline_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return

            name = data.get("name", pipeline_path.stem)

            # Don't override if already loaded (higher priority path wins)
            if name in self._pipelines:
                return

            # Parse stages
            stages = []
            for stage_data in data.get("stages", []):
                if isinstance(stage_data, str):
                    # Simple stage name
                    stage_name = stage_data
                    # Use mapping if available, otherwise use the name as-is
                    stage_class = self._stage_class_mapping.get(stage_name, stage_name)
                    stages.append(
                        StageDefinition(
                            name=stage_name,
                            stage_class=stage_class,
                        )
                    )
                elif isinstance(stage_data, dict):
                    # Detailed stage definition
                    stage_name = str(stage_data.get("name", "unknown"))
                    # Use explicit class, or mapping, or name as fallback
                    stage_class = str(
                        stage_data.get("class")
                        or self._stage_class_mapping.get(stage_name, stage_name)
                    )
                    stages.append(
                        StageDefinition(
                            name=stage_name,
                            stage_class=stage_class,
                            enabled=stage_data.get("enabled", True),
                            required=stage_data.get("required", True),
                            config=stage_data.get("config", {}),
                        )
                    )

            definition = PipelineDefinition(
                name=name,
                description=data.get("description", ""),
                version=data.get("version", "1.0"),
                stages=stages,
                metadata=data,
            )

            self._pipelines[name] = definition

        except Exception:
            # Skip invalid files
            pass

    def get(self, name: str) -> PipelineDefinition | None:
        """Get a pipeline definition by name.

        Args:
            name: Name of the pipeline.

        Returns:
            PipelineDefinition or None if not found.
        """
        return self._pipelines.get(name)

    def load(self, name: str) -> PipelineDefinition:
        """Load a pipeline by name.

        Args:
            name: Name of the pipeline.

        Returns:
            PipelineDefinition.

        Raises:
            FileNotFoundError: If pipeline not found.
        """
        pipeline = self.get(name)
        if pipeline is None:
            raise FileNotFoundError(f"Pipeline '{name}' not found")
        return pipeline

    def list_pipelines(self) -> list[str]:
        """List all registered pipeline names.

        Returns:
            List of pipeline names.
        """
        return sorted(self._pipelines.keys())

    def list_definitions(self) -> list[PipelineDefinition]:
        """List all registered pipeline definitions.

        Returns:
            List of PipelineDefinition instances.
        """
        return list(self._pipelines.values())


# Global loader instance
_loader: PipelineLoader | None = None


def get_pipeline_loader(app_name: str = "sycore") -> PipelineLoader:
    """Get the global pipeline loader instance.

    Args:
        app_name: Application name for path construction.

    Returns:
        PipelineLoader instance with discovered pipelines.
    """
    global _loader
    if _loader is None:
        _loader = PipelineLoader(app_name=app_name)
    return _loader


def reset_pipeline_loader() -> None:
    """Reset the global pipeline loader (useful for testing)."""
    global _loader
    _loader = None
