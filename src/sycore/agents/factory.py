"""Agent factory for creating agents from YAML definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AgentDefinition:
    """Definition of an agent from YAML configuration."""

    name: str
    description: str
    prompt_name: str
    skills: list[str] = field(default_factory=list)
    model: str = ""
    version: str = "1.0"
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentFactory:
    """Factory for creating agents from YAML definitions.

    Supports layered configuration with customizable search paths.
    Typical layering:
    1. Project config - highest priority
    2. User config - medium priority
    3. Package defaults - lowest priority
    """

    def __init__(
        self,
        search_paths: list[Path] | None = None,
        app_name: str = "sycore",
        project_dir: Path | None = None,
    ):
        """Initialize the agent factory.

        Args:
            search_paths: List of directories to search for agent definitions.
                         If None, uses layered config paths based on app_name.
            app_name: Application name for default path construction.
            project_dir: Project directory for .sy/{app_name}/ lookup.
        """
        if search_paths is None:
            project_root = project_dir or Path.cwd()

            # Search in order: project, user (first match wins)
            search_paths = [
                project_root / ".sy" / app_name / "agents",  # Project overrides
                Path.home() / ".sy" / app_name / "agents",  # User overrides
            ]

        self._search_paths = search_paths
        self._definitions: dict[str, AgentDefinition] = {}
        self._discover()

    def add_search_path(self, path: Path, priority: int = 0) -> None:
        """Add a search path for agent definitions.

        Args:
            path: Directory to search for agent YAML files.
            priority: 0 = highest priority (searched first), -1 = append to end.
        """
        if priority == 0:
            self._search_paths.insert(0, path)
        else:
            self._search_paths.append(path)
        self._discover()

    def _discover(self) -> None:
        """Discover and load all agent definitions from search paths."""
        self._definitions.clear()
        for search_path in self._search_paths:
            if not search_path.exists():
                continue

            for agent_file in search_path.glob("*.yaml"):
                self._load_definition(agent_file)

    def _load_definition(self, agent_path: Path) -> None:
        """Load an agent definition from a YAML file.

        Args:
            agent_path: Path to the agent YAML file.
        """
        try:
            with open(agent_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return

            name = data.get("name", agent_path.stem)
            # Don't override if already loaded (higher priority path wins)
            if name in self._definitions:
                return

            definition = AgentDefinition(
                name=name,
                description=data.get("description", ""),
                prompt_name=data.get("prompt", name),
                skills=data.get("skills", []),
                model=data.get("model", ""),
                version=data.get("version", "1.0"),
                category=data.get("category", "general"),
                metadata=data,
            )

            self._definitions[name] = definition

        except Exception:
            # Skip invalid files
            pass

    def get(self, name: str) -> AgentDefinition | None:
        """Get an agent definition by name.

        Args:
            name: Name of the agent.

        Returns:
            AgentDefinition or None if not found.
        """
        return self._definitions.get(name)

    def list_agents(self) -> list[str]:
        """List all registered agent names.

        Returns:
            List of agent names.
        """
        return sorted(self._definitions.keys())

    def list_definitions(self) -> list[AgentDefinition]:
        """List all registered agent definitions.

        Returns:
            List of AgentDefinition instances.
        """
        return list(self._definitions.values())

    def get_by_category(self, category: str) -> list[AgentDefinition]:
        """Get agents by category.

        Args:
            category: Category name.

        Returns:
            List of agents in the category.
        """
        return [d for d in self._definitions.values() if d.category == category]

    @classmethod
    def default(cls, app_name: str = "sycore") -> AgentFactory:
        """Get a default factory instance.

        Args:
            app_name: Application name for path construction.

        Returns:
            AgentFactory with default search paths.
        """
        return cls(app_name=app_name)
