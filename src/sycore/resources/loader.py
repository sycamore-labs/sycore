"""Resource loader for externalized YAML configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ResourceLoader:
    """Generic resource loader for YAML configuration files.

    Supports layered configuration with customizable search paths.
    This is a generic base that can be used for styles, themes,
    configurations, or any YAML-based resources.

    Example usage:
        loader = ResourceLoader(app_name="myapp", resource_type="themes")
        theme = loader.load("dark")
    """

    def __init__(
        self,
        search_paths: list[Path] | None = None,
        app_name: str = "sycore",
        resource_type: str = "resources",
        project_dir: Path | None = None,
    ):
        """Initialize the resource loader.

        Args:
            search_paths: List of directories to search for resources.
                         If None, uses layered config paths based on app_name.
            app_name: Application name for default path construction.
            resource_type: Type of resource (e.g., "styles", "themes").
            project_dir: Project directory for .sy/{app_name}/ lookup.
        """
        if search_paths is None:
            project_root = project_dir or Path.cwd()

            # Search in order: project, user (first match wins)
            search_paths = [
                project_root / ".sy" / app_name / resource_type,  # Project overrides
                Path.home() / ".sy" / app_name / resource_type,  # User overrides
            ]

        self._search_paths = search_paths
        self._resource_type = resource_type
        self._cache: dict[str, dict[str, Any]] = {}

    def add_search_path(self, path: Path, priority: int = 0) -> None:
        """Add a search path for resources.

        Args:
            path: Directory to search for resource files.
            priority: 0 = highest priority (searched first), -1 = append to end.
        """
        if priority == 0:
            self._search_paths.insert(0, path)
        else:
            self._search_paths.append(path)

    def _find_resource_file(self, name: str) -> Path | None:
        """Find a resource file by name in search paths.

        Args:
            name: Name of the resource (without extension).

        Returns:
            Path to the resource file, or None if not found.
        """
        for search_path in self._search_paths:
            resource_path = search_path / f"{name}.yaml"
            if resource_path.exists():
                return resource_path
        return None

    def load(self, name: str) -> dict[str, Any]:
        """Load a resource by name.

        Args:
            name: Name of the resource (e.g., 'modern', 'classic').

        Returns:
            Resource configuration dict.

        Raises:
            FileNotFoundError: If resource file doesn't exist.
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name].copy()

        resource_path = self._find_resource_file(name)
        if resource_path is None:
            raise FileNotFoundError(
                f"{self._resource_type.title()} '{name}' not found in search paths: {self._search_paths}"
            )

        # Load YAML
        with open(resource_path) as f:
            resource = yaml.safe_load(f) or {}

        # Cache and return copy
        self._cache[name] = resource
        return resource.copy()

    def load_with_defaults(self, name: str, defaults: dict[str, Any]) -> dict[str, Any]:
        """Load a resource and merge with defaults.

        Args:
            name: Name of the resource.
            defaults: Default values to use if not in resource.

        Returns:
            Merged resource configuration dict.
        """
        try:
            resource = self.load(name)
        except FileNotFoundError:
            return defaults.copy()

        # Merge: resource values override defaults
        result = defaults.copy()
        result.update(resource)
        return result

    def list_resources(self) -> list[str]:
        """List all available resources.

        Returns:
            List of resource names.
        """
        resources = set()
        for search_path in self._search_paths:
            if search_path.exists():
                for resource_file in search_path.glob("*.yaml"):
                    resources.add(resource_file.stem)
        return sorted(resources)

    def exists(self, name: str) -> bool:
        """Check if a resource exists.

        Args:
            name: Name of the resource.

        Returns:
            True if the resource file exists.
        """
        return self._find_resource_file(name) is not None

    def clear_cache(self) -> None:
        """Clear the resource cache."""
        self._cache.clear()


# Global loader instances per resource type
_loaders: dict[str, ResourceLoader] = {}


def get_resource_loader(
    resource_type: str = "resources",
    app_name: str = "sycore",
) -> ResourceLoader:
    """Get a resource loader for a specific resource type.

    Args:
        resource_type: Type of resource (e.g., "styles", "themes").
        app_name: Application name for path construction.

    Returns:
        ResourceLoader instance.
    """
    key = f"{app_name}:{resource_type}"
    if key not in _loaders:
        _loaders[key] = ResourceLoader(app_name=app_name, resource_type=resource_type)
    return _loaders[key]


def reset_resource_loaders() -> None:
    """Reset all resource loaders (useful for testing)."""
    global _loaders
    _loaders.clear()
