from pathlib import Path
import os
import sys
from typing import List, Tuple

class ProjectPath:
    _project_root_cache = {}

    @classmethod
    def get_start_path(cls):
        if '__file__' in globals():
            return Path(__file__).resolve().parent
        elif 'ipykernel' in sys.modules:
            # Specific handling for Jupyter notebooks
            return Path(os.getcwd())
        else:
            return Path(os.getcwd())

    @classmethod
    def find_project_root(cls, start_path: Path, markers: Tuple[str, ...]) -> Path:
        """
        Cache-enabled method to find the project root by looking for directory markers.

        :param start_path: Path to start search from.
        :param markers: Markers to look for in directories.
        :return: Path object representing the project root directory.
        """
        cache_key = (str(start_path), markers)
        if cache_key in cls._project_root_cache:
            return cls._project_root_cache[cache_key]

        current_path = start_path
        while True:
            if any((current_path / marker).exists() for marker in markers):
                cls._project_root_cache[cache_key] = current_path
                return current_path
            if current_path.parent == current_path:
                raise FileNotFoundError(f"Project root not found using the provided markers: {markers}.")
            current_path = current_path.parent

    @classmethod
    def invalidate_cache(cls):
        cls._project_root_cache.clear()

    def __init__(self, *path_parts, markers: List[str] = None):
        if markers is None:
            markers = ['.git', '.hg', '.svn', 'pyproject.toml', 'setup.py', '.project']
        self.markers = markers
        self.path_parts = path_parts

        start_path = self.get_start_path()
        self.project_root = self.find_project_root(start_path, tuple(self.markers))

    def add_marker(self, marker: str):
        if marker not in self.markers:
            self.markers.append(marker)
            # Invalidate cache as the markers have changed
            self._project_root_cache.clear()

    def __str__(self) -> str:
        return str(self.project_root.joinpath(*self.path_parts))

    def __fspath__(self) -> str:
        return str(self)

    def path(self) -> Path:
        """
        Return the Path object for the constructed path.
        """
        return self.project_root.joinpath(*self.path_parts)

# Example usage
project_file = ProjectPath("data", "mydata.csv", markers=['.git', 'my_project_marker.file'])
print(project_file)  # Print the path as a string
print(project_file.path())  # Get the path as a Path object
