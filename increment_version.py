# increment_version.py
import os
import toml
from pathlib import Path

def increment_version(file_path: str):
    '''
    This method updates the patch counter associated with the version of HAT.
    Python/pyproject.toml is updated, which contains metadata used for distribution
    by PYPI.
    '''
    file = Path(file_path)
    data = toml.loads(file.read_text())

    # Extract and increment version
    version_parts = data['project']['version'].split('.')
    version_parts[-1] = str(int(version_parts[-1]) + 1)
    data['project']['version'] = '.'.join(version_parts)

    # Write back updated version
    file.write_text(toml.dumps(data))
    print(f"Version updated to {data['project']['version']}")

if __name__ == "__main__":
    increment_version("pyproject.toml")
    