"""Where archived files are kept.

The archive outgrows a laptop's disk once the pipeline runs on a schedule, so
everything that writes or reads it goes through this interface instead of Path.
"""

from pathlib import Path
from typing import Protocol


class ObjectStore(Protocol):
    # put takes a key and returns a uri: a key is where a caller wants a file, a
    # uri is where it ended up, and only the store knows how to turn one into
    # the other. A remote store returns its own scheme while still taking keys.
    def put(self, key: str, data: bytes) -> str: ...

    def get(self, uri: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class FilesystemStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, key: str, data: bytes) -> str:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path.as_posix()

    def get(self, uri: str) -> bytes:
        return Path(uri).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.root / key).is_file()
