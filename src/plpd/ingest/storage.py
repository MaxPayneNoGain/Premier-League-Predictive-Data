from pathlib import Path
from typing import Protocol


class ObjectStore(Protocol):
    # A key is where a caller wants a file, a uri is where it ended up. A remote
    # store returns its own scheme from put while still taking plain keys.
    def put(self, key: str, data: bytes) -> str: ...

    def get(self, uri: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class FilesystemStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, key: str, data: bytes) -> str:
        path = self.root / key
        # Keys are content hashes, so a key already present holds these bytes.
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return path.as_posix()

    def get(self, uri: str) -> bytes:
        return Path(uri).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.root / key).is_file()
