from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import httpx

API_BASE = "https://api.github.com"
UPLOAD_BASE = "https://uploads.github.com"
PAGE_SIZE = 100


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


def asset_name(key: str) -> str:
    # GitHub rewrites anything outside letters, digits, dot, hyphen and
    # underscore in an asset name, so the key's slashes cannot survive.
    return key.replace("/", "-")


def parse_uri(uri: str) -> tuple[str, str, str]:
    parts = uri.removeprefix("gh://").split("/")
    if not uri.startswith("gh://") or len(parts) != 4:
        raise ValueError(f"expected gh://owner/repo/tag/asset, got {uri!r}")
    owner, repo, tag, asset = parts
    return f"{owner}/{repo}", tag, asset


class GitHubReleaseStore:
    def __init__(
        self,
        repo: str,
        token: str,
        *,
        tag: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.repo = repo
        self.tag = tag or f"snapshots-{datetime.now(UTC):%Y-%m}"
        # An asset download redirects to a CDN host that signs the request in
        # its query string and rejects one that also carries Authorization.
        # httpx drops that header when the origin changes.
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.client.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Authorization": f"Bearer {token}",
            }
        )
        self._releases: dict[str, int | None] = {}
        self._assets: dict[str, dict[str, int]] = {}

    def put(self, key: str, data: bytes) -> str:
        name = asset_name(key)
        assets = self._assets_for(self.tag)
        if name not in assets:
            release = self._ensure_release(self.tag)
            response = self.client.post(
                f"{UPLOAD_BASE}/repos/{self.repo}/releases/{release}/assets",
                params={"name": name},
                content=data,
                headers={"Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()
            assets[name] = int(response.json()["id"])
        return f"gh://{self.repo}/{self.tag}/{name}"

    def get(self, uri: str) -> bytes:
        repo, tag, name = parse_uri(uri)
        if repo != self.repo:
            raise ValueError(f"{uri} belongs to {repo}, not {self.repo}")

        asset = self._assets_for(tag).get(name)
        if asset is None:
            raise FileNotFoundError(uri)

        response = self.client.get(
            f"{API_BASE}/repos/{repo}/releases/assets/{asset}",
            headers={"Accept": "application/octet-stream"},
        )
        response.raise_for_status()
        return response.content

    def exists(self, key: str) -> bool:
        return asset_name(key) in self._assets_for(self.tag)

    def _release_id(self, tag: str) -> int | None:
        if tag not in self._releases:
            response = self.client.get(f"{API_BASE}/repos/{self.repo}/releases/tags/{tag}")
            if response.status_code == httpx.codes.NOT_FOUND:
                self._releases[tag] = None
            else:
                response.raise_for_status()
                self._releases[tag] = int(response.json()["id"])
        return self._releases[tag]

    def _ensure_release(self, tag: str) -> int:
        found = self._release_id(tag)
        if found is None:
            response = self.client.post(
                f"{API_BASE}/repos/{self.repo}/releases",
                json={"tag_name": tag, "name": tag},
            )
            response.raise_for_status()
            found = int(response.json()["id"])
            self._releases[tag] = found
        return found

    def _assets_for(self, tag: str) -> dict[str, int]:
        # put() records an upload in what this returns, so it has to hand back
        # the cached dict and not a copy.
        if tag in self._assets:
            return self._assets[tag]

        self._assets[tag] = {}
        release = self._release_id(tag)
        if release is None:
            return self._assets[tag]

        page = 1
        while True:
            response = self.client.get(
                f"{API_BASE}/repos/{self.repo}/releases/{release}/assets",
                params={"per_page": PAGE_SIZE, "page": page},
            )
            response.raise_for_status()
            batch = response.json()
            self._assets[tag].update({a["name"]: int(a["id"]) for a in batch})
            if len(batch) < PAGE_SIZE:
                return self._assets[tag]
            page += 1
