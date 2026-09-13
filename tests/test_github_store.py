import json

import httpx
import pytest

from plpd.ingest.storage import GitHubReleaseStore, asset_name, parse_uri

REPO = "owner/plpd-snapshots"
TAG = "snapshots-backfill"


class FakeGitHub:
    def __init__(self, *, release: bool = True) -> None:
        self.releases: dict[str, int] = {TAG: 1} if release else {}
        self.assets: dict[int, tuple[int, str, bytes]] = {}
        self.uploads = 0
        self.next_id = 100

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        if method == "GET" and path.startswith(f"/repos/{REPO}/releases/tags/"):
            tag = path.rsplit("/", 1)[-1]
            if tag not in self.releases:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json={"id": self.releases[tag], "tag_name": tag})

        if method == "POST" and path == f"/repos/{REPO}/releases":
            body = json.loads(request.content)
            self.releases[body["tag_name"]] = 7
            return httpx.Response(201, json={"id": 7, "tag_name": body["tag_name"]})

        if method == "GET" and path.endswith("/assets"):
            release = int(path.split("/")[-2])
            listed = [
                {"id": asset, "name": name}
                for asset, (held, name, _) in self.assets.items()
                if held == release
            ]
            return httpx.Response(200, json=listed)

        if method == "POST" and path.endswith("/assets"):
            release = int(path.split("/")[-2])
            name = request.url.params["name"]
            self.next_id += 1
            self.assets[self.next_id] = (release, name, request.content)
            self.uploads += 1
            return httpx.Response(201, json={"id": self.next_id, "name": name})

        if method == "GET" and "/releases/assets/" in path:
            asset = int(path.rsplit("/", 1)[-1])
            return httpx.Response(200, content=self.assets[asset][2])

        raise AssertionError(f"unexpected {method} {path}")


def build(github: FakeGitHub, *, tag: str | None = TAG) -> GitHubReleaseStore:
    transport = httpx.MockTransport(github.handler)
    client = httpx.Client(transport=transport, follow_redirects=True)
    return GitHubReleaseStore(REPO, "token", tag=tag, client=client)


def test_asset_name_flattens_the_key() -> None:
    assert asset_name("fpl_core/GW3__fixtures/abc.parquet") == "fpl_core-GW3__fixtures-abc.parquet"


def test_parse_uri_splits_repo_tag_and_asset() -> None:
    assert parse_uri("gh://owner/repo/snapshots-2026-09/a-b-c.parquet") == (
        "owner/repo",
        "snapshots-2026-09",
        "a-b-c.parquet",
    )


@pytest.mark.parametrize("uri", ["/data/a.parquet", "gh://owner/repo/only-three"])
def test_parse_uri_rejects_anything_else(uri: str) -> None:
    with pytest.raises(ValueError, match="gh://"):
        parse_uri(uri)


def test_put_returns_a_uri_that_get_reads_back() -> None:
    store = build(FakeGitHub())

    uri = store.put("fpl_core/teams/abc.parquet", b"bytes")

    assert uri == f"gh://{REPO}/{TAG}/fpl_core-teams-abc.parquet"
    assert store.get(uri) == b"bytes"


def test_put_uploads_a_key_only_once() -> None:
    github = FakeGitHub()
    store = build(github)

    first = store.put("fpl_core/teams/abc.parquet", b"bytes")
    second = store.put("fpl_core/teams/abc.parquet", b"bytes")

    assert first == second
    assert github.uploads == 1


def test_put_creates_the_release_when_the_tag_is_new() -> None:
    github = FakeGitHub(release=False)
    store = build(github)

    store.put("fpl_core/teams/abc.parquet", b"bytes")

    assert TAG in github.releases


def test_exists_reports_what_was_put() -> None:
    store = build(FakeGitHub())

    assert not store.exists("fpl_core/teams/abc.parquet")
    store.put("fpl_core/teams/abc.parquet", b"bytes")
    assert store.exists("fpl_core/teams/abc.parquet")


def test_get_rejects_an_asset_the_release_does_not_hold() -> None:
    store = build(FakeGitHub())

    with pytest.raises(FileNotFoundError):
        store.get(f"gh://{REPO}/{TAG}/fpl_core-teams-gone.parquet")


def test_get_rejects_a_uri_for_another_repository() -> None:
    store = build(FakeGitHub())

    with pytest.raises(ValueError, match="elsewhere"):
        store.get(f"gh://elsewhere/archive/{TAG}/a.parquet")


def test_the_default_tag_is_the_current_month() -> None:
    store = build(FakeGitHub(), tag=None)

    assert store.tag.startswith("snapshots-20")
