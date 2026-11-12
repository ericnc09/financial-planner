"""
Publishes digest JSON to a dedicated `digests` branch of this repository, which
ericcosta.ca/finalyse reads. The branch's history doubles as a timestamped record
of what was published and when.
"""

import json
import os
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger()

BRANCH = "digests"
DEFAULT_CLONE_DIR = Path("data/digests")
GIT_TIMEOUT_SECONDS = 120


class DigestPublisher:
    def __init__(self, clone_dir: Path = DEFAULT_CLONE_DIR, remote_url: str | None = None):
        self.clone_dir = Path(clone_dir)
        self.remote_url = remote_url

    def read_latest(self) -> dict | None:
        """Return the most recently published digest, or None if there isn't one."""
        try:
            self._sync()
            return json.loads((self.clone_dir / "latest.json").read_text())
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            logger.info("digest.no_previous", detail=str(e)[:200])
            return None

    def publish(self, digest: dict) -> bool:
        """Write the day's files and push them. Returns False on any failure."""
        try:
            self._sync()
            body = json.dumps(digest, separators=(",", ":"))
            (self.clone_dir / f"{digest['date']}.json").write_text(body)
            (self.clone_dir / "latest.json").write_text(body)
            self._write_index(digest["date"])

            self._git("add", "-A")
            if not self._git("status", "--porcelain").stdout.strip():
                return True
            self._git("commit", "-m", f"Digest for {digest['date']}")
            self._git("push", "origin", BRANCH)
            return True
        except (OSError, subprocess.SubprocessError) as e:
            detail = getattr(e, "stderr", None) or str(e)
            logger.warning("digest.publish_failed", error=str(detail)[:300])
            return False

    def _write_index(self, date: str) -> None:
        path = self.clone_dir / "index.json"
        dates = json.loads(path.read_text())["dates"] if path.exists() else []
        path.write_text(json.dumps({"dates": sorted({*dates, date}, reverse=True)}))

    def _sync(self) -> None:
        """Bring the local clone level with the remote branch, creating both on
        first use. This clone is used only for publishing."""
        if not (self.clone_dir / ".git").exists():
            self.clone_dir.mkdir(parents=True, exist_ok=True)
            self._git("init", "--quiet")
            self._git("remote", "add", "origin", self.remote_url or self._origin_url())

        if self._git("ls-remote", "--heads", "origin", BRANCH).stdout.strip():
            self._git("fetch", "--quiet", "origin", BRANCH)
            self._git("checkout", "--quiet", "-B", BRANCH, "FETCH_HEAD")
        elif self._git("rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip() != BRANCH:
            self._git("checkout", "--quiet", "--orphan", BRANCH)

    def _origin_url(self) -> str:
        return subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True, timeout=GIT_TIMEOUT_SECONDS,
        ).stdout.strip()

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        # BatchMode makes SSH fail rather than hang waiting for input under cron
        env = {**os.environ, "GIT_SSH_COMMAND": "ssh -o BatchMode=yes"}
        return subprocess.run(
            ["git", *args],
            cwd=self.clone_dir,
            capture_output=True,
            text=True,
            check=check,
            timeout=GIT_TIMEOUT_SECONDS,
            env=env,
        )
