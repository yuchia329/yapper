"""Configuration loading: ``pipeline.toml`` tunables + ``.env`` secrets.

Code reads tunables from here and never hard-codes thresholds. Per-movie working
directories are derived from the movie filename so re-runs are resumable.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "pipeline.toml"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w.-]+", "_", name).strip("_")
    return slug or "movie"


@dataclass
class Config:
    raw: dict
    repo_root: Path

    @property
    def artifacts_dir(self) -> Path:
        return self._resolve(self.raw["paths"]["artifacts_dir"])

    @property
    def scratch_dir(self) -> Path:
        return self._resolve(self.raw["paths"]["scratch_dir"])

    def _resolve(self, p: str) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() else self.repo_root / path

    def movie_dir(self, movie_path: str | Path) -> Path:
        """Per-movie working directory, e.g. artifacts/<slug>/."""
        slug = _slugify(Path(movie_path).stem)
        d = self.artifacts_dir / slug
        d.mkdir(parents=True, exist_ok=True)
        return d

    def section(self, name: str) -> dict:
        return self.raw.get(name, {})

    # convenience env accessors -------------------------------------------------
    @staticmethod
    def env(key: str, default: str | None = None) -> str | None:
        return os.environ.get(key, default)

    @staticmethod
    def require_env(key: str) -> str:
        val = os.environ.get(key)
        if not val:
            raise RuntimeError(f"required env var {key} is not set (see .env.example)")
        return val


def load_config(path: str | Path | None = None) -> Config:
    load_dotenv(REPO_ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    return Config(raw=raw, repo_root=REPO_ROOT)
