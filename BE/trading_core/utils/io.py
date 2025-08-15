# BE/trading_core/utils/io.py
"""
Lightweight file I/O helpers.
- YAML + JSON loaders
- Atomic writes for text/JSON
- Safe directory ensure

No runtime dependency on the rest of the app.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except Exception as e:  # pragma: no cover
    yaml = None  # type: ignore


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: Path | str) -> Dict[str, Any]:
    """
    Load a YAML file into a dict. Returns {} if file missing or parser not available.
    """
    p = Path(path)
    if not p.exists() or yaml is None:
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return data
        # allow top-level lists; wrap for callers expecting dict-like
        return {"_": data}


def read_json(path: Path | str, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_text(path: Path | str, content: str) -> None:
    """
    Write text to file atomically (write to temp → replace).
    """
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(content)
    tmp.replace(p)


def write_json(path: Path | str, data: Any, *, indent: int = 2) -> None:
    """
    Serialize to JSON with an atomic write.
    """
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    tmp.replace(p)
