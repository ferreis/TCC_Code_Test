from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / 'workspace' / 'registry.yaml'


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_path = _resolve(path)
    if not registry_path.exists():
        return {'version': 1, 'entries': {}}
    data = yaml.safe_load(registry_path.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        return {'version': 1, 'entries': {}}
    data.setdefault('version', 1)
    data.setdefault('entries', {})
    return data


def save_registry(registry: dict[str, Any], path: str | Path = DEFAULT_REGISTRY) -> Path:
    registry_path = _resolve(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return registry_path


def list_entries(kind: str, path: str | Path = DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    registry = load_registry(path)
    items = registry.get('entries', {}).get(kind, []) or []
    return [dict(item) for item in items if isinstance(item, dict)]


def add_entry(
    kind: str,
    name: str,
    value: str,
    *,
    metadata: dict[str, Any] | None = None,
    path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise ValueError('name e value são obrigatórios.')
    registry = load_registry(path)
    entries = registry.setdefault('entries', {}).setdefault(kind, [])
    entry = {'name': name, 'value': value}
    if metadata:
        entry.update(metadata)
    for idx, current in enumerate(entries):
        if current.get('name') == name or current.get('value') == value:
            entries[idx] = entry
            break
    else:
        entries.append(entry)
    save_registry(registry, path)
    return entry
