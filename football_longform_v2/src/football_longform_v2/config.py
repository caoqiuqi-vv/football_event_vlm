from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a LongForm V2 configuration violates a safety contract."""


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing {context}.{key}")
    return mapping[key]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    if raw.get("schema_version") != "football_longform_v2.alpha2":
        raise ConfigError("unsupported schema_version")

    task = _required(raw, "task", "config")
    features = _required(raw, "features", "config")
    acceptance = _required(raw, "acceptance", "config")
    families = list(_required(task, "proposal_families", "task"))
    if families != ["shot_chain", "restart", "generic_event"]:
        raise ConfigError(f"unexpected proposal families: {families}")
    for name in ("context", "motion"):
        stream = _required(features, name, "features")
        if not bool(stream.get("required", False)):
            raise ConfigError(f"features.{name} must be required for LF-A0")
        if int(stream.get("dim", 0)) <= 0:
            raise ConfigError(f"features.{name}.dim must be positive")
    if not bool(acceptance.get("rgb_only_checkpoint_selection", False)):
        raise ConfigError("RGB-only checkpoint selection is mandatory")
    if not bool(acceptance.get("forbid_detection_gated_proposals", False)):
        raise ConfigError("detection-gated proposals are forbidden")

    raw["_config_path"] = str(config_path)
    raw["_project_root"] = str(config_path.parent.parent.resolve())
    return raw

