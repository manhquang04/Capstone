"""Small report gate used by Priority 27 experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .interfaces import ServerKnowledge, TransmittedUpdate
from .validators import HarnessResultValidator


def emit_validated_json_report(
    path: Path,
    payload: dict[str, Any],
    *,
    validator: HarnessResultValidator,
    attack_name: str,
    domain: str,
    transmitted: TransmittedUpdate,
    server_knowledge: ServerKnowledge,
) -> None:
    validator.validate_for_report(attack_name, domain, transmitted, server_knowledge)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

