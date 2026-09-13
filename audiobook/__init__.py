"""Novel-to-audiobook preprocessing front-end.

Turns a raw ``.txt`` novel into a fully-annotated ``script.csv`` (the single
source of truth) plus a global ``cast.json``. Rendering (AuK) is a separate,
stateless stage that only consumes the script.

See ``docs/audiobook-workflow.md`` for the design.
"""

from __future__ import annotations

from .schema import Cast, Role, ScriptRow

__all__ = ["Cast", "Role", "ScriptRow"]
