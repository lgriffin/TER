"""Control limits adapter behind the :class:`~ter.ports.driven.ControlLimitsSource` port.

:class:`JsonControlLimits` reads a ``ter.control-limits/1`` JSON file, the
document ``python -m ter control limits`` writes and a developer tunes by
hand. Validation is the domain's (``ControlLimits.from_mapping``), so the
file and an in-memory document obey the same rules.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...domain.lean.control import ControlLimits, ControlLimitsError

__all__ = ["JsonControlLimits"]


class JsonControlLimits:
    """Reads control limits from a JSON file named by ``ref``."""

    name = "json"

    def limits(self, ref: str | Path) -> ControlLimits:
        path = Path(ref).expanduser()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ControlLimitsError(f"{path}: no such limits file") from None
        except (OSError, UnicodeDecodeError) as exc:
            raise ControlLimitsError(f"{path}: cannot read: {exc}") from None
        except json.JSONDecodeError as exc:
            raise ControlLimitsError(f"{path}: not JSON: {exc}") from None
        try:
            return ControlLimits.from_mapping(document)
        except ControlLimitsError as exc:
            raise ControlLimitsError(f"{path}: {exc}") from None
