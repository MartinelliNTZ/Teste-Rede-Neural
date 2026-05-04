from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class Preferences:
    def __init__(self, path: Path | str = Path("config") / "preferences.json"):
        self.path = Path(path)
        self._data: Dict[str, Any] = {}

    def loadpreferences(self) -> Dict[str, Any]:
        if not self.path.is_file():
            self._data = {}
            return self._data
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            self._data = loaded if isinstance(loaded, dict) else {}
        except Exception:
            self._data = {}
        return self._data

    def savepreferences(self, data: Dict[str, Any] | None = None) -> None:
        if data is not None:
            self._data = data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

