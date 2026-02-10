from __future__ import annotations
import hashlib, json, os
from dataclasses import dataclass
from typing import Any, Dict, Optional

def _stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]

@dataclass(frozen=True)
class DiskCache:
    root: str

    def path_for(self, key_obj: Any, suffix: str = ".json") -> str:
        h = _stable_hash(key_obj)
        sub = os.path.join(self.root, h[:2], h[2:4])
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, f"{h}{suffix}")

    def get_json(self, key_obj: Any) -> Optional[Dict[str, Any]]:
        p = self.path_for(key_obj)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def set_json(self, key_obj: Any, value: Dict[str, Any]) -> None:
        p = self.path_for(key_obj)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        os.replace(tmp, p)
