from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

@dataclass(frozen=True)
class ArtifactMeta:
    source: str
    downloaded_at: str
    available_at: str | None = None
    vintage: str | None = None
    license: str | None = None
    sha256: str | None = None
    status: str = "AVAILABLE"
    notes: str | None = None

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def write_meta(path: str | Path, meta: ArtifactMeta):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(meta), indent=2), encoding='utf-8')

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
