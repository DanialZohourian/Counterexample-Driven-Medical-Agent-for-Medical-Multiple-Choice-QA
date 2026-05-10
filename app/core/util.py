import hashlib
import os
import time
from typing import Any
import orjson

def now_ms() -> int:
    return int(time.time() * 1000)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def dumps_json(obj: Any) -> str:
    return orjson.dumps(obj).decode("utf-8")

def is_substring(needle: str, haystack: str) -> bool:
    return needle in haystack
