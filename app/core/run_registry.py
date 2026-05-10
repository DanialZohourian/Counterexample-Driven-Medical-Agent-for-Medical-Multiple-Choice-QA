from dataclasses import dataclass, field
from typing import Dict, Optional
from app.core.util import now_ms

@dataclass
class RunStatus:
    run_id: str
    created_ms: int = field(default_factory=now_ms)
    state: str = "PENDING"  # PENDING|RUNNING|DONE|ERROR
    message: str = ""
    metrics_path: Optional[str] = None
    trace_path: Optional[str] = None

class RunRegistry:
    def __init__(self) -> None:
        self._runs: Dict[str, RunStatus] = {}

    def create(self, run_id: str) -> RunStatus:
        st = RunStatus(run_id=run_id)
        self._runs[run_id] = st
        return st

    def get(self, run_id: str) -> Optional[RunStatus]:
        return self._runs.get(run_id)

    def update(self, run_id: str, **kwargs) -> None:
        st = self._runs.get(run_id)
        if not st:
            return
        for k, v in kwargs.items():
            setattr(st, k, v)

run_registry = RunRegistry()
