# app/util/bg_tasks.py
from __future__ import annotations
import threading, time
from typing import Dict, Any, Callable

_jobs: Dict[str, Dict[str, Any]] = {}

def start_job(name: str, target: Callable[[], Dict[str, Any]]):
    if name in _jobs and _jobs[name].get("status") in {"running", "done"}:
        return _jobs[name]
    _jobs[name] = {"status": "running", "started": time.time()}
    def _run():
        try:
            res = target()
            _jobs[name].update({"status": "done", "result": res, "finished": time.time()})
        except Exception as e:
            _jobs[name].update({"status": "error", "error": str(e), "finished": time.time()})
    threading.Thread(target=_run, daemon=True).start()
    return _jobs[name]

def get_job(name: str) -> Dict[str, Any]:
    return _jobs.get(name, {"status": "unknown"})
