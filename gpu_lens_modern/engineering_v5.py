from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class BatchJob:
    name: str
    parameters: dict
    gpu: str = "AUTO"
    status: str = "pending"
    result: dict = field(default_factory=dict)
    error: str = ""


class CheckpointBatch:
    """Resumable queue; GPU identifiers are assigned round-robin to worker callbacks."""

    def __init__(self, checkpoint: str | Path, jobs: list[BatchJob] | None = None):
        self.path = Path(checkpoint)
        self.jobs = jobs or []
        self.stop_event = threading.Event()

    @classmethod
    def load(cls, path: str | Path) -> "CheckpointBatch":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, [BatchJob(**item) for item in data.get("jobs", [])])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "jobs": [job.__dict__ for job in self.jobs]}
        fd, temp = tempfile.mkstemp(prefix=self.path.name, dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp): os.unlink(temp)

    def run(self, worker: Callable[[dict, str], dict], devices: list[str] | None = None,
            callback: Callable[[int, BatchJob], None] | None = None):
        devices = devices or ["AUTO"]
        for index, job in enumerate(self.jobs):
            if self.stop_event.is_set(): break
            if job.status == "completed": continue
            job.gpu = devices[index % len(devices)]
            job.status = "running"; self.save()
            try:
                job.result = worker(job.parameters, job.gpu)
                job.status = "completed"
            except Exception as exc:
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "failed"
            self.save()
            if callback: callback(index, job)
