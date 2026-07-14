"""
pipeline_logger.py
------------------
Lightweight structured logger for the DRS pipeline.
Logs each stage (detect, track, predict, classify, render) with timing
information to help profile and debug delivery processing.

Usage:
    from drs_opencv.pipeline_logger import PipelineLogger

    logger = PipelineLogger(job_id="abc123", verbose=True)
    with logger.stage("detection"):
        detection = detector.detect(frame)
    logger.log_metric("frames_with_ball", 42)
    logger.finish()
    summary = logger.get_summary()
"""

import time
import json
import os
import datetime
from contextlib import contextmanager


class PipelineLogger:
    """
    Records timing and metrics for each stage of the DRS pipeline.
    """

    STAGES = ["detection", "tracking", "prediction", "classification", "rendering"]

    def __init__(self, job_id="", verbose=False):
        self.job_id    = job_id
        self.verbose   = verbose
        self._start    = time.perf_counter()
        self._stages   = {}   # stage_name -> {'start', 'end', 'duration_ms'}
        self._metrics  = {}   # arbitrary key-value metrics
        self._errors   = []   # list of error strings
        self._warnings = []   # list of warning strings

    # ── Stage context manager ──────────────────────────────────────────

    @contextmanager
    def stage(self, name: str):
        """
        Context manager that times a pipeline stage.

        Usage:
            with logger.stage("detection"):
                # ... detection code ...
        """
        t0 = time.perf_counter()
        if self.verbose:
            print(f"[DRS] ▶ {name} …")
        try:
            yield
        finally:
            t1 = time.perf_counter()
            ms = round((t1 - t0) * 1000, 2)
            self._stages[name] = {
                "start_offset_ms": round((t0 - self._start) * 1000, 2),
                "duration_ms":     ms,
            }
            if self.verbose:
                print(f"[DRS] ✓ {name} — {ms:.1f} ms")

    # ── Metric / error / warning logging ──────────────────────────────

    def log_metric(self, key: str, value):
        """Record an arbitrary pipeline metric."""
        self._metrics[key] = value

    def log_error(self, message: str):
        self._errors.append(message)
        if self.verbose:
            print(f"[DRS] ✗ ERROR: {message}")

    def log_warning(self, message: str):
        self._warnings.append(message)
        if self.verbose:
            print(f"[DRS] ⚠ {message}")

    # ── Summary ───────────────────────────────────────────────────────

    def finish(self):
        """Mark the pipeline as complete."""
        self._total_ms = round((time.perf_counter() - self._start) * 1000, 2)

    def get_summary(self) -> dict:
        """Return a structured summary dict."""
        return {
            "job_id":     self.job_id,
            "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
            "total_ms":   getattr(self, "_total_ms", None),
            "stages":     self._stages,
            "metrics":    self._metrics,
            "errors":     self._errors,
            "warnings":   self._warnings,
        }

    def save(self, output_dir: str) -> str:
        """Save the summary as pipeline_log.json in output_dir."""
        self.finish()
        path = os.path.join(output_dir, "pipeline_log.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2)
        return path

    def __repr__(self):
        return f"<PipelineLogger job={self.job_id} stages={list(self._stages)}>"
