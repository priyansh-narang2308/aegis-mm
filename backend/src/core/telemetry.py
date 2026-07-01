
import time
import logging
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(name)s : %(message)s",
)
logger = logging.getLogger("AegisMM.Telemetry")


class LatencyTracer:
    """
    High-precision latency telemetry tracker for multimodal streaming passes.
    Tracks execution times across vision, audio, and fusion modules.
    """
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.history: Dict[str, List[float]] = {
            "vision_ms": [],
            "audio_ms": [],
            "fusion_ms": [],
            "total_ms": [],
        }

    @contextmanager
    def measure(self, stage_name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if stage_name in self.history:
                self.history[stage_name].append(elapsed_ms)
                if len(self.history[stage_name]) > self.max_history:
                    self.history[stage_name].pop(0)

    def record_stage(self, stage_name: str, duration_ms: float):
        if stage_name not in self.history:
            self.history[stage_name] = []
        self.history[stage_name].append(duration_ms)
        if len(self.history[stage_name]) > self.max_history:
            self.history[stage_name].pop(0)

    def compute_percentiles(self, stage_name: str = "total_ms") -> Dict[str, float]:
        data = self.history.get(stage_name, [])
        if not data:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "count": 0}
        
        arr = np.array(data)
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "mean": round(float(np.mean(arr)), 2),
            "count": len(data),
        }

    def get_full_telemetry_summary(self) -> Dict[str, Any]:
        return {
            stage: self.compute_percentiles(stage)
            for stage in self.history.keys()
        }


# Global singleton tracer
global_tracer = LatencyTracer()
