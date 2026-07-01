"""
Aegis-MM High-Concurrency Load Tester
Simulates N simultaneous real-time candidate video/audio streams under concurrent
async pressure to compute precise p50, p95, and p99 SLA percentiles.
"""
import time
import asyncio
import numpy as np
from typing import Dict, Any, List
from src.engine.stream_pipeline import AsyncStreamPipeline
from src.data.synthetic_stream import SyntheticStreamGenerator


async def run_concurrency_load_test(num_sessions: int = 16, requests_per_session: int = 10) -> Dict[str, Any]:
    """
    Simulates high-concurrency load against the streaming engine.
    Returns latency statistics and throughput metrics.
    """
    pipeline = AsyncStreamPipeline(batch_window_ms=20.0, max_batch_size=16)
    pipeline.start()
    
    # Warmup pass to prime PyTorch tensor memory and compilation caches
    warmup_gen = SyntheticStreamGenerator(seed=999)
    await pipeline.ingest_packet_and_predict("warmup_session", warmup_gen.get_next_frame("none"))
    
    generators = {f"candidate_{i}": SyntheticStreamGenerator(seed=i) for i in range(num_sessions)}
    latencies_ms: List[float] = []
    
    start_time = time.perf_counter()
    
    async def session_worker(session_id: str, gen: SyntheticStreamGenerator):
        for _ in range(requests_per_session):
            t0 = time.perf_counter()
            packet = gen.get_next_frame("none")
            await pipeline.ingest_packet_and_predict(session_id, packet)
            elapsed = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed)
            # Short yield simulating streaming network cadence
            await asyncio.sleep(0.005)
            
    tasks = [
        asyncio.create_task(session_worker(sid, gen))
        for sid, gen in generators.items()
    ]
    
    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time
    await pipeline.stop()
    
    total_requests = len(latencies_ms)
    throughput = total_requests / total_time
    
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))
    
    return {
        "concurrent_sessions": num_sessions,
        "total_requests": total_requests,
        "total_test_duration_sec": round(total_time, 3),
        "throughput_req_per_sec": round(throughput, 2),
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2)
        }
    }
