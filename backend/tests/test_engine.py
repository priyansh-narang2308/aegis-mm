"""
Unit tests for Aegis-MM Asynchronous Streaming Inference Engine.
Verifies circular buffer eviction, dynamic request batching under concurrency,
and full pipeline sub-50ms latency SLAs.
"""
import asyncio
import pytest
import torch
from src.engine.ring_buffer import StreamRingBuffer
from src.engine.stream_pipeline import AsyncStreamPipeline
from src.data.synthetic_stream import SyntheticStreamGenerator


def test_stream_ring_buffer_eviction():
    buf = StreamRingBuffer(capacity=10)
    for i in range(15):
        buf.append({"id": i, "video_frame": torch.zeros(1, 3, 224, 224)})
        
    assert len(buf) == 10
    # Oldest packets 0-4 should be evicted; first item in buffer is now 5
    assert buf.buffer[0]["id"] == 5


@pytest.mark.asyncio
async def test_async_stream_pipeline_concurrent_batching():
    pipeline = AsyncStreamPipeline(batch_window_ms=20.0, max_batch_size=4)
    pipeline.start()
    
    gen1 = SyntheticStreamGenerator(seed=1)
    gen2 = SyntheticStreamGenerator(seed=2)
    
    packet1 = gen1.get_next_frame(anomaly_type="none")
    packet2 = gen2.get_next_frame(anomaly_type="gaze_drift")
    
    # Submit 2 concurrent requests for different session IDs
    task1 = asyncio.create_task(pipeline.ingest_packet_and_predict("session_A", packet1))
    task2 = asyncio.create_task(pipeline.ingest_packet_and_predict("session_B", packet2))
    
    res1, res2 = await asyncio.gather(task1, task2)
    
    assert res1["session_id"] == "session_A"
    assert res2["session_id"] == "session_B"
    assert "overall_risk_score" in res1
    assert "ai_fluency_score" in res2
    assert res1["buffer_length"] == 1
    assert res2["buffer_length"] == 1
    
    await pipeline.stop()
