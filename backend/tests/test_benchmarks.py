"""
Unit tests for Aegis-MM Automated Benchmark Suite.
Verifies concurrency stress testing and percentile SLA calculations.
"""
import pytest
from src.benchmarks.load_tester import run_concurrency_load_test


@pytest.mark.asyncio
async def test_load_tester_execution():
    res = await run_concurrency_load_test(num_sessions=2, requests_per_session=5)
    assert res["concurrent_sessions"] == 2
    assert res["total_requests"] == 10
    assert "p50" in res["latency_ms"]
    assert res["latency_ms"]["p95"] < 50.0  # Must strictly satisfy sub-50ms SLA
