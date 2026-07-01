"""
Aegis-MM REST API Routes
Exposes system health monitoring, synchronous single-frame evaluation,
and production latency telemetry metrics.
"""
from fastapi import APIRouter, HTTPException
from .schemas import HealthResponse, FrameAnalysisRequest, TelemetryResponse, MetricsSummaryResponse
from src.engine.stream_pipeline import AsyncStreamPipeline
from src.data.synthetic_stream import SyntheticStreamGenerator
from src.core.telemetry import global_tracer

router = APIRouter()

# Global pipeline instance shared across REST and WebSocket routers
pipeline = AsyncStreamPipeline()

generator = SyntheticStreamGenerator()


@router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def get_health():
    return HealthResponse(
        status="healthy",
        active_sessions=len(pipeline.sessions),
        quantization_mode="INT8 / INT4 Supported"
    )


@router.post("/analyze/frame", response_model=TelemetryResponse, tags=["Inference"])
async def analyze_single_frame(request: FrameAnalysisRequest):
    """
    Ingests synthetic or client-supplied packet into session ring buffer and returns multimodal telemetry.
    """
    packet = generator.get_next_frame(anomaly_type=request.inject_anomaly)
    telemetry = await pipeline.ingest_packet_and_predict(request.session_id, packet)
    return telemetry


@router.get("/metrics", response_model=MetricsSummaryResponse, tags=["Monitoring"])
async def get_latency_metrics():
    """
    Returns rolling sub-millisecond p50, p95, and p99 execution latencies across pipeline stages.
    """
    summary = global_tracer.get_full_telemetry_summary()
    return {"stages": summary}
