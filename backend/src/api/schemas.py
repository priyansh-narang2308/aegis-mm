
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class HealthResponse(BaseModel):
    status: str = "healthy"
    project: str = "Aegis-MM Real-Time Multimodal Guardrail"
    version: str = "0.1.0"
    active_sessions: int = 0
    quantization_mode: str = "INT8"


class FrameAnalysisRequest(BaseModel):
    session_id: str = Field(..., description="Unique identifier for candidate interview session")
    inject_anomaly: Optional[str] = Field(None, description="Optional adversarial anomaly: 'gaze_drift', 'tts_audio', or 'prompt_injection'")


class TelemetryResponse(BaseModel):
    session_id: str
    timestamp: float
    overall_risk_score: float = Field(..., ge=0.0, le=1.0)
    ai_fluency_score: float = Field(..., ge=0.0, le=1.0)
    synchronization_discrepancy: float = Field(..., ge=0.0, le=1.0)
    gaze_anomaly_score: float = Field(..., ge=0.0, le=1.0)
    synthetic_speech_score: float = Field(..., ge=0.0, le=1.0)
    integrity_confidence: float = Field(..., ge=0.0, le=1.0)
    buffer_length: int


class StageLatencyPercentiles(BaseModel):
    p50: float
    p95: float
    p99: float
    mean: float
    count: int


class MetricsSummaryResponse(BaseModel):
    stages: Dict[str, StageLatencyPercentiles]
