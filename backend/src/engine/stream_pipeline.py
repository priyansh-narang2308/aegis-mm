"""
Aegis-MM Asynchronous Streaming Pipeline Orchestrator
Coordinates sliding-window session ring buffers, dynamic request batching,
and PyTorch multimodal forward passes with sub-millisecond stage telemetry.
"""
import time
import torch
from typing import Dict, Any, Optional
from src.vision.attention_net import VisionAnomalyPipeline
from src.audio.spectral_net import SpectralAudioNet
from src.peft.cognitive_alignment import CognitiveAlignmentModule
from src.fusion.multimodal_guardrail import MultimodalGuardrailFusionNet
from src.engine.ring_buffer import StreamRingBuffer
from src.engine.batching_scheduler import DynamicBatchingScheduler
from src.core.telemetry import global_tracer


class AsyncStreamPipeline:
    """
    Master production streaming pipeline. Maintains multi-session state and orchestrates
    unified low-latency inference passes.
    """
    def __init__(
        self,
        vision_dim: int = 128,
        audio_dim: int = 128,
        batch_window_ms: float = 50.0,
        max_batch_size: int = 16
    ):
        self.vision_model = VisionAnomalyPipeline(feature_dim=vision_dim)
        self.audio_model = SpectralAudioNet(feature_dim=audio_dim)
        self.cognitive_model = CognitiveAlignmentModule(vision_dim=vision_dim, audio_dim=audio_dim)
        self.fusion_model = MultimodalGuardrailFusionNet(vision_dim=vision_dim, audio_dim=audio_dim)
        
        self.vision_model.eval()
        self.audio_model.eval()
        self.cognitive_model.eval()
        self.fusion_model.eval()
        
        self.sessions: Dict[str, StreamRingBuffer] = {}
        
        self.scheduler = DynamicBatchingScheduler(
            model_inference_fn=self._batched_forward_pass,
            batching_window_ms=batch_window_ms,
            max_batch_size=max_batch_size
        )

    def get_or_create_session(self, session_id: str) -> StreamRingBuffer:
        if session_id not in self.sessions:
            self.sessions[session_id] = StreamRingBuffer(capacity=300)
        return self.sessions[session_id]

    def _batched_forward_pass(self, batch_payload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Synchronous batched forward pass executed by worker task.
        """
        t_total = time.perf_counter()
        
        # 1. Vision stage
        t0 = time.perf_counter()
        v_out = self.vision_model(batch_payload["video_frame"])
        global_tracer.record_stage("vision_ms", (time.perf_counter() - t0) * 1000.0)
        
        # 2. Audio stage
        t0 = time.perf_counter()
        a_out = self.audio_model(batch_payload["audio_chunk"])
        global_tracer.record_stage("audio_ms", (time.perf_counter() - t0) * 1000.0)
        
        # 3. Cognitive LoRA stage
        t0 = time.perf_counter()
        c_out = self.cognitive_model(v_out["vision_embedding"], a_out["audio_embedding"], batch_payload["telemetry"])
        global_tracer.record_stage("cognitive_ms", (time.perf_counter() - t0) * 1000.0)
        
        # 4. Multimodal Late Fusion stage
        t0 = time.perf_counter()
        f_out = self.fusion_model(
            v_out["vision_embedding"], a_out["audio_embedding"], c_out["cognitive_embedding"], batch_payload["telemetry"]
        )
        global_tracer.record_stage("fusion_ms", (time.perf_counter() - t0) * 1000.0)
        
        global_tracer.record_stage("total_ms", (time.perf_counter() - t_total) * 1000.0)
        
        # Assemble batched output dictionary
        return {
            "overall_risk_score": f_out["overall_risk_score"],
            "ai_fluency_score": c_out["ai_fluency_score"],
            "synchronization_discrepancy": f_out["synchronization_discrepancy"],
            "gaze_anomaly_score": v_out["gaze_anomaly_score"],
            "synthetic_speech_score": a_out["synthetic_speech_score"],
            "integrity_confidence": c_out["integrity_confidence"],
        }

    async def ingest_packet_and_predict(self, session_id: str, packet: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Ingests real-time stream packet into session ring buffer and schedules batched inference.
        """
        buffer = self.get_or_create_session(session_id)
        buffer.append(packet)
        
        result = await self.scheduler.submit_request(session_id, packet)
        
        return {
            "session_id": session_id,
            "timestamp": time.time(),
            "overall_risk_score": round(float(result["overall_risk_score"].item()), 4),
            "ai_fluency_score": round(float(result["ai_fluency_score"].item()), 4),
            "synchronization_discrepancy": round(float(result["synchronization_discrepancy"].item()), 4),
            "gaze_anomaly_score": round(float(result["gaze_anomaly_score"].item()), 4),
            "synthetic_speech_score": round(float(result["synthetic_speech_score"].item()), 4),
            "integrity_confidence": round(float(result["integrity_confidence"].item()), 4),
            "buffer_length": len(buffer)
        }

    def start(self):
        self.scheduler.start()

    async def stop(self):
        await self.scheduler.stop()
