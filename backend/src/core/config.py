
import os
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class VisionConfig:
    input_channels: int = 3
    image_height: int = 224
    image_width: int = 224
    feature_dim: int = 128
    attention_heads: int = 4
    dropout: float = 0.1


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    feature_dim: int = 128
    temporal_window_ms: int = 1000


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class EngineConfig:
    ring_buffer_capacity: int = 300  # Stores last 10 seconds at 30 FPS
    sliding_window_frames: int = 15   # 0.5 sec temporal window for forward pass
    batching_window_ms: float = 50.0  # Dynamic request batching collection window
    max_batch_size: int = 16
    use_4bit_quantization: bool = True
    target_p95_latency_ms: float = 50.0


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8000"))
    ws_ping_interval: float = 20.0
    ws_ping_timeout: float = 20.0
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"


@dataclass
class AppConfig:
    project_name: str = "Aegis-MM Real-Time Multimodal Guardrail"
    version: str = "0.1.0"
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "version": self.version,
            "vision": self.vision.__dict__,
            "audio": self.audio.__dict__,
            "lora": self.lora.__dict__,
            "engine": self.engine.__dict__,
            "server": self.server.__dict__,
        }


# Global immutable settings instance
settings = AppConfig()
