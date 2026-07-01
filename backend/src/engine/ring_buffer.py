"""
Aegis-MM Stream Ring Buffer
Fixed-capacity circular buffer storing candidate stream packets per session.
Enables sliding-window temporal aggregation for continuous integrity inference.
"""
import torch
from collections import deque
from typing import Dict, List, Optional


class StreamRingBuffer:
    """
    Per-session circular buffer holding recent video frames, audio chunks, and telemetry.
    Automatically evicts oldest packets when capacity is exceeded.
    """
    def __init__(self, capacity: int = 300):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, packet: Dict[str, torch.Tensor]):
        """
        packet expects keys: 'video_frame', 'audio_chunk', 'telemetry', 'timestamp'
        """
        self.buffer.append(packet)

    def get_sliding_window(self, window_frames: int = 15) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieves the most recent window_frames aggregated as stacked tensors.
        Returns None if buffer has fewer packets than window_frames.
        """
        if len(self.buffer) < window_frames:
            return None
        
        recent = list(self.buffer)[-window_frames:]
        
        # Stack temporal frames
        video_stack = torch.cat([p["video_frame"] for p in recent], dim=0) # (W, 3, 224, 224)
        audio_stack = torch.cat([p["audio_chunk"] for p in recent], dim=0) # (W, 16000)
        telemetry_stack = torch.cat([p["telemetry"] for p in recent], dim=0) # (W, 32)
        
        # Take temporal mean or latest representation for single-step inference
        latest_video = recent[-1]["video_frame"]
        latest_audio = recent[-1]["audio_chunk"]
        latest_telemetry = recent[-1]["telemetry"]
        
        return {
            "window_video": video_stack,
            "window_audio": audio_stack,
            "window_telemetry": telemetry_stack,
            "latest_video": latest_video,
            "latest_audio": latest_audio,
            "latest_telemetry": latest_telemetry,
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
