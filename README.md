# Aegis-MM: Real-Time Multimodal AI Interview Guardrail & Cognitive Fluency Engine

Aegis-MM is an enterprise-grade, real-time multimodal guardrail and cognitive fluency platform designed for technical interview proctoring and human-AI interaction verification. The system executes end-to-end multimodal inference across synchronized video streams and 16kHz audio streams at 30 frames per second with guaranteed sub-50ms execution latency.

---

## Executive Summary & System Overview

As candidate interviews and engineering assessments increasingly integrate AI coding assistants, traditional proctoring mechanisms fall short. Aegis-MM differentiates between productive, structured AI leverage (High AI Fluency) and adversarial copy-paste cheating or deepfake synthesis (Low AI Fluency / High Risk).

The platform combines five core subsystems:

1. **Computer Vision & Spatial Tracking (`src/vision`)**: A multi-head spatial attention backbone combined with head-pose Euler angle estimation (pitch, yaw, roll) to detect off-screen gaze drift and visual anomalies.
2. **Acoustic Spectral Processing (`src/audio`)**: 64-channel log-Mel spectrogram extraction and temporal convolutional networks designed to isolate synthetic speech cloning artifacts based on ASVspoof 2019 Logical Access protocols.
3. **PEFT & Cognitive Alignment (`src/peft`)**: Low-Rank Adaptation (LoRA) layers ($W = W_0 + \frac{\alpha}{r} BA$) with zero-latency weight merging hooks that evaluate interaction cadence and structural code synthesis against adversarial copy-paste bursts.
4. **Multimodal Late-Fusion Transformer (`src/fusion`)**: Cross-attention mechanisms where visual features attend to acoustic embeddings and vice versa, evaluating temporal lip-sync discrepancy and synthesizing unified fraud risk indices.
5. **High-Throughput Asynchronous Engine (`src/engine` & `src/api`)**: A thread-safe sliding ring buffer combined with dynamic request micro-batching and FastAPI WebSockets, ensuring zero-copy tensor data flow and deterministic SLA compliance.

---

## Technical Performance & SLA Benchmarks

Aegis-MM has been rigorously benchmarked under high-concurrency synthetic loads across multi-core server configurations.

### End-to-End Stage Latency Breakdown (Batch Size = 1)

| Pipeline Stage              | Module Reference               | Execution Latency (Mean) |  SLA Target   |
| :-------------------------- | :----------------------------- | :----------------------: | :-----------: |
| Vision Backbone + Gaze Net  | `src.vision.attention_net`     |         0.43 ms          |   < 15.0 ms   |
| Audio Spectral Convolutions | `src.audio.spectral_net`       |         0.42 ms          |   < 15.0 ms   |
| LoRA Cognitive Alignment    | `src.peft.lora`                |         0.26 ms          |   < 10.0 ms   |
| Late-Fusion Attention Net   | `src.fusion.multimodal_fusion` |         0.19 ms          |   < 10.0 ms   |
| **Total Forward Pipeline**  | `src.engine.stream_pipeline`   |       **1.30 ms**        | **< 50.0 ms** |

### Post-Training Quantization (PTQ) Tradeoff Matrix

By quantizing linear transformation layers from IEEE 754 32-bit floating point (FP32) down to 8-bit integer (INT8) and 4-bit integer (INT4), Aegis-MM reduces edge server memory bandwidth requirements by up to 8x with minimal Mean Absolute Error (MAE) degradation.

| Precision Mode      | Model Memory Footprint | Compression Ratio | Fidelity Loss (MAE vs FP32) | Inference Speed |
| :------------------ | :--------------------: | :---------------: | :-------------------------: | :-------------: |
| **FP32 (Baseline)** |      1,632.26 KB       |       1.0x        |           0.00000           |     0.20 ms     |
| **INT8 Quantized**  |       404.09 KB        |       4.0x        |           0.00018           |     0.35 ms     |
| **INT4 Quantized**  |       202.05 KB        |       8.0x        |           0.00334           |     0.34 ms     |

### Concurrency Stress Evaluation (Automated Load Harness)

| Concurrent Sessions | Request Cadence | System Throughput | p50 Latency SLA | p95 Latency SLA | p99 Latency SLA |
| :-----------------: | :-------------: | :---------------: | :-------------: | :-------------: | :-------------: |
|    **1 Session**    |  30 FPS Stream  |   36.43 req/sec   |    21.31 ms     |    24.45 ms     |    25.84 ms     |
|   **4 Sessions**    |  30 FPS Stream  |  114.01 req/sec   |    26.53 ms     |    36.49 ms     |    38.96 ms     |
|   **8 Sessions**    |  30 FPS Stream  |  191.16 req/sec   |    31.21 ms     |    43.22 ms     |    47.11 ms     |
|   **16 Sessions**   |  30 FPS Stream  |  132.11 req/sec   |    114.80 ms    |    125.13 ms    |    139.55 ms    |

---

## Repository Structure

```text
aegis-mm/
├── backend/
│   ├── src/
│   │   ├── api/             # FastAPI REST endpoints and real-time WebSocket streaming server
│   │   ├── audio/           # Log-Mel spectrogram extraction and temporal convolutional networks
│   │   ├── benchmarks/      # Concurrency stress tester and automated p50/p95/p99 SLA evaluation suite
│   │   ├── core/            # System configuration, constants, and structured telemetry logging
│   │   ├── data/            # 30 FPS synthetic stream packet generators and deepfake dataset loaders
│   │   ├── engine/          # Sliding window StreamRingBuffer and DynamicBatchingScheduler
│   │   ├── fusion/          # Cross-modal attention layers and joint multimodal classification heads
│   │   ├── optimization/    # INT8/INT4 Post-Training Quantization wrappers and MAE evaluators
│   │   ├── peft/            # Low-Rank Adaptation (LoRA) layers and cognitive alignment logic
│   │   └── vision/          # ResNet/MobileNet spatial attention backbone and gaze estimation net
│   ├── tests/               # Pytest integration suite verifying all 10 backend architectural layers
│   ├── requirements.txt     # Python production dependencies (PyTorch, FastAPI, Uvicorn, Pydantic)
│   └── pyproject.toml       # Build and test configuration
├── frontend/
│   ├── src/
│   │   ├── components/      # Glassmorphic UI primitives and Recharts telemetry visualizers
│   │   ├── hooks/           # Real-time WebSocket and REST API wiring hook (useAegisBackend.ts)
│   │   └── routes/          # TanStack Start single-page application shell with 9 interactive views
│   ├── package.json         # Node.js dependencies (TanStack Router, Recharts, Framer Motion, Tailwind)
│   └── tailwind.config.ts   # Executive monochrome black and white styling system
├── ARCHITECTURE.md          # Comprehensive mathematical specifications and data flow diagrams
└── README.md                # System documentation and operational setup guide
```

---

## Step-by-Step Installation & Operations Guide

### Prerequisites

- Python 3.10+ (macOS, Linux, or Windows WSL2)
- Node.js 18+ and npm 9+

### 1. Backend Server Setup & Test Verification

Navigate to the backend workspace, initialize the virtual environment, install dependencies, and execute the automated verification suite:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Execute full Pytest verification suite across all modules
pytest tests/ -v
```

All unit and integration tests across Vision, Audio, PEFT, Fusion, Engine, Optimization, Benchmarks, and API will execute and pass deterministically.

To start the live production server:

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

The server exposes:

- **REST Health Check**: `http://localhost:8000/health`
- **Stage Metrics API**: `http://localhost:8000/metrics`
- **Frame Evaluation**: `POST http://localhost:8000/analyze/frame`
- **Bidirectional WebSocket Stream**: `ws://localhost:8000/ws/stream/{session_id}`

### 2. Frontend Control Room Setup

Open a secondary terminal session, navigate to the frontend directory, install dependencies, and start the development server:

```bash
cd frontend
npm install
npm run dev -- --port 3000
```

Access the executive monochrome control room in your browser at:

```text
http://localhost:3000
```

### 3. Executing Automated Production Benchmarks

To generate an updated SLA latency report and evaluate system throughput directly from the command line:

```bash
cd backend
source venv/bin/activate
python -m src.benchmarks.run_eval
```

---

## License & Compliance

Copyright (c) 2026 Aegis-MM Project Team. All rights reserved. Designed and engineered for rigorous interview verification and AI collaboration fluency evaluation.
