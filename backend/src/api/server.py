"""
Aegis-MM Master FastAPI Application Server
Entrypoint combining REST routers, WebSocket streaming endpoints, and CORS
middleware for Next.js interactive frontend dashboard integration.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router as rest_router, pipeline
from .websockets import ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.start()
    yield
    await pipeline.stop()


app = FastAPI(
    title="Aegis-MM Real-Time Multimodal Guardrail API",
    description="High-performance PyTorch + FastAPI streaming server for interview integrity and AI Fluency monitoring.",
    version="0.1.0",
    lifespan=lifespan
)

# Allow Next.js frontend (default port 3000) and local dev environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rest_router)
app.include_router(ws_router)
