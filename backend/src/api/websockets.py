
import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .routes import pipeline, generator
from src.core.telemetry import logger

ws_router = APIRouter()


@ws_router.websocket("/ws/stream/{session_id}")
async def websocket_stream_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connected for Session ID: {session_id}")
    
    current_anomaly = None
    try:
        while True:
            # Receive client control commands or simulated streaming triggers
            try:
                msg_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.033)  # ~30 FPS polling
                data = json.loads(msg_text)
                if "anomaly" in data:
                    current_anomaly = data["anomaly"] if data["anomaly"] != "none" else None
            except asyncio.TimeoutError:
                pass  # Continuous streaming loop when no new command arrives
                
            packet = generator.get_next_frame(anomaly_type=current_anomaly)
            telemetry = await pipeline.ingest_packet_and_predict(session_id, packet)
            
            await websocket.send_json(telemetry)
            await asyncio.sleep(0.033)  # Regulate stream to 30 FPS
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for Session ID: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket streaming error for {session_id}: {str(e)}")
