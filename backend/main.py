"""
main.py
=======
FastAPI server for the Bengaluru Two-Wheeler Rider
Skin Health & Sun Exposure Advisor.

Endpoints:
  POST /api/advisory          — Full 6-agent analysis pipeline
  POST /api/agent-chat        — Interactive AI Agent Console
  GET  /api/analytics         — Weekly dashboard analytics
  GET  /api/health            — Health check
  GET  /                      — Serve frontend
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents_engine import (
    AdvisoryPipeline,
    AnalyticsDashboardAgent,
    RiderProfile,
    TripRequest,
    _apply_guardrails,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
log = structlog.get_logger()

app = FastAPI(
    title="Bengaluru Rider Skin Health & Sun Exposure Advisor",
    description="Multi-Agent AI advisory powered by IBM Granite on watsonx.ai",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Singletons
pipeline         = AdvisoryPipeline()
analytics_agent  = AnalyticsDashboardAgent()


# ─────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────

class AdvisoryRequest(BaseModel):
    route:          str   = "Electronic City – Indiranagar"
    duration_hours: float = 1.0
    departure_time: Optional[str] = None
    rider_id:       Optional[int] = None
    # Rider profile fields (flattened for easy frontend use)
    name:                   str   = "Rider"
    skin_type:              str   = "combination"
    skin_sensitivity:       str   = "moderate"
    uses_sunscreen:         bool  = False
    spf_rating:             str   = "none"
    reapplies_sunscreen:    bool  = False
    wears_arm_sleeves:      bool  = False
    wears_full_jacket:      bool  = False
    visor_type:             str   = "clear"
    wears_gloves:           bool  = False
    uses_face_mask_gaiter:  bool  = False


class AgentChatRequest(BaseModel):
    agent_name:  str
    message:     str
    session_id:  str = ""
    rider_id:    Optional[int] = None
    # Optional context
    route:              str   = ""
    duration_hours:     float = 1.0
    skin_sensitivity:   str   = "moderate"
    skin_type:          str   = "combination"
    uv_index:           float = 0.0
    pm25:               float = 0.0
    travel_verdict:     str   = ""
    risk_score:         float = 0.0


# ─────────────────────────────────────────────────────────────
# Observability Middleware
# ─────────────────────────────────────────────────────────────

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    log.info("http.request",
             method=request.method,
             path=request.url.path,
             status=response.status_code,
             elapsed_ms=elapsed,
             request_id=request_id)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed}ms"
    return response


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main React-free frontend."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h2>Frontend not found. Build/place frontend/index.html.</h2>", status_code=404)


@app.get("/api/health")
async def health_check():
    """Service health check — used by IBM Cloud Code Engine probes."""
    return {
        "status": "healthy",
        "service": "bengaluru-rider-advisor",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@app.post("/api/advisory")
async def get_advisory(req: AdvisoryRequest):
    """
    Run the full 6-agent advisory pipeline.
    Returns travel verdict, best windows, skincare advice, gear gaps.
    """
    try:
        rider = RiderProfile(
            name=req.name,
            skin_type=req.skin_type,
            skin_sensitivity=req.skin_sensitivity,
            uses_sunscreen=req.uses_sunscreen,
            spf_rating=req.spf_rating,
            reapplies_sunscreen=req.reapplies_sunscreen,
            wears_arm_sleeves=req.wears_arm_sleeves,
            wears_full_jacket=req.wears_full_jacket,
            visor_type=req.visor_type,
            wears_gloves=req.wears_gloves,
            uses_face_mask_gaiter=req.uses_face_mask_gaiter,
        )
        trip = TripRequest(
            rider=rider,
            route=req.route,
            duration_hours=req.duration_hours,
            departure_time=req.departure_time,
            rider_id=req.rider_id,
            session_id=str(uuid.uuid4()),
        )
        result = await pipeline.run(trip)
        return result.model_dump()

    except Exception as exc:
        log.error("advisory.error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Advisory pipeline error: {str(exc)}")


@app.post("/api/agent-chat")
async def agent_chat(req: AgentChatRequest):
    """
    Interactive AI Agent Console — chat with any specialist agent.
    """
    # Guardrail check
    blocked = _apply_guardrails(req.message)
    if blocked:
        return {"response": blocked, "agent": req.agent_name,
                "provider": "guardrail", "tokens": 0, "cost_usd": 0.0}

    if not req.session_id:
        req.session_id = str(uuid.uuid4())

    context = {
        "route": req.route,
        "duration_hours": req.duration_hours,
        "skin_sensitivity": req.skin_sensitivity,
        "skin_type": req.skin_type,
        "uv_index": req.uv_index,
        "pm25": req.pm25,
        "travel_verdict": req.travel_verdict,
        "risk_score": req.risk_score,
    }

    try:
        output = analytics_agent.chat(
            agent_name=req.agent_name,
            message=req.message,
            context=context,
            session_id=req.session_id,
            rider_id=req.rider_id,
        )
        return {
            "response":    output.advice,
            "agent":       req.agent_name,
            "session_id":  req.session_id,
            "provider":    output.llm_provider,
            "model":       output.llm_model,
            "tokens":      output.tokens_used,
            "cost_usd":    output.cost_usd,
        }
    except Exception as exc:
        log.error("agent_chat.error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/analytics")
async def get_analytics():
    """Weekly exposure analytics from MySQL."""
    try:
        data = analytics_agent.get_weekly_analytics()
        return data
    except Exception as exc:
        log.error("analytics.error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/routes")
async def list_routes():
    """Known Bengaluru commute corridors."""
    return {
        "routes": [
            {"id": "ec_indiranagar",    "label": "Electronic City – Indiranagar",     "avg_duration_h": 1.25},
            {"id": "whitefield_kor",    "label": "Whitefield – Koramangala ORR",      "avg_duration_h": 1.50},
            {"id": "hebbal_mg",         "label": "Hebbal – MG Road",                   "avg_duration_h": 1.00},
            {"id": "btm_marathahalli",  "label": "BTM Layout – Marathahalli",          "avg_duration_h": 1.75},
            {"id": "jpnagar_bellandur", "label": "JP Nagar – Bellandur",               "avg_duration_h": 0.75},
            {"id": "kr_puram_silk",     "label": "KR Puram – Silk Board",             "avg_duration_h": 1.00},
            {"id": "yeshwantpur_hsr",   "label": "Yeshwanthpur – HSR Layout",         "avg_duration_h": 1.50},
            {"id": "banaswadi_itpl",    "label": "Banaswadi – ITPL Whitefield",       "avg_duration_h": 1.25},
        ]
    }


# ─────────────────────────────────────────────────────────────
# Entry point (dev)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("APP_ENV", "development") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
