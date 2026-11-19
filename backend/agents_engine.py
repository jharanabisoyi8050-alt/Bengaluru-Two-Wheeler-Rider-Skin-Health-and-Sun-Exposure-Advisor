"""
agents_engine.py  (v2 — Enhanced)
==================================
Multi-Agent AI Engine for Bengaluru Two-Wheeler Rider
Skin Health & Sun Exposure Advisor.

New in v2:
  - Richer telemetry: wind_speed, cloud_cover, apparent_temp, uv_index_clear_sky,
    dew_point, visibility, surface_pressure, weather_code, AQI categories
  - Customizable routes with estimated distance, typical traffic multiplier
  - Pre-ride safety gear reminder agent (Agent 7)
  - Full Groq fallback with live API calls
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx
import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Structured logger
# ─────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
IBM_CLOUD_API_KEY   = os.getenv("IBM_CLOUD_API_KEY", "")
WATSONX_URL         = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID  = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_MODEL_ID    = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_ID       = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-20b")

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_ID     = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

HITL_WEBHOOK_URL    = os.getenv("HITL_WEBHOOK_URL", "")
HITL_RISK_THRESHOLD = float(os.getenv("HITL_RISK_THRESHOLD", "15.0"))

COST_PER_1K = {
    "granite": float(os.getenv("GRANITE_COST_PER_1K_TOKENS", "0.00008")),
    "groq":    float(os.getenv("GROQ_COST_PER_1K_TOKENS",    "0.00006")),
    "gemini":  float(os.getenv("GEMINI_COST_PER_1K_TOKENS",  "0.00010")),
}

BLR_LAT = 12.9716
BLR_LON = 77.5946

# WMO weather code descriptions
WMO_CODES = {
    0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
    45:"Fog", 48:"Icy fog", 51:"Light drizzle", 53:"Moderate drizzle",
    55:"Dense drizzle", 61:"Slight rain", 63:"Moderate rain", 65:"Heavy rain",
    71:"Slight snow", 80:"Light showers", 81:"Moderate showers", 82:"Heavy showers",
    95:"Thunderstorm", 96:"Thunderstorm with hail", 99:"Thunderstorm heavy hail",
}

# ─────────────────────────────────────────────────────────────
# Customizable Routes Registry
# ─────────────────────────────────────────────────────────────
BENGALURU_ROUTES: Dict[str, Dict] = {
    "Electronic City – Indiranagar": {
        "distance_km": 22.5, "typical_duration_h": 1.25,
        "traffic_multiplier": 1.4,
        "peak_hours": ["08:00-10:00", "17:30-20:00"],
        "major_junctions": ["Silk Board", "BTM Layout", "Koramangala 5th Block"],
        "avg_speed_kmh": 18,
        "highway_stretch": True,
        "open_exposure_pct": 85,
    },
    "Whitefield – Koramangala ORR": {
        "distance_km": 28.0, "typical_duration_h": 1.50,
        "traffic_multiplier": 1.6,
        "peak_hours": ["08:00-10:30", "18:00-20:30"],
        "major_junctions": ["ITPL Signal", "Marathahalli Bridge", "Silk Board"],
        "avg_speed_kmh": 19,
        "highway_stretch": True,
        "open_exposure_pct": 90,
    },
    "Hebbal – MG Road": {
        "distance_km": 14.0, "typical_duration_h": 1.00,
        "traffic_multiplier": 1.3,
        "peak_hours": ["08:30-10:00", "17:00-19:30"],
        "major_junctions": ["Hebbal Flyover", "Mekhri Circle", "Sadashivanagar"],
        "avg_speed_kmh": 14,
        "highway_stretch": False,
        "open_exposure_pct": 70,
    },
    "BTM Layout – Marathahalli": {
        "distance_km": 18.5, "typical_duration_h": 1.75,
        "traffic_multiplier": 1.7,
        "peak_hours": ["08:00-10:30", "18:00-21:00"],
        "major_junctions": ["Silk Board Junction", "HSR Sector 1", "Outer Ring Road"],
        "avg_speed_kmh": 11,
        "highway_stretch": True,
        "open_exposure_pct": 88,
    },
    "JP Nagar – Bellandur": {
        "distance_km": 12.0, "typical_duration_h": 0.75,
        "traffic_multiplier": 1.2,
        "peak_hours": ["08:30-09:30", "18:00-19:30"],
        "major_junctions": ["Bannerghatta Road", "JP Nagar 7th Phase"],
        "avg_speed_kmh": 16,
        "highway_stretch": False,
        "open_exposure_pct": 75,
    },
    "KR Puram – Silk Board": {
        "distance_km": 20.0, "typical_duration_h": 1.00,
        "traffic_multiplier": 1.5,
        "peak_hours": ["07:30-10:00", "17:30-20:00"],
        "major_junctions": ["KR Puram Bridge", "Marathahalli", "Outer Ring Road"],
        "avg_speed_kmh": 20,
        "highway_stretch": True,
        "open_exposure_pct": 92,
    },
    "Yeshwanthpur – HSR Layout": {
        "distance_km": 25.0, "typical_duration_h": 1.50,
        "traffic_multiplier": 1.4,
        "peak_hours": ["08:00-10:00", "17:00-19:30"],
        "major_junctions": ["Yeshwanthpur Circle", "Rajajinagar", "Silk Board"],
        "avg_speed_kmh": 17,
        "highway_stretch": False,
        "open_exposure_pct": 78,
    },
    "Banaswadi – ITPL Whitefield": {
        "distance_km": 22.0, "typical_duration_h": 1.25,
        "traffic_multiplier": 1.3,
        "peak_hours": ["08:00-09:30", "18:30-20:00"],
        "major_junctions": ["Tin Factory", "KR Puram", "Whitefield Main Road"],
        "avg_speed_kmh": 18,
        "highway_stretch": True,
        "open_exposure_pct": 85,
    },
}

def get_route_info(route_name: str) -> Dict:
    """Return route metadata, or a sensible default for custom routes."""
    if route_name in BENGALURU_ROUTES:
        return BENGALURU_ROUTES[route_name]
    # Custom route — return defaults
    return {
        "distance_km": 15.0, "typical_duration_h": 1.0,
        "traffic_multiplier": 1.3,
        "peak_hours": ["08:00-10:00", "17:30-19:30"],
        "major_junctions": ["Custom route"],
        "avg_speed_kmh": 15,
        "highway_stretch": True,
        "open_exposure_pct": 80,
    }

# ─────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────

class RiderProfile(BaseModel):
    name:             str  = "Rider"
    skin_type:        str  = "combination"
    skin_sensitivity: str  = "moderate"
    uses_sunscreen:         bool = False
    spf_rating:             str  = "none"
    reapplies_sunscreen:    bool = False
    wears_arm_sleeves:      bool = False
    wears_full_jacket:      bool = False
    visor_type:             str  = "clear"
    wears_gloves:           bool = False
    uses_face_mask_gaiter:  bool = False
    # Pre-ride reminder preferences
    reminder_minutes_before: int  = 15   # how many min before ride to get reminder
    reminder_enabled:        bool = True


class TripRequest(BaseModel):
    rider:          RiderProfile
    route:          str           = "Electronic City – Indiranagar"
    custom_start:   Optional[str] = None   # e.g. "Jayanagar 4th Block"
    custom_end:     Optional[str] = None   # e.g. "Manyata Tech Park"
    duration_hours: float         = 1.0
    departure_time: Optional[str] = None
    rider_id:       Optional[int] = None
    session_id:     str           = Field(default_factory=lambda: str(uuid.uuid4()))


class TelemetrySnapshot(BaseModel):
    # Core
    uv_index:              float
    uv_index_clear_sky:    float = 0.0
    temperature_c:         float
    apparent_temp_c:       float = 0.0
    humidity_pct:          float
    dew_point_c:           float = 0.0
    precipitation_prob:    int
    precipitation_mm:      float = 0.0
    wind_speed_kmh:        float = 0.0
    wind_direction_deg:    int   = 0
    cloud_cover_pct:       int   = 0
    visibility_km:         float = 10.0
    surface_pressure_hpa:  float = 1013.0
    weather_code:          int   = 0
    weather_description:   str   = "Clear sky"
    # AQI
    pm25:                  float
    pm10:                  float = 0.0
    dust:                  float
    ozone:                 float = 0.0
    nitrogen_dioxide:      float = 0.0
    aqi_category:          str   = "Moderate"
    # Hourly arrays (24h)
    hourly_uv:             List[float]
    hourly_pm25:           List[float]
    hourly_precip:         List[int]
    hourly_temp:           List[float]  = Field(default_factory=list)
    hourly_wind:           List[float]  = Field(default_factory=list)
    hourly_cloud:          List[int]    = Field(default_factory=list)
    source:                str = "open-meteo"


class GearReminder(BaseModel):
    gear_item:      str
    priority:       str   # critical | high | recommended
    reason:         str
    emoji:          str


class AgentOutput(BaseModel):
    agent:         str
    verdict:       Optional[str]       = None
    score:         Optional[float]     = None
    best_windows:  Optional[List[Dict]] = None
    advice:        Optional[str]       = None
    gaps:          Optional[List[str]] = None
    gear_reminders: Optional[List[GearReminder]] = None
    llm_provider:  str   = "none"
    llm_model:     str   = "none"
    tokens_used:   int   = 0
    cost_usd:      float = 0.0
    raw_response:  str   = ""


class FullAdvisoryResponse(BaseModel):
    session_id:           str
    timestamp:            str
    telemetry:            TelemetrySnapshot
    route_info:           Dict
    risk_score:           float
    travel_verdict:       str
    verdict_explanation:  str
    best_travel_windows:  List[Dict]
    skincare_advice:      str
    protection_gaps:      List[str]
    gear_recommendations: str
    gear_reminders:       List[GearReminder]
    reminder_message:     str
    agents:               List[AgentOutput]
    total_tokens:         int
    total_cost_usd:       float
    hitl_triggered:       bool = False


# ─────────────────────────────────────────────────────────────
# Fallback Bengaluru telemetry
# ─────────────────────────────────────────────────────────────
FALLBACK_HOURLY_UV   = [0,0,0,0,0,0,0.5,1.8,3.9,6.2,8.5,10.1,11.3,11.8,10.9,9.2,7.1,4.6,2.3,0.9,0.2,0,0,0]
FALLBACK_HOURLY_PM25 = [38,35,32,30,28,29,35,48,72,88,92,85,78,72,68,75,88,110,98,85,68,55,45,40]
FALLBACK_HOURLY_TEMP = [22,22,21,21,21,22,23,25,27,29,31,33,34,34,33,32,30,28,27,26,25,24,23,23]
FALLBACK_HOURLY_WIND = [8,7,6,6,7,8,10,12,14,15,16,15,14,14,15,16,17,16,14,12,10,9,8,8]
FALLBACK_HOURLY_CLOUD= [10,10,5,5,5,10,15,20,25,30,35,40,45,50,45,40,35,30,25,20,15,10,10,10]
FALLBACK_HOURLY_PRECIP=[5,5,5,5,5,5,5,5,10,15,20,20,25,25,20,15,15,20,15,10,10,5,5,5]

def _aqi_category(pm25: float) -> str:
    if pm25 < 12: return "Good"
    if pm25 < 35: return "Moderate"
    if pm25 < 55: return "Unhealthy for Sensitive"
    if pm25 < 150: return "Unhealthy"
    return "Hazardous"

def _fallback_telemetry(hour: int = None) -> TelemetrySnapshot:
    h = max(0, min(23, hour if hour is not None else datetime.now().hour))
    pm25 = float(FALLBACK_HOURLY_PM25[h])
    return TelemetrySnapshot(
        uv_index=FALLBACK_HOURLY_UV[h], uv_index_clear_sky=FALLBACK_HOURLY_UV[h]*1.1,
        temperature_c=float(FALLBACK_HOURLY_TEMP[h]),
        apparent_temp_c=float(FALLBACK_HOURLY_TEMP[h])+2,
        humidity_pct=65.0, dew_point_c=18.0,
        precipitation_prob=FALLBACK_HOURLY_PRECIP[h], precipitation_mm=0.0,
        wind_speed_kmh=float(FALLBACK_HOURLY_WIND[h]), wind_direction_deg=225,
        cloud_cover_pct=FALLBACK_HOURLY_CLOUD[h], visibility_km=8.0,
        surface_pressure_hpa=916.0,
        weather_code=1, weather_description="Mainly clear",
        pm25=pm25, pm10=pm25*1.6, dust=10.0, ozone=38.0, nitrogen_dioxide=22.0,
        aqi_category=_aqi_category(pm25),
        hourly_uv=FALLBACK_HOURLY_UV,
        hourly_pm25=[float(x) for x in FALLBACK_HOURLY_PM25],
        hourly_precip=FALLBACK_HOURLY_PRECIP,
        hourly_temp=[float(x) for x in FALLBACK_HOURLY_TEMP],
        hourly_wind=[float(x) for x in FALLBACK_HOURLY_WIND],
        hourly_cloud=FALLBACK_HOURLY_CLOUD,
        source="fallback",
    )


# ─────────────────────────────────────────────────────────────
# AGENT 1 — UV Exposure Monitoring Agent (Enhanced)
# ─────────────────────────────────────────────────────────────

class UVMonitoringAgent:
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    AQI_URL  = "https://air-quality-api.open-meteo.com/v1/air-quality"

    async def fetch(self) -> TelemetrySnapshot:
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                w_resp, a_resp = await client.get(self.BASE_URL, params={
                    "latitude": BLR_LAT, "longitude": BLR_LON,
                    "hourly": ",".join([
                        "uv_index","uv_index_clear_sky","temperature_2m","apparent_temperature",
                        "relative_humidity_2m","dew_point_2m","precipitation_probability",
                        "precipitation","wind_speed_10m","wind_direction_10m",
                        "cloud_cover","visibility","surface_pressure","weather_code",
                    ]),
                    "current": ",".join([
                        "uv_index","temperature_2m","apparent_temperature",
                        "relative_humidity_2m","precipitation_probability","precipitation",
                        "wind_speed_10m","wind_direction_10m","cloud_cover",
                        "surface_pressure","weather_code",
                    ]),
                    "timezone": "Asia/Kolkata", "forecast_days": 1,
                }), await client.get(self.AQI_URL, params={
                    "latitude": BLR_LAT, "longitude": BLR_LON,
                    "hourly": "pm2_5,pm10,dust,ozone,nitrogen_dioxide",
                    "current": "pm2_5,pm10,dust,ozone,nitrogen_dioxide",
                    "timezone": "Asia/Kolkata", "forecast_days": 1,
                })
                w_resp.raise_for_status()
                a_resp.raise_for_status()
                w = w_resp.json()
                a = a_resp.json()

            def _c(d, k, default=0): return float(d.get(k, default) or default)
            def _ci(d, k, default=0): return int(d.get(k, default) or default)
            def _h(d, k, fb): return [(float(v) if v is not None else 0) for v in d.get(k, fb)][:24]
            def _hi(d, k, fb): return [(int(v) if v is not None else 0) for v in d.get(k, fb)][:24]

            cw = w.get("current", {})
            ca = a.get("current", {})
            hw = w.get("hourly", {})
            ha = a.get("hourly", {})

            pm25 = _c(ca, "pm2_5", 55)
            wcode = _ci(cw, "weather_code", 0)

            snap = TelemetrySnapshot(
                uv_index            =_c(cw, "uv_index"),
                uv_index_clear_sky  =max(_c(cw, "uv_index")*1.1, _c(cw, "uv_index")),
                temperature_c       =_c(cw, "temperature_2m", 27),
                apparent_temp_c     =_c(cw, "apparent_temperature", 27),
                humidity_pct        =_c(cw, "relative_humidity_2m", 65),
                dew_point_c         =_c(cw, "dew_point_2m", 18),
                precipitation_prob  =_ci(cw, "precipitation_probability", 10),
                precipitation_mm    =_c(cw, "precipitation", 0),
                wind_speed_kmh      =_c(cw, "wind_speed_10m", 10),
                wind_direction_deg  =_ci(cw, "wind_direction_10m", 225),
                cloud_cover_pct     =_ci(cw, "cloud_cover", 20),
                visibility_km       =_c(cw, "visibility", 10000)/1000,
                surface_pressure_hpa=_c(cw, "surface_pressure", 916),
                weather_code        =wcode,
                weather_description =WMO_CODES.get(wcode, "Clear"),
                pm25                =pm25,
                pm10                =_c(ca, "pm10", pm25*1.6),
                dust                =_c(ca, "dust", 10),
                ozone               =_c(ca, "ozone", 38),
                nitrogen_dioxide    =_c(ca, "nitrogen_dioxide", 22),
                aqi_category        =_aqi_category(pm25),
                hourly_uv    =_h(hw, "uv_index",                   FALLBACK_HOURLY_UV),
                hourly_pm25  =_h(ha, "pm2_5",                      FALLBACK_HOURLY_PM25),
                hourly_precip=_hi(hw,"precipitation_probability",   FALLBACK_HOURLY_PRECIP),
                hourly_temp  =_h(hw, "temperature_2m",              FALLBACK_HOURLY_TEMP),
                hourly_wind  =_h(hw, "wind_speed_10m",              FALLBACK_HOURLY_WIND),
                hourly_cloud =_hi(hw,"cloud_cover",                 FALLBACK_HOURLY_CLOUD),
                source="open-meteo",
            )
            log.info("uv_agent.fetch_ok",
                     uv=snap.uv_index, pm25=snap.pm25,
                     temp=snap.temperature_c, wind=snap.wind_speed_kmh,
                     aqi=snap.aqi_category, weather=snap.weather_description)
            return snap

        except Exception as exc:
            log.warning("uv_agent.fetch_failed", error=str(exc), fallback=True)
            return _fallback_telemetry()


# ─────────────────────────────────────────────────────────────
# LLM Gateway
# ─────────────────────────────────────────────────────────────

class LLMGateway:
    _watsonx_token: Optional[str] = None
    _watsonx_token_expiry: float  = 0.0

    def _get_iam_token(self) -> Optional[str]:
        if not IBM_CLOUD_API_KEY:
            return None
        now = time.time()
        if self._watsonx_token and now < self._watsonx_token_expiry - 60:
            return self._watsonx_token
        try:
            resp = httpx.post(
                "https://iam.cloud.ibm.com/identity/token",
                data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": IBM_CLOUD_API_KEY},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self._watsonx_token        = data["access_token"]
            self._watsonx_token_expiry = now + int(data.get("expires_in", 3600))
            return self._watsonx_token
        except Exception as e:
            log.warning("iam_token.failed", error=str(e))
            return None

    def _call_granite(self, prompt: str, max_tokens: int = 600) -> Tuple[str, int, int]:
        """Uses watsonx.ai chat API — required for granite-4-h-small and all instruct/chat models."""
        token = self._get_iam_token()
        if not token or not WATSONX_PROJECT_ID:
            raise RuntimeError("watsonx credentials not configured")
        url  = f"{WATSONX_URL}/ml/v1/text/chat?version=2024-05-31"
        body = {
            "model_id":   WATSONX_MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens":  max_tokens,
            "temperature": 0.7,
        }
        resp = httpx.post(url, json=body,
                          headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                          timeout=35)
        resp.raise_for_status()
        rj       = resp.json()
        text     = rj["choices"][0]["message"]["content"].strip()
        usage    = rj.get("usage", {})
        p_tokens = usage.get("prompt_tokens",     len(prompt.split()) * 2)
        c_tokens = usage.get("completion_tokens", len(text.split())   * 2)
        return text, p_tokens, c_tokens

    def _call_groq(self, prompt: str, max_tokens: int = 600) -> Tuple[str, int, int]:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not configured")
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={"model": GROQ_MODEL_ID,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": 0.7},
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data  = resp.json()
        text  = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def _call_gemini(self, prompt: str, max_tokens: int = 600) -> Tuple[str, int, int]:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not configured")
        url  = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_ID}:generateContent?key={GEMINI_API_KEY}"
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7}}
        resp = httpx.post(url, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        meta = data.get("usageMetadata", {})
        return text, meta.get("promptTokenCount", 0), meta.get("candidatesTokenCount", 0)

    def generate(self, prompt: str, max_tokens: int = 600) -> Tuple[str, str, str, int, int, float]:
        """Returns: (text, provider_key, model_id, p_tokens, c_tokens, cost_usd)"""
        providers = [
            ("granite", WATSONX_MODEL_ID, self._call_granite),
            ("groq",    GROQ_MODEL_ID,    self._call_groq),
            ("gemini",  GEMINI_MODEL_ID,  self._call_gemini),
        ]
        for provider_key, model_id, fn in providers:
            try:
                text, p_tok, c_tok = fn(prompt, max_tokens)
                total_tok = p_tok + c_tok
                cost = (total_tok / 1000) * COST_PER_1K.get(provider_key, 0.0001)
                log.info("llm_gateway.success", provider=provider_key, tokens=total_tok, cost=round(cost,7))
                return text, provider_key, model_id, p_tok, c_tok, cost
            except Exception as e:
                log.warning("llm_gateway.provider_failed", provider=provider_key, error=str(e))
        log.error("llm_gateway.all_failed")
        return (
            "AI analysis temporarily unavailable. Using rule-based recommendations.",
            "fallback", "none", 0, 0, 0.0
        )


_llm = LLMGateway()


# ─────────────────────────────────────────────────────────────
# Risk Math
# ─────────────────────────────────────────────────────────────

SENSITIVITY_MULTIPLIER = {
    "low": 0.85, "moderate": 1.00, "high": 1.30, "hyper_reactive": 1.60,
}

def _calc_protection_discount(rider: RiderProfile) -> float:
    d = 0.0
    if rider.uses_sunscreen:
        d += 0.25 if rider.spf_rating == "SPF50+" else 0.15 if rider.spf_rating == "SPF30" else 0
    if rider.reapplies_sunscreen:    d += 0.05
    if rider.wears_arm_sleeves:      d += 0.15
    if rider.wears_full_jacket:      d += 0.15
    if rider.visor_type == "uv_tinted": d += 0.15
    elif rider.visor_type == "clear":   d += 0.05
    if rider.wears_gloves:           d += 0.05
    if rider.uses_face_mask_gaiter:  d += 0.05
    return min(d, 0.65)

def _calc_net_score(uv: float, duration_h: float, pm25: float,
                     multiplier: float, discount: float) -> float:
    return round(((uv * duration_h) + (pm25 / 45)) * multiplier * (1 - discount), 4)

def _verdict_from_score(score: float) -> str:
    if score < 3.0:  return "SAFE_TO_TRAVEL"
    if score < 7.0:  return "TRAVEL_WITH_GEAR"
    if score < 12.0: return "DELAY_RECOMMENDED"
    return "AVOID_NON_ESSENTIAL"


# ─────────────────────────────────────────────────────────────
# AGENT 2 — Risk & Travel Verdict
# ─────────────────────────────────────────────────────────────

class RiskVerdictAgent:
    name = "Sun Exposure Risk & Travel Verdict Agent"

    def run(self, telemetry: TelemetrySnapshot, rider: RiderProfile,
            duration_h: float, route_info: Dict) -> AgentOutput:
        multiplier = SENSITIVITY_MULTIPLIER.get(rider.skin_sensitivity, 1.0)
        discount   = _calc_protection_discount(rider)
        score      = _calc_net_score(telemetry.uv_index, duration_h, telemetry.pm25, multiplier, discount)
        verdict    = _verdict_from_score(score)
        exposure_pct = route_info.get("open_exposure_pct", 80)

        prompt = f"""You are a skin health advisor for Bengaluru two-wheeler commuters.

CURRENT CONDITIONS (Bengaluru, real-time):
- UV Index: {telemetry.uv_index} (clear-sky max: {telemetry.uv_index_clear_sky:.1f})
- Apparent Temperature: {telemetry.apparent_temp_c}C (actual: {telemetry.temperature_c}C)
- Humidity: {telemetry.humidity_pct}% | Dew Point: {telemetry.dew_point_c}C
- Wind: {telemetry.wind_speed_kmh} km/h | Cloud Cover: {telemetry.cloud_cover_pct}%
- Visibility: {telemetry.visibility_km:.1f} km | Weather: {telemetry.weather_description}
- PM2.5: {telemetry.pm25} ug/m3 | PM10: {telemetry.pm10:.1f} | Ozone: {telemetry.ozone:.1f} ppb
- AQI Category: {telemetry.aqi_category}

RIDER: {rider.skin_type} skin, {rider.skin_sensitivity} sensitivity
ROUTE: {duration_h:.2f}h ride, {exposure_pct}% open-air exposure, traffic multiplier {route_info.get("traffic_multiplier",1.3)}x
NET EXPOSURE SCORE: {score:.2f} -> VERDICT: {verdict}

In 3-4 sentences explain the verdict, mentioning wind chill effect, smog conditions, and what the rider should watch out for TODAY specifically."""

        text, provider, model, p_tok, c_tok, cost = _llm.generate(prompt, max_tokens=350)
        return AgentOutput(
            agent=self.name, verdict=verdict, score=score,
            llm_provider=provider, llm_model=model,
            tokens_used=p_tok+c_tok, cost_usd=cost, raw_response=text,
        )


# ─────────────────────────────────────────────────────────────
# AGENT 3 — Travel Timing Advisory
# ─────────────────────────────────────────────────────────────

class TimingAdvisoryAgent:
    name = "Travel Timing Advisory Agent"

    def run(self, telemetry: TelemetrySnapshot, rider: RiderProfile,
            duration_h: float, route: str, route_info: Dict) -> AgentOutput:
        multiplier = SENSITIVITY_MULTIPLIER.get(rider.skin_sensitivity, 1.0)
        discount   = _calc_protection_discount(rider)
        peak_hours = route_info.get("peak_hours", [])

        windows = []
        for h in range(5, 21):
            if telemetry.hourly_precip[h] >= 65:
                continue
            uv_h    = telemetry.hourly_uv[h]
            pm25_h  = telemetry.hourly_pm25[h]
            wind_h  = telemetry.hourly_wind[h] if telemetry.hourly_wind else 10.0
            cloud_h = telemetry.hourly_cloud[h] if telemetry.hourly_cloud else 20
            score_h = _calc_net_score(uv_h, duration_h, pm25_h, multiplier, discount)
            # Peak hour penalty
            is_peak = any(
                int(ph.split("-")[0].split(":")[0]) <= h <= int(ph.split("-")[1].split(":")[0])
                for ph in peak_hours if "-" in ph
            )
            windows.append({
                "hour": h, "label": f"{h:02d}:00-{h+1:02d}:00",
                "uv": round(uv_h, 2), "pm25": round(pm25_h, 2),
                "wind_kmh": round(wind_h, 1), "cloud_pct": cloud_h,
                "score": round(score_h, 4), "is_peak_traffic": is_peak,
                "rain_prob": telemetry.hourly_precip[h],
            })

        windows_sorted = sorted(windows, key=lambda x: (x["score"], x["is_peak_traffic"]))
        top3 = windows_sorted[:3]

        current_score = _calc_net_score(
            telemetry.uv_index, duration_h, telemetry.pm25, multiplier, discount
        )
        for w in top3:
            w["reduction_pct"] = round((1 - w["score"] / current_score) * 100, 1) if current_score > 0 else 0.0

        prompt = f"""You are a Bengaluru commute timing specialist.

Route: {route} | Duration: {duration_h}h
Peak traffic hours: {', '.join(peak_hours)}
Current score: {current_score:.2f} ({_verdict_from_score(current_score)})

Top 3 best departure windows (sorted by lowest exposure):
{json.dumps(top3, indent=2)}

In 3 sentences: recommend the single best window and why, mentioning traffic conditions, UV, wind, and PM2.5. Include the exact percentage improvement."""

        text, provider, model, p_tok, c_tok, cost = _llm.generate(prompt, max_tokens=280)
        return AgentOutput(
            agent=self.name, best_windows=top3,
            llm_provider=provider, llm_model=model,
            tokens_used=p_tok+c_tok, cost_usd=cost, raw_response=text,
        )


# ─────────────────────────────────────────────────────────────
# AGENT 4 — Skin Care Advisory
# ─────────────────────────────────────────────────────────────

class SkinCareAdvisoryAgent:
    name = "Skin Care Advisory Agent"

    def run(self, telemetry: TelemetrySnapshot, rider: RiderProfile, verdict: str) -> AgentOutput:
        prompt = f"""You are a Bengaluru-specialist dermatology advisor for two-wheeler commuters.

RIDER: {rider.skin_type} skin, {rider.skin_sensitivity} sensitivity | Verdict: {verdict}
CONDITIONS: UV {telemetry.uv_index}, Apparent Temp {telemetry.apparent_temp_c}C, 
Humidity {telemetry.humidity_pct}%, Wind {telemetry.wind_speed_kmh} km/h, 
PM2.5 {telemetry.pm25} ug/m3 (AQI: {telemetry.aqi_category}), Weather: {telemetry.weather_description}

Provide exactly:
PRE-RIDE PREP (4 steps):
• ...
• ...
• ...
• ...

POST-RIDE RECOVERY (4 steps):
• ...
• ...
• ...
• ...

SPECIAL WARNING for today's conditions (1 sentence):

Be specific to Bengaluru's combination of UV, smog, heat, and helmet friction."""

        text, provider, model, p_tok, c_tok, cost = _llm.generate(prompt, max_tokens=450)
        return AgentOutput(
            agent=self.name, advice=text,
            llm_provider=provider, llm_model=model,
            tokens_used=p_tok+c_tok, cost_usd=cost, raw_response=text,
        )


# ─────────────────────────────────────────────────────────────
# AGENT 5 — Protective Measures
# ─────────────────────────────────────────────────────────────

class ProtectiveMeasuresAgent:
    name = "Protective Measures Recommendation Agent"

    _GAPS_MAP = {
        "No SPF sunscreen":                      lambda r: not r.uses_sunscreen,
        "Using SPF30 instead of SPF50+":         lambda r: r.uses_sunscreen and r.spf_rating == "SPF30",
        "No sunscreen reapplication every 2-3h": lambda r: r.uses_sunscreen and not r.reapplies_sunscreen,
        "No UPF arm sleeves or full jacket":     lambda r: not r.wears_arm_sleeves and not r.wears_full_jacket,
        "Clear visor (0% UV protection)":        lambda r: r.visor_type == "clear",
        "No riding gloves":                      lambda r: not r.wears_gloves,
        "No face mask/neck gaiter":              lambda r: not r.uses_face_mask_gaiter,
    }

    def run(self, rider: RiderProfile, telemetry: TelemetrySnapshot,
            verdict: str, route_info: Dict) -> AgentOutput:
        gaps = [msg for msg, check in self._GAPS_MAP.items() if check(rider)]
        wind_note = "High wind today increases particle exposure." if telemetry.wind_speed_kmh > 20 else ""

        prompt = f"""You are a protective gear advisor for Bengaluru two-wheeler riders.

PROTECTION GAPS:
{chr(10).join(f'- {g}' for g in gaps) if gaps else '- No major gaps detected'}

CURRENT RISK: Verdict={verdict}, UV={telemetry.uv_index}, PM2.5={telemetry.pm25} ({telemetry.aqi_category})
Wind={telemetry.wind_speed_kmh} km/h. {wind_note}
Route open-air exposure: {route_info.get("open_exposure_pct",80)}%

For each gap: one specific actionable fix with an affordable India-available product/option.
End with: PROTECTION RATING: X/10 and ONE highest-priority action today."""

        text, provider, model, p_tok, c_tok, cost = _llm.generate(prompt, max_tokens=420)
        return AgentOutput(
            agent=self.name, gaps=gaps, advice=text,
            llm_provider=provider, llm_model=model,
            tokens_used=p_tok+c_tok, cost_usd=cost, raw_response=text,
        )


# ─────────────────────────────────────────────────────────────
# AGENT 7 — Pre-Ride Safety Gear Reminder Agent (NEW)
# ─────────────────────────────────────────────────────────────

class GearReminderAgent:
    name = "Pre-Ride Safety Gear Reminder Agent"

    # Priority gear checklist
    _GEAR_CHECKLIST = [
        ("Helmet",                "critical",   "helmet",      "Legal requirement. Protects from UV and pollution ingress."),
        ("SPF50+ Sunscreen",      "critical",   "sunscreen",   "Apply 20 min before ride. UV protection is non-negotiable."),
        ("UPF Arm Sleeves",       "high",       "sleeves",     "Covers forearms — highest UV exposure area on a bike."),
        ("UV-Tinted Visor",       "high",       "visor",       "Blocks 99% UV to eyes and face vs clear visor's 5%."),
        ("Riding Gloves",         "high",       "gloves",      "Protects hands from UV and PM2.5 particle deposition."),
        ("Face Mask/Neck Gaiter", "high",       "mask",        "Filters PM2.5 — especially critical with AQI Unhealthy."),
        ("Full Riding Jacket",    "recommended","jacket",      "Best protection but can be hot above 32C — your call."),
        ("Sunglasses/Goggles",    "recommended","eyewear",     "Protects eyes from UV, dust, and wind tear formation."),
        ("Hydration (500ml water)","recommended","hydration",  "Riding dehydrates. UV + heat load is highest midday."),
    ]

    _GEAR_EMOJIS = {
        "helmet": "🪖", "sunscreen": "🧴", "sleeves": "👕",
        "visor": "😎", "gloves": "🧤", "mask": "😷",
        "jacket": "🧥", "eyewear": "🕶️", "hydration": "💧",
    }

    def run(self, rider: RiderProfile, telemetry: TelemetrySnapshot,
            verdict: str, departure_time: Optional[str] = None) -> AgentOutput:
        reminders: List[GearReminder] = []

        for gear, priority, key, base_reason in self._GEAR_CHECKLIST:
            # Adjust priority based on live conditions
            if key == "sunscreen" and telemetry.uv_index >= 8:
                priority = "critical"
            if key == "mask" and telemetry.pm25 > 55:
                priority = "critical"
            if key == "jacket" and telemetry.temperature_c > 33:
                priority = "recommended"   # hot — downgrade to recommended
            if key == "visor" and telemetry.uv_index >= 6:
                priority = "critical"

            # Build contextual reason
            if key == "sunscreen":
                reason = f"UV Index is {telemetry.uv_index} ({telemetry.weather_description}). {base_reason}"
            elif key == "mask":
                reason = f"PM2.5 is {telemetry.pm25:.0f} ug/m3 — {telemetry.aqi_category}. {base_reason}"
            elif key == "visor":
                reason = f"UV {telemetry.uv_index} with {telemetry.cloud_cover_pct}% cloud cover. {base_reason}"
            else:
                reason = base_reason

            reminders.append(GearReminder(
                gear_item=gear, priority=priority,
                reason=reason, emoji=self._GEAR_EMOJIS.get(key, "✅")
            ))

        # Sort: critical first, then high, then recommended
        order = {"critical": 0, "high": 1, "recommended": 2}
        reminders.sort(key=lambda r: order.get(r.priority, 3))

        # LLM-generated personalized reminder message
        critical_items = [r.gear_item for r in reminders if r.priority == "critical"]
        missing_items  = []
        if not rider.uses_sunscreen:               missing_items.append("sunscreen")
        if not rider.wears_gloves:                 missing_items.append("gloves")
        if not rider.uses_face_mask_gaiter:        missing_items.append("face mask")
        if rider.visor_type == "clear":            missing_items.append("UV-tinted visor")
        if not rider.wears_arm_sleeves and not rider.wears_full_jacket: missing_items.append("arm sleeves")

        dep_str = departure_time or "your upcoming ride"
        prompt = f"""You are a friendly safety reminder bot for a Bengaluru bike commuter.

The rider is about to depart for {dep_str}.
Current conditions: UV {telemetry.uv_index}, PM2.5 {telemetry.pm25:.0f} ({telemetry.aqi_category}), Temp {telemetry.temperature_c}C, {telemetry.weather_description}.
Travel verdict: {verdict}
Critical gear items for today: {', '.join(critical_items)}
Items the rider is NOT currently using: {', '.join(missing_items) if missing_items else 'None — well equipped!'}

Write a friendly, motivating pre-ride safety reminder in 2-3 sentences. 
Address the rider directly. Mention the top 2 most important items to put on RIGHT NOW.
Keep it warm, not scolding."""

        text, provider, model, p_tok, c_tok, cost = _llm.generate(prompt, max_tokens=200)
        return AgentOutput(
            agent=self.name, advice=text, gear_reminders=reminders,
            llm_provider=provider, llm_model=model,
            tokens_used=p_tok+c_tok, cost_usd=cost, raw_response=text,
        )


# ─────────────────────────────────────────────────────────────
# AGENT 6 — Analytics Dashboard Agent
# ─────────────────────────────────────────────────────────────

class AnalyticsDashboardAgent:
    name = "Skin Health Analytics Dashboard Agent"

    def __init__(self):
        self._db_conn = None

    def _get_db(self):
        if self._db_conn is not None:
            try:
                self._db_conn.ping(reconnect=True, attempts=1, delay=0)
                return self._db_conn
            except Exception:
                self._db_conn = None
        try:
            import mysql.connector
            self._db_conn = mysql.connector.connect(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database=os.getenv("MYSQL_DB", "bengaluru_rider_advisor"),
                connection_timeout=5,
            )
            return self._db_conn
        except Exception as e:
            log.warning("analytics_agent.db_unavailable", error=str(e))
            return None

    def log_exposure(self, req: TripRequest, telemetry: TelemetrySnapshot,
                     score: float, verdict: str, windows: List[Dict],
                     skincare: str, gaps: List[str], ai_recs: Dict,
                     llm_provider: str, llm_model: str,
                     p_tok: int, c_tok: int, cost: float) -> bool:
        db = self._get_db()
        if not db:
            return False
        try:
            cur = db.cursor()
            cur.execute("""
                INSERT INTO exposure_logs
                  (rider_id,route,departure_time,duration_hours,
                   uv_index,temperature_c,humidity_pct,precipitation_prob,pm25,dust,
                   net_exposure_score,sensitivity_multiplier,protection_discount,
                   travel_verdict,best_travel_window,protection_gap_advice,
                   skincare_advice,ai_recommendations,llm_provider_used,llm_model_used,
                   prompt_tokens,completion_tokens,total_cost_usd)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                req.rider_id, req.route,
                req.departure_time or datetime.utcnow().isoformat(),
                req.duration_hours,
                telemetry.uv_index, telemetry.temperature_c, telemetry.humidity_pct,
                telemetry.precipitation_prob, telemetry.pm25, telemetry.dust,
                score,
                SENSITIVITY_MULTIPLIER.get(req.rider.skin_sensitivity, 1.0),
                _calc_protection_discount(req.rider),
                verdict, json.dumps(windows[:3]),
                "\n".join(gaps), skincare,
                json.dumps(ai_recs),
                llm_provider, llm_model, p_tok, c_tok, cost,
            ))
            db.commit()
            return True
        except Exception as e:
            log.error("analytics_agent.log_failed", error=str(e))
            return False

    def get_weekly_analytics(self) -> Dict:
        db = self._get_db()
        if not db:
            return {"error": "Database unavailable"}
        try:
            cur = db.cursor(dictionary=True)
            cur.execute("""
                SELECT DATE(logged_at) as date, COUNT(*) as trips,
                  AVG(net_exposure_score) as avg_score,
                  AVG(uv_index) as avg_uv, AVG(pm25) as avg_pm25,
                  SUM(CASE WHEN travel_verdict='SAFE_TO_TRAVEL' THEN 1 ELSE 0 END) as safe,
                  SUM(CASE WHEN travel_verdict='TRAVEL_WITH_GEAR' THEN 1 ELSE 0 END) as gear,
                  SUM(CASE WHEN travel_verdict='DELAY_RECOMMENDED' THEN 1 ELSE 0 END) as delay,
                  SUM(CASE WHEN travel_verdict='AVOID_NON_ESSENTIAL' THEN 1 ELSE 0 END) as avoid,
                  SUM(total_cost_usd) as total_cost_usd
                FROM exposure_logs
                WHERE logged_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY DATE(logged_at) ORDER BY date DESC
            """)
            rows = cur.fetchall()
            cur.execute("""
                SELECT SUM(prompt_tokens+completion_tokens) as total_tokens,
                       SUM(total_cost_usd) as total_cost, COUNT(*) as total_logs
                FROM exposure_logs WHERE logged_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            """)
            summary = cur.fetchone()
            return {"daily": rows, "summary": summary}
        except Exception as e:
            return {"error": str(e)}

    def chat(self, agent_name: str, message: str, context: Dict,
             session_id: str, rider_id: Optional[int] = None) -> AgentOutput:
        blocked = _apply_guardrails(message)
        if blocked:
            return AgentOutput(agent=f"Chat:{agent_name}", advice=blocked,
                               llm_provider="guardrail", llm_model="none")

        agent_descriptions = {
            "uv_monitor":   "UV Exposure Monitoring Agent — real-time UV/PM2.5/weather data specialist",
            "risk_verdict": "Risk & Verdict Agent — calculates net exposure scores and verdicts",
            "timing":       "Timing Advisory Agent — optimal departure window analysis",
            "skincare":     "Skin Care Agent — pre/post ride dermatology advice",
            "protection":   "Protective Gear Agent — gap analysis and gear recommendations",
            "reminder":     "Pre-Ride Reminder Agent — safety checklist and gear reminders",
            "analytics":    "Analytics Agent — weekly trends and health pattern insights",
        }
        agent_desc = agent_descriptions.get(agent_name, "Bengaluru Rider Advisor")

        prompt = f"""You are the {agent_desc} for Bengaluru two-wheeler commuters.

Session context: {json.dumps(context, default=str)[:800]}

Rider question: {message}

Answer helpfully in 2-4 sentences, specific to Bengaluru riding conditions and current weather."""

        text, provider, model, p_tok, c_tok, cost = _llm.generate(prompt, max_tokens=350)

        db = self._get_db()
        if db:
            try:
                cur = db.cursor()
                cur.executemany(
                    "INSERT INTO agent_chat_history (rider_id,session_id,agent_name,role,message,tokens_used,cost_usd) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (rider_id, session_id, agent_name, "user",      message, 0,              0.0),
                        (rider_id, session_id, agent_name, "assistant", text,    p_tok+c_tok,    cost),
                    ]
                )
                db.commit()
            except Exception as e:
                log.warning("chat_log_failed", error=str(e))

        return AgentOutput(
            agent=f"Chat:{agent_name}", advice=text,
            llm_provider=provider, llm_model=model,
            tokens_used=p_tok+c_tok, cost_usd=cost, raw_response=text,
        )


# ─────────────────────────────────────────────────────────────
# Human-in-the-Loop
# ─────────────────────────────────────────────────────────────

def _trigger_hitl(score: float, verdict: str, rider: RiderProfile,
                  telemetry: TelemetrySnapshot) -> bool:
    if not HITL_WEBHOOK_URL or score < HITL_RISK_THRESHOLD:
        return False
    payload = {
        "alert": "HIGH_RISK_RIDE", "score": score, "verdict": verdict,
        "rider_sensitivity": rider.skin_sensitivity,
        "uv_index": telemetry.uv_index, "pm25": telemetry.pm25,
        "aqi": telemetry.aqi_category, "weather": telemetry.weather_description,
        "timestamp": datetime.utcnow().isoformat(),
    }
    try:
        httpx.post(HITL_WEBHOOK_URL, json=payload, timeout=5)
        log.info("hitl_triggered", score=score)
        return True
    except Exception as e:
        log.warning("hitl_webhook_failed", error=str(e))
        return False


# ─────────────────────────────────────────────────────────────
# Guardrails
# ─────────────────────────────────────────────────────────────

SAFE_RESPONSE = "I can only provide information related to UV exposure, skin health, and riding safety for Bengaluru commuters."

def _apply_guardrails(user_input: str) -> Optional[str]:
    blocked = ["hack","exploit","malware","jailbreak","ignore previous",
               "ignore all instructions","prompt injection","bypass"]
    lower = user_input.lower()
    for t in blocked:
        if t in lower:
            return SAFE_RESPONSE
    if len(user_input) > 2000:
        return "Please keep your question under 2000 characters."
    return None


# ─────────────────────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────

class AdvisoryPipeline:
    def __init__(self):
        self.uv_agent         = UVMonitoringAgent()
        self.risk_agent       = RiskVerdictAgent()
        self.timing_agent     = TimingAdvisoryAgent()
        self.skincare_agent   = SkinCareAdvisoryAgent()
        self.protection_agent = ProtectiveMeasuresAgent()
        self.reminder_agent   = GearReminderAgent()
        self.analytics_agent  = AnalyticsDashboardAgent()

    async def run(self, req: TripRequest) -> FullAdvisoryResponse:
        start = time.time()

        # Determine route — custom or preset
        route_name = req.route
        if req.custom_start and req.custom_end:
            route_name = f"{req.custom_start} – {req.custom_end}"
        route_info = get_route_info(route_name)

        # Adjust duration from route_info if not explicitly provided differently
        duration_h = req.duration_hours if req.duration_hours != 1.0 else route_info.get("typical_duration_h", 1.0)

        # Agent 1 — Telemetry
        telemetry = await self.uv_agent.fetch()

        # Agent 2 — Risk & Verdict
        risk_out = self.risk_agent.run(telemetry, req.rider, duration_h, route_info)

        # Agent 3 — Timing
        timing_out = self.timing_agent.run(telemetry, req.rider, duration_h, route_name, route_info)

        # Agent 4 — Skincare
        skincare_out = self.skincare_agent.run(telemetry, req.rider, risk_out.verdict)

        # Agent 5 — Protective Measures
        protection_out = self.protection_agent.run(req.rider, telemetry, risk_out.verdict, route_info)

        # Agent 7 — Gear Reminders
        reminder_out = self.reminder_agent.run(
            req.rider, telemetry, risk_out.verdict, req.departure_time
        )

        agents = [risk_out, timing_out, skincare_out, protection_out, reminder_out]
        total_tokens = sum(a.tokens_used for a in agents)
        total_cost   = sum(a.cost_usd   for a in agents)

        hitl = _trigger_hitl(risk_out.score, risk_out.verdict, req.rider, telemetry)

        ai_recs = {
            "risk":       risk_out.raw_response,
            "timing":     timing_out.raw_response,
            "skincare":   skincare_out.raw_response,
            "protection": protection_out.raw_response,
            "reminder":   reminder_out.raw_response,
        }
        self.analytics_agent.log_exposure(
            req, telemetry, risk_out.score, risk_out.verdict,
            timing_out.best_windows or [], skincare_out.advice or "",
            protection_out.gaps or [], ai_recs,
            risk_out.llm_provider, risk_out.llm_model,
            total_tokens // 2, total_tokens // 2, total_cost,
        )

        log.info("pipeline.complete",
                 elapsed_s=round(time.time()-start, 2),
                 verdict=risk_out.verdict, score=risk_out.score,
                 provider=risk_out.llm_provider, route=route_name)

        return FullAdvisoryResponse(
            session_id=req.session_id,
            timestamp=datetime.utcnow().isoformat(),
            telemetry=telemetry,
            route_info={**route_info, "name": route_name},
            risk_score=risk_out.score,
            travel_verdict=risk_out.verdict,
            verdict_explanation=risk_out.raw_response,
            best_travel_windows=timing_out.best_windows or [],
            skincare_advice=skincare_out.advice or "",
            protection_gaps=protection_out.gaps or [],
            gear_recommendations=protection_out.advice or "",
            gear_reminders=reminder_out.gear_reminders or [],
            reminder_message=reminder_out.raw_response,
            agents=agents,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            hitl_triggered=hitl,
        )
