"""
agents_engine.py
================
Multi-Agent AI Engine for Bengaluru Two-Wheeler Rider
Skin Health & Sun Exposure Advisor.

Agents:
  1. UV Exposure Monitoring Agent
  2. Sun Exposure Risk & Travel Verdict Agent  (IBM Granite)
  3. Travel Timing Advisory Agent              (IBM Granite / Groq fallback)
  4. Skin Care Advisory Agent                  (IBM Granite)
  5. Protective Measures Recommendation Agent  (IBM Granite)
  6. Skin Health Analytics Dashboard Agent     (MySQL + IBM Granite)

LLM Providers (priority order):
  primary   → IBM watsonx.ai  (ibm/granite-3-3-8b-instruct)
  fallback1 → Groq            (llama-3.3-70b-versatile)
  fallback2 → Google Gemini   (gemini-2.5-flash)
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
IBM_CLOUD_API_KEY    = os.getenv("IBM_CLOUD_API_KEY", "")
WATSONX_URL          = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID   = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_MODEL_ID     = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-3-8b-instruct")

GROQ_API_KEY         = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_ID        = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_ID      = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

HITL_WEBHOOK_URL     = os.getenv("HITL_WEBHOOK_URL", "")
HITL_RISK_THRESHOLD  = float(os.getenv("HITL_RISK_THRESHOLD", "15.0"))

# Cost per 1K tokens (USD)
COST_PER_1K = {
    "granite": float(os.getenv("GRANITE_COST_PER_1K_TOKENS", "0.00008")),
    "groq":    float(os.getenv("GROQ_COST_PER_1K_TOKENS",    "0.00006")),
    "gemini":  float(os.getenv("GEMINI_COST_PER_1K_TOKENS",  "0.00010")),
}

# Bengaluru coordinates
BLR_LAT = 12.9716
BLR_LON = 77.5946

# ─────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────

class RiderProfile(BaseModel):
    name:             str           = "Rider"
    skin_type:        str           = "combination"   # oily|dry|combination|sensitive|normal
    skin_sensitivity: str           = "moderate"       # low|moderate|high|hyper_reactive
    # Protection habits
    uses_sunscreen:         bool    = False
    spf_rating:             str     = "none"            # none|SPF30|SPF50+
    reapplies_sunscreen:    bool    = False
    wears_arm_sleeves:      bool    = False
    wears_full_jacket:      bool    = False
    visor_type:             str     = "clear"           # clear|uv_tinted
    wears_gloves:           bool    = False
    uses_face_mask_gaiter:  bool    = False


class TripRequest(BaseModel):
    rider:          RiderProfile
    route:          str           = "Electronic City – Indiranagar"
    duration_hours: float         = 1.0
    departure_time: Optional[str] = None   # ISO-8601 or "now"
    rider_id:       Optional[int] = None
    session_id:     str           = Field(default_factory=lambda: str(uuid.uuid4()))


class TelemetrySnapshot(BaseModel):
    uv_index:              float
    temperature_c:         float
    humidity_pct:          float
    precipitation_prob:    int
    pm25:                  float
    dust:                  float
    hourly_uv:             List[float]
    hourly_pm25:           List[float]
    hourly_precip:         List[int]
    source:                str = "open-meteo"


class AgentOutput(BaseModel):
    agent:         str
    verdict:       Optional[str] = None
    score:         Optional[float] = None
    best_windows:  Optional[List[Dict]] = None
    advice:        Optional[str] = None
    gaps:          Optional[List[str]] = None
    llm_provider:  str = "none"
    llm_model:     str = "none"
    tokens_used:   int = 0
    cost_usd:      float = 0.0
    raw_response:  str = ""


class FullAdvisoryResponse(BaseModel):
    session_id:          str
    timestamp:           str
    telemetry:           TelemetrySnapshot
    risk_score:          float
    travel_verdict:      str
    verdict_explanation: str
    best_travel_windows: List[Dict]
    skincare_advice:     str
    protection_gaps:     List[str]
    gear_recommendations: str
    agents:              List[AgentOutput]
    total_tokens:        int
    total_cost_usd:      float
    hitl_triggered:      bool = False


# ─────────────────────────────────────────────────────────────
# Fallback Bengaluru telemetry dataset (offline / API down)
# ─────────────────────────────────────────────────────────────
FALLBACK_HOURLY_UV = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # 00–05
    0.5, 1.8, 3.9, 6.2, 8.5, 10.1,  # 06–11
    11.3, 11.8, 10.9, 9.2, 7.1, 4.6, # 12–17
    2.3, 0.9, 0.2, 0.0, 0.0, 0.0,   # 18–23
]
FALLBACK_HOURLY_PM25 = [
    38, 35, 32, 30, 28, 29,          # 00–05 (early morning cleaner)
    35, 48, 72, 88, 92, 85,          # 06–11 (morning rush)
    78, 72, 68, 75, 88, 110,         # 12–17 (afternoon & evening rush)
    98, 85, 68, 55, 45, 40,          # 18–23
]
FALLBACK_HOURLY_PRECIP = [5]*24


def _fallback_telemetry(hour: int = None) -> TelemetrySnapshot:
    h = hour if hour is not None else datetime.now().hour
    h = max(0, min(23, h))
    return TelemetrySnapshot(
        uv_index=FALLBACK_HOURLY_UV[h],
        temperature_c=27.5,
        humidity_pct=65.0,
        precipitation_prob=FALLBACK_HOURLY_PRECIP[h],
        pm25=float(FALLBACK_HOURLY_PM25[h]),
        dust=10.0,
        hourly_uv=FALLBACK_HOURLY_UV,
        hourly_pm25=[float(x) for x in FALLBACK_HOURLY_PM25],
        hourly_precip=FALLBACK_HOURLY_PRECIP,
        source="fallback",
    )


# ─────────────────────────────────────────────────────────────
# AGENT 1 — UV Exposure Monitoring Agent
# ─────────────────────────────────────────────────────────────

class UVMonitoringAgent:
    """Fetches real-time + 24-hour hourly telemetry from Open-Meteo."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    AQI_URL  = "https://air-quality-api.open-meteo.com/v1/air-quality"

    async def fetch(self) -> TelemetrySnapshot:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                weather_task = client.get(self.BASE_URL, params={
                    "latitude":  BLR_LAT,
                    "longitude": BLR_LON,
                    "hourly":    "uv_index,temperature_2m,relative_humidity_2m,precipitation_probability",
                    "current":   "uv_index,temperature_2m,relative_humidity_2m,precipitation_probability",
                    "timezone":  "Asia/Kolkata",
                    "forecast_days": 1,
                })
                aqi_task = client.get(self.AQI_URL, params={
                    "latitude":  BLR_LAT,
                    "longitude": BLR_LON,
                    "hourly":    "pm2_5,dust",
                    "current":   "pm2_5,dust",
                    "timezone":  "Asia/Kolkata",
                    "forecast_days": 1,
                })
                w_resp, a_resp = await weather_task, await aqi_task
                w_resp.raise_for_status()
                a_resp.raise_for_status()
                w = w_resp.json()
                a = a_resp.json()

            current_uv    = float(w.get("current", {}).get("uv_index", 0) or 0)
            current_temp  = float(w.get("current", {}).get("temperature_2m", 27) or 27)
            current_hum   = float(w.get("current", {}).get("relative_humidity_2m", 65) or 65)
            current_prec  = int(w.get("current", {}).get("precipitation_probability", 10) or 10)
            current_pm25  = float(a.get("current", {}).get("pm2_5", 55) or 55)
            current_dust  = float(a.get("current", {}).get("dust", 8) or 8)

            hourly_uv     = [float(v or 0) for v in w.get("hourly", {}).get("uv_index", FALLBACK_HOURLY_UV)]
            hourly_pm25   = [float(v or 0) for v in a.get("hourly", {}).get("pm2_5", FALLBACK_HOURLY_PM25)]
            hourly_prec   = [int(v or 0)   for v in w.get("hourly", {}).get("precipitation_probability", FALLBACK_HOURLY_PRECIP)]

            # Pad / trim to 24 elements
            hourly_uv   = (hourly_uv   + FALLBACK_HOURLY_UV)[:24]
            hourly_pm25 = (hourly_pm25 + [float(x) for x in FALLBACK_HOURLY_PM25])[:24]
            hourly_prec = (hourly_prec + FALLBACK_HOURLY_PRECIP)[:24]

            log.info("uv_agent.fetch_ok", uv=current_uv, pm25=current_pm25)
            return TelemetrySnapshot(
                uv_index=current_uv, temperature_c=current_temp,
                humidity_pct=current_hum, precipitation_prob=current_prec,
                pm25=current_pm25, dust=current_dust,
                hourly_uv=hourly_uv, hourly_pm25=hourly_pm25,
                hourly_precip=hourly_prec,
                source="open-meteo",
            )
        except Exception as exc:
            log.warning("uv_agent.fetch_failed", error=str(exc), fallback=True)
            return _fallback_telemetry()


# ─────────────────────────────────────────────────────────────
# LLM Gateway — multi-provider with token cost tracking
# ─────────────────────────────────────────────────────────────

class LLMGateway:
    """Routes inference requests across IBM Granite → Groq → Gemini."""

    # ---- IBM watsonx.ai -------------------------------------------------
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

    def _call_granite(self, prompt: str, max_tokens: int = 512) -> Tuple[str, int, int]:
        token = self._get_iam_token()
        if not token or not WATSONX_PROJECT_ID:
            raise RuntimeError("watsonx credentials not configured")
        url  = f"{WATSONX_URL}/ml/v1/text/generation?version=2024-05-31"
        body = {
            "model_id":  WATSONX_MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "input":     prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens":  max_tokens,
                "temperature":     0.7,
                "repetition_penalty": 1.1,
            },
        }
        resp = httpx.post(url, json=body,
                          headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                          timeout=30)
        resp.raise_for_status()
        data     = resp.json()
        result   = data["results"][0]
        text     = result.get("generated_text", "").strip()
        p_tokens = result.get("input_token_count", len(prompt.split()) * 2)
        c_tokens = result.get("generated_token_count", len(text.split()) * 2)
        return text, p_tokens, c_tokens

    def _call_groq(self, prompt: str, max_tokens: int = 512) -> Tuple[str, int, int]:
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
        data     = resp.json()
        text     = data["choices"][0]["message"]["content"].strip()
        usage    = data.get("usage", {})
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def _call_gemini(self, prompt: str, max_tokens: int = 512) -> Tuple[str, int, int]:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not configured")
        url  = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_ID}:generateContent?key={GEMINI_API_KEY}"
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7}}
        resp = httpx.post(url, json=body, timeout=30)
        resp.raise_for_status()
        data  = resp.json()
        text  = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        meta  = data.get("usageMetadata", {})
        return text, meta.get("promptTokenCount", 0), meta.get("candidatesTokenCount", 0)

    def generate(self, prompt: str, max_tokens: int = 512) -> Tuple[str, str, str, int, int, float]:
        """Returns: (text, provider_key, model_id, prompt_tokens, completion_tokens, cost_usd)"""
        providers = [
            ("granite", WATSONX_MODEL_ID,  self._call_granite),
            ("groq",    GROQ_MODEL_ID,     self._call_groq),
            ("gemini",  GEMINI_MODEL_ID,   self._call_gemini),
        ]
        last_err = None
        for provider_key, model_id, fn in providers:
            try:
                text, p_tok, c_tok = fn(prompt, max_tokens)
                total_tok = p_tok + c_tok
                cost = (total_tok / 1000) * COST_PER_1K.get(provider_key, 0.0001)
                log.info("llm_gateway.success", provider=provider_key, tokens=total_tok, cost=cost)
                return text, provider_key, model_id, p_tok, c_tok, cost
            except Exception as e:
                log.warning("llm_gateway.provider_failed", provider=provider_key, error=str(e))
                last_err = e
        # All providers failed — return deterministic fallback
        log.error("llm_gateway.all_failed", error=str(last_err))
        return (
            "⚠️ AI analysis temporarily unavailable. Using rule-based recommendations.",
            "fallback", "none", 0, 0, 0.0
        )


_llm = LLMGateway()


# ─────────────────────────────────────────────────────────────
# AGENT 2 — Sun Exposure Risk & Travel Verdict Agent
# ─────────────────────────────────────────────────────────────

SENSITIVITY_MULTIPLIER = {
    "low": 0.85, "moderate": 1.00, "high": 1.30, "hyper_reactive": 1.60,
}

def _calc_protection_discount(rider: RiderProfile) -> float:
    discount = 0.0
    if rider.uses_sunscreen:
        if rider.spf_rating == "SPF50+":
            discount += 0.25
        elif rider.spf_rating == "SPF30":
            discount += 0.15
    if rider.reapplies_sunscreen:
        discount += 0.05
    if rider.wears_arm_sleeves:
        discount += 0.15
    if rider.wears_full_jacket:
        discount += 0.15
    if rider.visor_type == "uv_tinted":
        discount += 0.15
    elif rider.visor_type == "clear":
        discount += 0.05
    if rider.wears_gloves:
        discount += 0.05
    if rider.uses_face_mask_gaiter:
        discount += 0.05
    return min(discount, 0.65)


def _calc_net_score(uv: float, duration_h: float, pm25: float,
                     multiplier: float, discount: float) -> float:
    return round(((uv * duration_h) + (pm25 / 45)) * multiplier * (1 - discount), 4)


def _verdict_from_score(score: float) -> str:
    if score < 3.0:
        return "SAFE_TO_TRAVEL"
    elif score < 7.0:
        return "TRAVEL_WITH_GEAR"
    elif score < 12.0:
        return "DELAY_RECOMMENDED"
    else:
        return "AVOID_NON_ESSENTIAL"


class RiskVerdictAgent:
    name = "Sun Exposure Risk & Travel Verdict Agent"

    def run(self, telemetry: TelemetrySnapshot, rider: RiderProfile,
            duration_h: float) -> AgentOutput:
        multiplier = SENSITIVITY_MULTIPLIER.get(rider.skin_sensitivity, 1.0)
        discount   = _calc_protection_discount(rider)
        score      = _calc_net_score(telemetry.uv_index, duration_h,
                                     telemetry.pm25, multiplier, discount)
        verdict    = _verdict_from_score(score)

        prompt = f"""You are a skin health advisor for Bengaluru two-wheeler commuters.

Current conditions:
- UV Index: {telemetry.uv_index} | PM2.5: {telemetry.pm25} μg/m³ | Dust: {telemetry.dust} μg/m³
- Temperature: {telemetry.temperature_c}°C | Humidity: {telemetry.humidity_pct}% | Rain probability: {telemetry.precipitation_prob}%
- Rider: {rider.skin_type} skin, {rider.skin_sensitivity} sensitivity
- Duration: {duration_h} hours | Net Exposure Score: {score:.2f}
- Travel Verdict: {verdict}

In 3-4 sentences, explain clearly why this verdict was given and what the rider should be aware of right now.
Focus on immediate health risk, not generic advice."""

        text, provider, model, p_tok, c_tok, cost = _llm.generate(prompt, max_tokens=300)
        return AgentOutput(
            agent=self.name, verdict=verdict, score=score,
            llm_provider=provider, llm_model=model,
            tokens_used=p_tok + c_tok, cost_usd=cost, raw_response=text,
        )


# ─────────────────────────────────────────────────────────────
# AGENT 3 — Travel Timing Advisory Agent
# ─────────────────────────────────────────────────────────────

class TimingAdvisoryAgent:
    name = "Travel Timing Advisory Agent"

    def run(self, telemetry: TelemetrySnapshot, rider: RiderProfile,
            duration_h: float, route: str) -> AgentOutput:
        multiplier = SENSITIVITY_MULTIPLIER.get(rider.skin_sensitivity, 1.0)
        discount   = _calc_protection_discount(rider)

        windows = []
        for h in range(6, 20):  # 06:00 to 19:00
            if telemetry.hourly_precip[h] >= 65:
                continue
            uv_h   = telemetry.hourly_uv[h]
            pm25_h = telemetry.hourly_pm25[h]
            score  = _calc_net_score(uv_h, duration_h, pm25_h, multiplier, discount)
            windows.append({"hour": h, "uv": round(uv_h, 2),
                             "pm25": round(pm25_h, 2), "score": round(score, 4),
                             "label": f"{h:02d}:00–{h+1:02d}:00"})

        windows_sorted = sorted(windows, key=lambda x: x["score"])
        top3 = windows_sorted[:3]

        # Compute percentage reduction vs current hour
        current_h = datetime.now().hour
        current_score = _calc_net_score(
            telemetry.uv_index, duration_h, telemetry.pm25, multiplier, discount
        )
        for w in top3:
            if current_score > 0:
                w["reduction_pct"] = round((1 - w["score"] / current_score) * 100, 1)
            else:
                w["reduction_pct"] = 0.0

        prompt = f"""You are a Bengaluru commute timing advisor.

Route: {route} | Duration: {duration_h}h | Rider sensitivity: {rider.skin_sensitivity}

Top 3 lowest-exposure departure windows today:
{json.dumps(top3, indent=2)}

In 2-3 sentences, tell the rider the best time to depart and why, referencing UV and PM2.5 levels.
Be specific with times and percentage improvements."""

        text, provider, model, p_tok, c_tok, cost = _llm.generate(prompt, max_tokens=250)
        return AgentOutput(
            agent=self.name, best_windows=top3,
            llm_provider=provider, llm_model=model,
            tokens_used=p_tok + c_tok, cost_usd=cost, raw_response=text,
        )


# ─────────────────────────────────────────────────────────────
# AGENT 4 — Skin Care Advisory Agent
# ─────────────────────────────────────────────────────────────

class SkinCareAdvisoryAgent:
    name = "Skin Care Advisory Agent"

    def run(self, telemetry: TelemetrySnapshot, rider: RiderProfile,
            verdict: str) -> AgentOutput:
        prompt = f"""You are a dermatology-informed skincare advisor for Bengaluru two-wheeler commuters.

Rider profile:
- Skin type: {rider.skin_type}
- Skin sensitivity: {rider.skin_sensitivity}
- Travel verdict: {verdict}

Environmental conditions:
- UV Index: {telemetry.uv_index} | PM2.5: {telemetry.pm25} μg/m³ | Temperature: {telemetry.temperature_c}°C | Humidity: {telemetry.humidity_pct}%

Bengaluru-specific challenges: high UV, heavy traffic dust/PM2.5, helmet friction, sweat.

Provide:
PRE-RIDE (3 bullet points): Specific barrier prep steps for this skin type
POST-RIDE (3 bullet points): Cleansing and recovery steps

Be practical, product-agnostic, and tailored to Bengaluru conditions."""

        text, provider, model, p_tok, c_tok, cost = _llm.generate(prompt, max_tokens=400)
        return AgentOutput(
            agent=self.name, advice=text,
            llm_provider=provider, llm_model=model,
            tokens_used=p_tok + c_tok, cost_usd=cost, raw_response=text,
        )


# ─────────────────────────────────────────────────────────────
# AGENT 5 — Protective Measures Recommendation Agent
# ─────────────────────────────────────────────────────────────

class ProtectiveMeasuresAgent:
    name = "Protective Measures Recommendation Agent"

    # Protection gap thresholds
    _GAPS_MAP = {
        "No SPF sunscreen":              lambda r: not r.uses_sunscreen,
        "Using SPF30 instead of SPF50+": lambda r: r.uses_sunscreen and r.spf_rating == "SPF30",
        "No sunscreen reapplication every 2–3h": lambda r: r.uses_sunscreen and not r.reapplies_sunscreen,
        "No UPF arm sleeves or full riding jacket": lambda r: not r.wears_arm_sleeves and not r.wears_full_jacket,
        "Clear visor (no UV protection)": lambda r: r.visor_type == "clear",
        "No riding gloves":              lambda r: not r.wears_gloves,
        "No cotton face mask/neck gaiter": lambda r: not r.uses_face_mask_gaiter,
    }

    def run(self, rider: RiderProfile, telemetry: TelemetrySnapshot,
            verdict: str) -> AgentOutput:
        gaps = [msg for msg, check in self._GAPS_MAP.items() if check(rider)]

        prompt = f"""You are a protective gear advisor for Bengaluru two-wheeler riders.

Current protection gaps identified:
{chr(10).join(f"• {g}" for g in gaps) if gaps else "• No major gaps detected"}

Travel verdict: {verdict}
UV Index: {telemetry.uv_index} | PM2.5: {telemetry.pm25} μg/m³
Rider sensitivity: {rider.skin_sensitivity}

Recommend exactly what gear upgrades are needed (helmet-compatible, Bengaluru traffic suitable).
For each gap, give ONE specific actionable recommendation in 1 sentence.
End with an overall protection rating out of 10."""

        text, provider, model, p_tok, c_tok, cost = _llm.generate(prompt, max_tokens=400)
        return AgentOutput(
            agent=self.name, gaps=gaps, advice=text,
            llm_provider=provider, llm_model=model,
            tokens_used=p_tok + c_tok, cost_usd=cost, raw_response=text,
        )


# ─────────────────────────────────────────────────────────────
# AGENT 6 — Skin Health Analytics Dashboard Agent
# ─────────────────────────────────────────────────────────────

class AnalyticsDashboardAgent:
    name = "Skin Health Analytics Dashboard Agent"

    def __init__(self):
        self._db_conn = None

    def _get_db(self):
        """Lazy DB connection — returns None if DB unavailable."""
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
            log.warning("analytics_agent.log_skipped_no_db")
            return False
        try:
            cur = db.cursor()
            cur.execute("""
                INSERT INTO exposure_logs
                  (rider_id, route, departure_time, duration_hours,
                   uv_index, temperature_c, humidity_pct, precipitation_prob, pm25, dust,
                   net_exposure_score, sensitivity_multiplier, protection_discount,
                   travel_verdict, best_travel_window, protection_gap_advice,
                   skincare_advice, ai_recommendations, llm_provider_used, llm_model_used,
                   prompt_tokens, completion_tokens, total_cost_usd)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                req.rider_id,
                req.route,
                req.departure_time or datetime.utcnow().isoformat(),
                req.duration_hours,
                telemetry.uv_index, telemetry.temperature_c, telemetry.humidity_pct,
                telemetry.precipitation_prob, telemetry.pm25, telemetry.dust,
                score,
                SENSITIVITY_MULTIPLIER.get(req.rider.skin_sensitivity, 1.0),
                _calc_protection_discount(req.rider),
                verdict,
                json.dumps(windows[:3]),
                "\n".join(gaps),
                skincare,
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
            return {"error": "Database unavailable", "summary": "Using cached/offline mode."}
        try:
            cur = db.cursor(dictionary=True)
            cur.execute("""
                SELECT
                    DATE(logged_at) as date,
                    COUNT(*) as trips,
                    AVG(net_exposure_score) as avg_score,
                    AVG(uv_index) as avg_uv,
                    AVG(pm25) as avg_pm25,
                    SUM(CASE WHEN travel_verdict='SAFE_TO_TRAVEL' THEN 1 ELSE 0 END) as safe,
                    SUM(CASE WHEN travel_verdict='TRAVEL_WITH_GEAR' THEN 1 ELSE 0 END) as gear,
                    SUM(CASE WHEN travel_verdict='DELAY_RECOMMENDED' THEN 1 ELSE 0 END) as delay,
                    SUM(CASE WHEN travel_verdict='AVOID_NON_ESSENTIAL' THEN 1 ELSE 0 END) as avoid,
                    SUM(total_cost_usd) as total_cost_usd
                FROM exposure_logs
                WHERE logged_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY DATE(logged_at)
                ORDER BY date DESC
            """)
            rows = cur.fetchall()
            cur.execute("""
                SELECT SUM(prompt_tokens+completion_tokens) as total_tokens,
                       SUM(total_cost_usd) as total_cost,
                       COUNT(*) as total_logs
                FROM exposure_logs WHERE logged_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            """)
            summary = cur.fetchone()
            return {"daily": rows, "summary": summary}
        except Exception as e:
            return {"error": str(e)}

    def chat(self, agent_name: str, message: str, context: Dict,
             session_id: str, rider_id: Optional[int] = None) -> AgentOutput:
        agent_descriptions = {
            "uv_monitor":    "UV Exposure Monitoring Agent — real-time UV/PM2.5 data specialist",
            "risk_verdict":  "Sun Exposure Risk & Travel Verdict Agent — calculates exposure scores",
            "timing":        "Travel Timing Advisory Agent — recommends best departure windows",
            "skincare":      "Skin Care Advisory Agent — pre/post ride skincare expert",
            "protection":    "Protective Measures Agent — gear gap analysis and recommendations",
            "analytics":     "Skin Health Analytics Agent — weekly trends and health patterns",
        }
        agent_desc = agent_descriptions.get(agent_name, "General Bengaluru Rider Advisor Agent")
        ctx_str    = json.dumps(context, default=str)[:800]

        prompt = f"""You are the {agent_desc} for Bengaluru two-wheeler commuters.

Current session context:
{ctx_str}

Rider question: {message}

Answer concisely and helpfully in 2-4 sentences, specific to Bengaluru riding conditions."""

        text, provider, model, p_tok, c_tok, cost = _llm.generate(prompt, max_tokens=350)

        # Log to DB if available
        db = self._get_db()
        if db:
            try:
                cur = db.cursor()
                cur.executemany(
                    "INSERT INTO agent_chat_history (rider_id,session_id,agent_name,role,message,tokens_used,cost_usd) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (rider_id, session_id, agent_name, "user",      message, 0,              0.0),
                        (rider_id, session_id, agent_name, "assistant", text,    p_tok + c_tok, cost),
                    ]
                )
                db.commit()
            except Exception as e:
                log.warning("chat_log_failed", error=str(e))

        return AgentOutput(
            agent=f"Chat:{agent_name}", advice=text,
            llm_provider=provider, llm_model=model,
            tokens_used=p_tok + c_tok, cost_usd=cost, raw_response=text,
        )


# ─────────────────────────────────────────────────────────────
# Human-in-the-Loop (HITL) Trigger
# ─────────────────────────────────────────────────────────────

def _trigger_hitl(score: float, verdict: str, rider: RiderProfile,
                  telemetry: TelemetrySnapshot) -> bool:
    """Posts to webhook when risk score exceeds HITL_RISK_THRESHOLD."""
    if not HITL_WEBHOOK_URL or score < HITL_RISK_THRESHOLD:
        return False
    payload = {
        "alert": "HIGH_RISK_RIDE",
        "score": score, "verdict": verdict,
        "rider_sensitivity": rider.skin_sensitivity,
        "uv_index": telemetry.uv_index, "pm25": telemetry.pm25,
        "timestamp": datetime.utcnow().isoformat(),
    }
    try:
        httpx.post(HITL_WEBHOOK_URL, json=payload, timeout=5)
        log.info("hitl_triggered", score=score, verdict=verdict)
        return True
    except Exception as e:
        log.warning("hitl_webhook_failed", error=str(e))
        return False


# ─────────────────────────────────────────────────────────────
# Guardrails
# ─────────────────────────────────────────────────────────────

SAFE_RESPONSE = "I can only provide information related to UV exposure, skin health, and riding safety for Bengaluru commuters."

def _apply_guardrails(user_input: str) -> Optional[str]:
    """Returns a blocked message if input is off-topic/harmful, else None."""
    blocked_topics = [
        "hack", "exploit", "malware", "jailbreak", "ignore previous",
        "ignore all instructions", "prompt injection", "bypass",
    ]
    lower = user_input.lower()
    for topic in blocked_topics:
        if topic in lower:
            return SAFE_RESPONSE
    if len(user_input) > 2000:
        return "Please keep your question under 2000 characters."
    return None


# ─────────────────────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────

class AdvisoryPipeline:
    """Runs all 6 agents in sequence and returns a consolidated response."""

    def __init__(self):
        self.uv_agent         = UVMonitoringAgent()
        self.risk_agent       = RiskVerdictAgent()
        self.timing_agent     = TimingAdvisoryAgent()
        self.skincare_agent   = SkinCareAdvisoryAgent()
        self.protection_agent = ProtectiveMeasuresAgent()
        self.analytics_agent  = AnalyticsDashboardAgent()

    async def run(self, req: TripRequest) -> FullAdvisoryResponse:
        start = time.time()

        # Agent 1 — Telemetry
        telemetry = await self.uv_agent.fetch()

        # Agent 2 — Risk & Verdict
        risk_out = self.risk_agent.run(telemetry, req.rider, req.duration_hours)

        # Agent 3 — Timing
        timing_out = self.timing_agent.run(telemetry, req.rider, req.duration_hours, req.route)

        # Agent 4 — Skincare
        skincare_out = self.skincare_agent.run(telemetry, req.rider, risk_out.verdict)

        # Agent 5 — Protective Measures
        protection_out = self.protection_agent.run(req.rider, telemetry, risk_out.verdict)

        agents = [risk_out, timing_out, skincare_out, protection_out]
        total_tokens = sum(a.tokens_used for a in agents)
        total_cost   = sum(a.cost_usd   for a in agents)

        # HITL check
        hitl = _trigger_hitl(risk_out.score, risk_out.verdict, req.rider, telemetry)

        # Agent 6 — Analytics logging
        ai_recs = {
            "risk":       risk_out.raw_response,
            "timing":     timing_out.raw_response,
            "skincare":   skincare_out.raw_response,
            "protection": protection_out.raw_response,
        }
        self.analytics_agent.log_exposure(
            req, telemetry, risk_out.score, risk_out.verdict,
            timing_out.best_windows or [], skincare_out.advice or "",
            protection_out.gaps or [], ai_recs,
            risk_out.llm_provider, risk_out.llm_model,
            total_tokens // 2, total_tokens // 2, total_cost,
        )

        elapsed = round(time.time() - start, 2)
        log.info("pipeline.complete", elapsed_s=elapsed, verdict=risk_out.verdict, score=risk_out.score)

        return FullAdvisoryResponse(
            session_id=req.session_id,
            timestamp=datetime.utcnow().isoformat(),
            telemetry=telemetry,
            risk_score=risk_out.score,
            travel_verdict=risk_out.verdict,
            verdict_explanation=risk_out.raw_response,
            best_travel_windows=timing_out.best_windows or [],
            skincare_advice=skincare_out.advice or "",
            protection_gaps=protection_out.gaps or [],
            gear_recommendations=protection_out.advice or "",
            agents=agents,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            hitl_triggered=hitl,
        )
