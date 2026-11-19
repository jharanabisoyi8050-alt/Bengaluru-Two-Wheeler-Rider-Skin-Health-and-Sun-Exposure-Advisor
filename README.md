# Bengaluru Two-Wheeler Rider Skin Health & Sun Exposure Advisor

> **IBM Bob Hackathon — Problem Statement 12**  
> Domain: Healthcare, Environmental Health & Road Mobility  
> Team: Multi-Agent AI on IBM watsonx.ai

---

## 🏍️ What It Does

A production-ready **Multi-Agent AI Application** that analyzes real-time UV Index, air quality (PM2.5/dust), weather, and rider profile to give Bengaluru two-wheeler commuters:

1. ✅ **Travel Verdict** — `SAFE_TO_TRAVEL`, `TRAVEL_WITH_GEAR`, `DELAY_RECOMMENDED`, or `AVOID_NON_ESSENTIAL`  
2. ⏰ **Best Travel Windows** — Top 3 hourly slots with minimum UV + pollution exposure  
3. 🧴 **Personalized Skincare** — Pre-ride & post-ride protocol for your skin type  
4. 🛡️ **Gear Gap Analysis** — What protection you're missing and exactly what to add  
5. 💬 **AI Agent Console** — Live chat with any specialist agent  
6. 📊 **Analytics Dashboard** — Weekly exposure trends + LLM cost monitoring

---

## 🤖 Multi-Agent Architecture

| Agent | Role | LLM |
|-------|------|-----|
| UV Exposure Monitoring Agent | Real-time Open-Meteo telemetry fetch | — |
| Risk & Travel Verdict Agent | Net exposure score + travel verdict | IBM Granite |
| Travel Timing Advisory Agent | Best departure window analysis | IBM Granite / Groq |
| Skin Care Advisory Agent | Pre/post ride skincare protocol | IBM Granite |
| Protective Measures Agent | Gear gap audit & recommendations | IBM Granite |
| Analytics Dashboard Agent | MySQL logging, chat console, weekly trends | IBM Granite |

**LLM Routing:** IBM Granite (`ibm/granite-3-3-8b-instruct`) → Groq (`llama-3.3-70b-versatile`) → Gemini (`gemini-2.5-flash`)

---

## 🗂️ Project Structure

```
bengaluru-rider-advisor/
├── backend/
│   ├── agents_engine.py      # All 6 agents, LLM gateway, pipeline
│   └── main.py               # FastAPI server + REST endpoints
├── frontend/
│   └── index.html            # Full responsive UI with Chart.js
├── database/
│   └── schema.sql            # MySQL schema + seed data
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── Dockerfile                # IBM Cloud Code Engine ready
└── README.md
```

---

## ⚡ Quick Start (Local)

### Prerequisites
- Python 3.11+
- MySQL 8.0+ (or IBM Cloud Databases for MySQL)
- IBM Cloud account with watsonx.ai project

### 1. Clone the repository
```bash
git clone https://github.com/jharanabisoyi8050-alt/Bengaluru-Two-Wheeler-Rider-Skin-Health-and-Sun-Exposure-Advisor.git
cd Bengaluru-Two-Wheeler-Rider-Skin-Health-and-Sun-Exposure-Advisor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env with your IBM Cloud API key, watsonx Project ID, MySQL creds, etc.
```

### 4. Initialize MySQL database
```bash
mysql -u root -p < database/schema.sql
```

### 5. Run the server
```bash
cd backend
python main.py
```

Open **http://localhost:8000** in your browser.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/advisory` | Full 6-agent analysis pipeline |
| `POST` | `/api/agent-chat` | Interactive AI Agent Console |
| `GET`  | `/api/analytics` | Weekly dashboard analytics |
| `GET`  | `/api/health` | Health check (Code Engine probe) |
| `GET`  | `/api/routes` | Available Bengaluru corridors |
| `GET`  | `/` | Serve frontend UI |

### Example: Get Advisory
```bash
curl -X POST http://localhost:8000/api/advisory \
  -H "Content-Type: application/json" \
  -d '{
    "route": "Electronic City – Indiranagar",
    "duration_hours": 1.25,
    "skin_type": "oily",
    "skin_sensitivity": "high",
    "uses_sunscreen": true,
    "spf_rating": "SPF50+",
    "wears_arm_sleeves": false,
    "visor_type": "clear"
  }'
```

---

## 🐳 Docker Build & Run

```bash
# Build
docker build -t bengaluru-rider-advisor .

# Run (with env file)
docker run -p 8000:8000 --env-file .env bengaluru-rider-advisor
```

---

## ☁️ Deploy on IBM Cloud Code Engine

### Prerequisites
- [IBM Cloud CLI](https://cloud.ibm.com/docs/cli)
- Code Engine plugin: `ibmcloud plugin install code-engine`
- Container Registry access

```bash
# 1. Login to IBM Cloud
ibmcloud login --apikey $IBM_CLOUD_API_KEY -r us-south

# 2. Target a resource group
ibmcloud target -g Default

# 3. Create / target a Code Engine project
ibmcloud ce project create --name rider-advisor
ibmcloud ce project select --name rider-advisor

# 4. Push image to IBM Container Registry
ibmcloud cr namespace-add rider-advisor
docker tag bengaluru-rider-advisor us.icr.io/rider-advisor/bengaluru-rider-advisor:latest
docker push us.icr.io/rider-advisor/bengaluru-rider-advisor:latest

# 5. Create the Code Engine application
ibmcloud ce application create \
  --name bengaluru-rider-advisor \
  --image us.icr.io/rider-advisor/bengaluru-rider-advisor:latest \
  --port 8000 \
  --cpu 1 \
  --memory 2G \
  --env IBM_CLOUD_API_KEY=$IBM_CLOUD_API_KEY \
  --env WATSONX_PROJECT_ID=$WATSONX_PROJECT_ID \
  --env GROQ_API_KEY=$GROQ_API_KEY \
  --env GEMINI_API_KEY=$GEMINI_API_KEY \
  --env MYSQL_HOST=$MYSQL_HOST \
  --env MYSQL_USER=$MYSQL_USER \
  --env MYSQL_PASSWORD=$MYSQL_PASSWORD \
  --env MYSQL_DB=bengaluru_rider_advisor

# 6. Get your application URL
ibmcloud ce application get --name bengaluru-rider-advisor
```

### IBM Cloud Databases for MySQL

```bash
# Provision a MySQL instance
ibmcloud resource service-instance-create rider-mysql databases-for-mysql standard us-south

# Get connection credentials and update your Code Engine env vars
ibmcloud resource service-key-create rider-mysql-key Administrator \
  --instance-name rider-mysql
```

---

## 🔧 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `IBM_CLOUD_API_KEY` | ✅ | IBM Cloud API key (IAM) |
| `WATSONX_URL` | ✅ | `https://us-south.ml.cloud.ibm.com` |
| `WATSONX_PROJECT_ID` | ✅ | Your watsonx.ai project ID |
| `GROQ_API_KEY` | ⚡ | Groq API key (fallback LLM) |
| `GEMINI_API_KEY` | ⚡ | Google Gemini API key (2nd fallback) |
| `MYSQL_HOST` | ✅ | MySQL host |
| `MYSQL_USER` | ✅ | MySQL username |
| `MYSQL_PASSWORD` | ✅ | MySQL password |
| `MYSQL_DB` | ✅ | `bengaluru_rider_advisor` |
| `HITL_WEBHOOK_URL` | 📌 | Optional Human-in-the-Loop webhook |
| `HITL_RISK_THRESHOLD` | 📌 | Score to trigger HITL (default: 15.0) |

---

## 🛡️ Security & Guardrails

- **Input guardrails:** Prompt injection detection blocks off-topic/harmful inputs
- **Human-in-the-Loop:** High risk scores (>15.0) trigger webhook alerts
- **LLM routing:** Automatic provider failover with graceful degradation
- **CORS:** Configurable allowed origins
- **Observability:** Structured JSON logging (structlog), request tracing headers

---

## 📐 Risk Score Formula

```
Net_Score = ((UV_Index × Duration_Hours) + (PM2.5 / 45))
            × Sensitivity_Multiplier
            × (1 − Protection_Discount)
```

| Verdict | Score Range |
|---------|-------------|
| ✅ SAFE_TO_TRAVEL | < 3.0 |
| ⚠️ TRAVEL_WITH_GEAR | 3.0 – 7.0 |
| 🕐 DELAY_RECOMMENDED | 7.0 – 12.0 |
| 🚫 AVOID_NON_ESSENTIAL | ≥ 12.0 |

---

## 🏆 Hackathon Feature Checklist

- [x] Working Agentic AI application with dashboard
- [x] Multi-Agent System (6 specialized agents)
- [x] Multiple LLM providers (IBM Granite + Groq + Gemini)
- [x] MCP-compatible tool integration via REST
- [x] Agent memory via MySQL session logging
- [x] LLM Gateway with token & cost monitoring
- [x] Observability: structured logging, request tracing
- [x] Guardrails: input validation, prompt injection blocking
- [x] Human-in-the-Loop: high-risk webhook alerts
- [x] AI governance: audit logs, provider fallback policies
- [x] Dockerized for IBM Cloud Code Engine deployment

---

## 📄 License

MIT License — built for the IBM Bob Hackathon with IBM SkillsBuild.
