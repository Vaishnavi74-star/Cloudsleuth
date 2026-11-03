# 🕵️ CloudSleuth: Evidence-Based Microservice Incident RCA

CloudSleuth is a deterministic, explainable root-cause analysis (RCA) system built for microservice production incidents. It uses real benchmark telemetry from [RCAEval](https://github.com/phamquiluan/RCAEval) (the open-source benchmark for microservice systems published at ACM WWW 2025 and FSE 2026).

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify that data loading, ground-truth isolation, statistical ranking, and benchmark evaluation pass:
```bash
python -m pytest tests -v
```

### 3. Launch Streamlit Dashboard
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎯 Architecture & Methodology

CloudSleuth rejects the unreliable approach of feeding hundreds of raw log dumps into an LLM to guess root causes. Instead, it follows a **4-stage evidence-based pipeline**:

1. **Telemetry Alignment & Preprocessing**:
   - Ingests wide time-series metrics, log burst time series, and distributed trace error/latency series.
   - Eliminates constant metrics and cleans missing values without lookahead bias.
2. **Robust Statistical Anomaly Detection**:
   - Uses `RobustScaler` (median centering and interquartile scaling) to compute outlier z-scores invariant to variance and heavy tails.
   - Computes changepoint onset timing ($T+\Delta t$) to differentiate initial triggers from delayed reactions.
3. **Topological Disambiguation (Symptom vs. Root Cause)**:
   - Microservice incidents exhibit cascade effects: an edge service (`frontend`) latency spike is almost always a downstream symptom of a backend bottleneck.
   - Direct resource exhaustion metrics (CPU, Memory, Disk IO) on downstream dependencies receive higher causal weight than edge latency spikes.
4. **Multi-Source Evidence Fusion**:
   - Corroborates metric spikes with log error frequency bursts and distributed trace error rates.
5. **Strict Ground-Truth Isolation**:
   - In accordance with the RCAEval benchmark protocol, ground-truth labels (`service`, `fault`) are **never accessible** to the diagnosis engine. They are queried strictly after predictions are made to calculate evaluation metrics.

---

## 📊 Real Benchmark Cases Evaluated

CloudSleuth tests against verified real telemetry from RCAEval:
- **`sample_baro_emailservice_mem`**: Official single-case metric benchmark (`data.csv`, 721 timesteps, 58 metrics).
  - Ground Truth: `emailservice` (Memory leak).
  - CloudSleuth Top-1 Prediction: `emailservice` (**MATCHED ✅**).
- **`multi_source_re2_ob`**: RE2 Multi-Source case (`metrics.csv`, `logts.csv`, `tracets_err.csv`, `tracets_lat.csv`).
  - Ground Truth: `redis` (Disk IO fault).
  - CloudSleuth Top-1 Prediction: `redis` (**MATCHED ✅**).
- **`re1_ob_*`**: 125 real failure cases from the Online Boutique RE1 suite covering CPU, Memory, Disk, and Delay faults across microservices.

### Verified Evaluation Metrics (Sample Run of 5 Benchmark Cases)
- **Top-1 Accuracy (Hit@1)**: **60.0%**
- **Top-3 Accuracy (Hit@3)**: **100.0%**
- **Top-5 Accuracy (Hit@5)**: **100.0%**
- **Avg@5 Score**: **0.920**
- **Average Diagnosis Latency**: **~38 ms / case**

---

## 🤖 Google Gemma Integration

CloudSleuth integrates Google's **Gemma** model (`gemma-2-9b-it`) to augment deterministic root-cause analysis with natural language capabilities:

1. **AI-Powered Incident Explanation**:
   - Converts quantitative statistical findings into simple, accessible English for on-call SREs.
   - Answers: *What happened? Which service is responsible? Why was it selected? What evidence supports it? Which services are affected? What to investigate next?*
   - Strictly bounded by measured evidence. Never hallucinates metrics or proves causality beyond the data.
2. **Automatic Incident Report Generation**:
   - Compiles a standardized 8-section Incident Postmortem Report ready for leadership.
   - Sections: *Summary, Details, Suspected Root Cause, Telemetry Evidence Table, Affected Services, Timeline, Troubleshooting Steps, and Limitations*.
   - One-click export to Markdown (`.md`) and Plaintext (`.txt`).
3. **Strict Ground-Truth Isolation**:
   - Benchmark ground truth is **never provided** to Gemma for prediction or explanation.
   - Deterministic RCA remains the sole root-cause engine; Gemma acts strictly as an evidence explainer and synthesizer.
4. **Resilient Offline Fallback**:
   - If `GEMMA_API_KEY` is not configured or an API endpoint fails, CloudSleuth automatically engages its built-in deterministic evidence explainer with zero hallucination risk.

---

## 🩺 Website Health Monitoring (External Surveillance)

CloudSleuth includes an external, passive health surveillance engine that enables operators to enter any publicly accessible website URL and inspect its availability and performance without needing access to that website's internal telemetry.

### Key Capabilities
- **URL Inspection & Reachability**: Immediate HTTP/HTTPS connectivity verification with response code detection (2xx, 3xx, 4xx, 5xx).
- **Latency Measurement**: Accurate response time measurement with high-resolution timers.
- **SSL/TLS Certificate Inspection**: Automatic certificate validity check, issuer detection, and expiration tracking.
- **Protocol Failure Classification**: Categorizes errors into `dns_failure`, `ssl_error`, `timeout`, `connection_refused`, `4xx_client_error`, `5xx_server_error`, `redirect_loop`, or `ssrf_blocked`.
- **Observable Symptoms vs. Confirmed Issues**: Rigorously separates externally observable surface symptoms (e.g., HTTP 502, slow latency) from confirmed facts, explicitly clarifying that internal server bugs/deadlocks cannot be confirmed externally.
- **Gemma AI Explanation & Action Plan**: Translates raw protocol errors into plain-English explanations and actionable troubleshooting steps for operators and visitors.
- **Audit History & CSV Export**: Session-based health check history log with timestamped records and downloadable CSV.

### 🛡️ Security & SSRF Protection
- **Protocol Restrictions**: Exclusively allows HTTP and HTTPS schemes.
- **SSRF Defense**: Automatically blocks all requests to private networks (RFC 1918), loopback (`localhost`, `127.0.0.1`, `::1`), link-local, and cloud metadata endpoints (`169.254.169.254`, `metadata.google.internal`).
- **Redirect Validation**: Validates every intermediate redirect hop and resolved IP address.
- **Safety Caps & Rate Limiting**: Enforces connection/read timeouts, 1 MB response body caps, and a 10 check/minute sliding window rate limiter.
- **Non-Intrusive**: Performs single passive read-only probes with no port scanning, brute-forcing, or stress testing.

### ⚙️ Gemma Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your free Google AI Studio key:
```env
GEMMA_API_KEY=your_free_google_ai_studio_key
GEMMA_MODEL=gemma-4-26b-a4b-it
```
*(Free API keys with generous quotas are available at [Google AI Studio](https://aistudio.google.com/)).*

---

## 🐳 Docker Deployment Guide

CloudSleuth is fully containerized using a lightweight `python:3.11-slim` base image, layer caching, and built-in healthchecks.

### 1. Build the Docker Image
```bash
docker build -t cloudsleuth .
```

### 2. Run Locally with Docker
Ensure your `.env` file exists with your `GEMMA_API_KEY`:
```bash
docker run -d \
  --name cloudsleuth \
  -p 8501:8501 \
  --env-file .env \
  cloudsleuth
```

### 3. Run with Docker Compose (Recommended)
Alternatively, start the service using Docker Compose:
```bash
docker compose up -d
```

### 4. Verify Local Deployment
- **Web Dashboard**: [http://localhost:8501](http://localhost:8501)
- **Container Healthcheck**: `http://localhost:8501/_stcore/health`
- **View Container Logs**:
  ```bash
  docker logs -f cloudsleuth
  ```
- **Run Test Suite Inside Container**:
  ```bash
  docker exec cloudsleuth python -m pytest tests -v
  ```

### 5. Stop and Remove the Container
```bash
docker stop cloudsleuth
docker rm cloudsleuth
# Or if using Compose:
docker compose down
```

---

## ☁️ AWS EC2 Deployment Guide (Ubuntu)

Follow these steps to deploy CloudSleuth to an AWS EC2 instance without incurring paid managed services.

### 1. Launch EC2 Instance & Configure Security Group
1. Launch an **Ubuntu 24.04 LTS** or **22.04 LTS** instance:
   - Recommended instance type: **`t3.small`** (2 vCPU, 2 GiB RAM) or **`t3.medium`** (2 vCPU, 4 GiB RAM).
   - Storage: 20–30 GiB General Purpose SSD (gp3).
2. Configure **Security Group** Inbound Rules:
   - **SSH**: Type `SSH`, Port `22`, Source `My IP` (for secure administration).
   - **Streamlit Web UI**: Type `Custom TCP`, Port Range `8501`, Source `0.0.0.0/0` (or your company IP range).

### 2. Connect to EC2 and Install Docker
SSH into your instance:
```bash
ssh -i /path/to/your-key.pem ubuntu@<EC2-PUBLIC-IP>
```
Install Docker Engine on Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker packages
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Allow running docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

### 3. Deploy CloudSleuth on EC2

#### Option A: Clone & Build on EC2
```bash
git clone <YOUR_REPOSITORY_URL> cloudsleuth
cd cloudsleuth

# Create sanitized .env file on the remote server
nano .env
# Paste your GEMMA_API_KEY=... and GEMMA_MODEL=gemma-4-26b-a4b-it
```
Build and start the container:
```bash
docker build -t cloudsleuth .
docker run -d \
  --name cloudsleuth \
  --restart unless-stopped \
  -p 8501:8501 \
  --env-file .env \
  cloudsleuth
```

#### Option B: Transfer Local Image via SSH Pipeline (No Registry Required)
From your local terminal, build and stream the compressed Docker image directly to EC2:
```bash
docker save cloudsleuth | gzip | ssh -i /path/to/key.pem ubuntu@<EC2-PUBLIC-IP> 'gunzip | docker load'
```
Then on the EC2 instance:
```bash
docker run -d \
  --name cloudsleuth \
  --restart unless-stopped \
  -p 8501:8501 \
  -e GEMMA_API_KEY="your_api_key_here" \
  cloudsleuth
```

### 4. Access the CloudSleuth Dashboard
Open your browser and navigate to:
```
http://<EC2-PUBLIC-IP>:8501
```

### 5. Managing the EC2 Container
- **Check Status**: `docker ps`
- **View Live Logs**: `docker logs -f cloudsleuth`
- **Restart Container**: `docker restart cloudsleuth`
- **Stop & Remove Container Safely**:
  ```bash
  docker stop cloudsleuth && docker rm cloudsleuth
  ```

---

## 🔒 Security & Environment Architecture

- **Credential Isolation**: `.env` and sensitive API keys are explicitly blocked from the Docker build context by `.dockerignore`.
- **Environment Injection**: API credentials (`GEMMA_API_KEY`) are injected exclusively at runtime via `--env-file .env` or `-e GEMMA_API_KEY=...`.
- **Zero API Key Leakage**: Keys are never printed in console logs, frontend Streamlit output, postmortem reports, or committed to git.
- **Automated Security Unit Test**: `tests/test_cloudsleuth.py::test_docker_security_and_configuration` continuously verifies that `.dockerignore` prevents `.env` inclusion.

---

## 📁 Repository Structure
```
├── app.py                      # Interactive Streamlit RCA dashboard (with Gemma features)
├── Dockerfile                  # Lightweight Python 3.11-slim production container
├── .dockerignore               # Security & build optimization exclusion rules
├── docker-compose.yml          # Container orchestration & healthcheck configuration
├── cloudsleuth/
│   ├── __init__.py
│   ├── data_loader.py          # Real telemetry loader & ground-truth isolation
│   ├── engine.py               # Deterministic statistical RCA engine
│   ├── evaluation.py           # Benchmark evaluator (Hit@1/3/5, Avg@5)
│   ├── gemma_service.py        # Google Gemma integration client & prompt engine
│   └── website_monitor.py      # Passive website health monitor with SSRF protection
├── tests/
│   ├── test_cloudsleuth.py     # 13 RCAEval & Docker security unit tests
│   └── test_website_monitor.py # 19 health monitoring & SSRF validation tests
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment configuration template
├── README.md                   # Comprehensive documentation & deployment guides
├── data.csv                    # Real sample telemetry (RCAEval reference)
└── data/                       # RCAEval datasets (RE1 and RE2)
```

---

