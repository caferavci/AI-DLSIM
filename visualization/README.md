# AI-DLSIM Visualization

React + Leaflet + Chart.js dashboard for AI-DLSIM run outputs.

## 1) Start FastAPI backend

From repo root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_dashboard_api.py
```

API base URL: `http://localhost:8000`

## 2) Start frontend

From `visualization/`:

```bash
npm install
cp .env.example .env
npm run dev
```

Frontend URL: `http://localhost:5173`

## Available backend endpoints

- `GET /health`
- `GET /runs`
- `GET /runs/{run_id}/summary`
- `GET /runs/{run_id}/network?max_links=1500`
- `GET /runs/{run_id}/route`

