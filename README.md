# AI-DLSIM

AI-DLSIM is an AI-agent-driven workflow for traffic simulation and map-based result visualization.

## Current Architecture

The current query pipeline supports a two-engine orchestration model:

- **DLSim route engine**: generates route/path and trip-level simulation output
- **DTALite traffic engine**: intended for traffic-state evaluation on route/network
- **Fusion layer**: combines engine outputs into a single LLM-readable summary
- **Visualization app**: React + Leaflet UI backed by FastAPI endpoints

Pipeline stages:

1. LLM query parsing (`origin`, `destination`, `departure_time`, `mode`)
2. place-name to node resolution
3. `input_agent.csv` generation
4. run DLSim route engine
5. run DTALite traffic engine
6. parse/fuse outputs + generate final answer text

## Repository Structure

```text
AI-DLSIM/
  InterfaceSpecification.md
  requirements.txt
  scripts/
    run_dashboard_api.py
    run_dlsim.py
  src/ai_dlsim/
    api/
    adapters/
    preprocessing/
    simulation/
    postprocessing/
    schemas/
    workflows/
  data/
    Ithaca/
      14850_roads.osm
      node.csv
      link.csv
      demand.csv
      input_agent.csv
  outputs/
    runs/
  visualization/
```

## Setup

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Configure API key (for LLM pre/post agents)

Create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=your_key_here
```

`run_query_pipeline.py` loads `.env` automatically.

### 3) Initialize DLSim submodule (if needed)

```bash
git submodule update --init --recursive
```

## How To Run (Backend)

### A) Baseline scaffold run

```bash
python3 src/ai_dlsim/workflows/run_baseline.py
```

This validates baseline wiring and prints a structured status dictionary.

### B) Direct DLSim scenario run

```bash
python3 scripts/run_dlsim.py
```

This script copies required inputs into `outputs/runs/ithaca_dlsim/`, runs DLSim, and reports produced output files.

### C) End-to-end AI query pipeline (dual-engine orchestration)

```bash
python3 src/ai_dlsim/workflows/run_query_pipeline.py \
  --query "What is the travel time from Cornell to Ithaca Commons at 8:30 AM?"
```

Optional model override:

```bash
python3 src/ai_dlsim/workflows/run_query_pipeline.py \
  --query "How long does it take to drive from Cornell University to Ithaca Commons at 9 AM?" \
  --llm-model "openai.gpt-5-mini"
```

## Run Dashboard (Frontend + API)

### 1) Start FastAPI server

```bash
source .venv/bin/activate
python scripts/run_dashboard_api.py
```

Default URL: `http://localhost:8000`

### 2) Start visualization app

```bash
cd visualization
npm install
npm run dev
```

Default URL: `http://localhost:5173`

### 3) Submit query from frontend

Use the prompt input in the dashboard UI and click **Run Query**.  
This calls `POST /query`, triggers `run_query_pipeline.py`, and refreshes results automatically.

## Example Prompts

Use prompts that clearly include origin, destination, and departure time:

- `What is the travel time from Cornell University to Ithaca Commons at 8:30 AM?`
- `Estimate driving time from Collegetown to Ithaca Tompkins International Airport at 7:45 AM.`
- `If I leave downtown Ithaca at 5:15 PM, how long to reach Cornell?`
- `What route and travel time do you predict from Cayuga Heights to Ithaca Commons at 9:00 AM?`
- `How long is the trip from Cornell to Wegmans at 6:30 PM by car?`

## Outputs

Typical output folders:

- `outputs/runs/ithaca_dlsim/`
- `outputs/runs/query_pipeline/`
- `outputs/runs/query_pipeline/dlsim_engine/`
- `outputs/runs/query_pipeline/dtalite_engine/`

Typical output files:

- `input_agent.csv`
- `link_performance.csv`
- `agent.csv`
- `solution.csv`
- `final_answer.txt`
- `dashboard_summary.json`

Engine-specific files are written under `dlsim_engine/` and `dtalite_engine/`.

## FastAPI Endpoints (current)

- `GET /health`
- `GET /runs`
- `GET /runs/{run_id}/summary`
- `GET /runs/{run_id}/network?max_links=1500`
- `GET /runs/{run_id}/route`
- `GET /runs/{run_id}/engines`
- `GET /runs/{run_id}/engines/{engine}/view?max_links=1500`
- `POST /query` (run pipeline from frontend)

## Notes

- Baseline data is Ithaca-specific.
- Interface and schema details are defined in `InterfaceSpecification.md`.
- If a module import fails (for example `dotenv`), ensure the virtual environment is activated and dependencies were installed in that same environment.
- If DTALite fails with `Bad CPU type in executable`, the current binary is not compatible with your machine architecture. DLSim route-engine output will still run.
