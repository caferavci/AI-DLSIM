# AI-DLSIM

AI-DLSIM is an AI-agent-driven workflow around dynamic traffic simulation using DLSim as the simulation engine.

## Current Project Status

The repository currently supports:

- LLM-based query parsing and result interpretation
- location resolution from place names to network node IDs
- generation of `input_agent.csv` from natural-language queries
- DLSim execution scripts that produce simulation CSV outputs
- baseline service/adapter scaffolding under `src/ai_dlsim/`

The codebase is configured around the Ithaca dataset in `data/Ithaca/`.

## Repository Structure

```text
AI-DLSIM/
  InterfaceSpecification.md
  requirements.txt
  scripts/
    run_dlsim.py
  src/ai_dlsim/
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

## How To Run

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

### C) End-to-end AI query pipeline

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

Typical output files:

- `input_agent.csv`
- `link_performance.csv`
- `agent.csv`
- `solution.csv`

## Notes

- Baseline data is Ithaca-specific.
- Interface and schema details are defined in `InterfaceSpecification.md`.
- If a module import fails (for example `dotenv`), ensure the virtual environment is activated and dependencies were installed in that same environment.
