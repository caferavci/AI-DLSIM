# AI-DLSIM

AI-DLSIM is an AI-agent-driven workflow around dynamic traffic simulation using DLSim as the simulation engine.

## Baseline Status

The repository currently contains a **baseline scaffold** with:

- modular package structure under `src/ai_dlsim/`
- preprocessing and postprocessing LLM entrypoints
- simulation service + adapter boundary
- workflow scripts for baseline execution
- interface specification in `InterfaceSpecification.md`

Current baseline is intentionally partial:

- preprocessing data pipeline is teammate-owned and in progress
- simulation adapter is currently a stub (real DLSim run integration is next)
- visualization/reporting modules are planned next

## Repository Structure (Baseline)

```text
AI-DLSIM/
  InterfaceSpecification.md
  requirements.txt
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
```

## Implemented Baseline Modules

- `src/ai_dlsim/preprocessing/llm_query_parser.py`  
  Raw user query -> structured request (`QueryRequest`)

- `src/ai_dlsim/simulation/simulation_service.py`  
  Simulation orchestration service (currently uses stub adapter)

- `src/ai_dlsim/adapters/dlsim_adapter.py`  
  Boundary for external DLSim integration (stubbed for now)

- `src/ai_dlsim/postprocessing/llm_result_interpreter.py`  
  Converts structured simulation output into user-facing text

- `src/ai_dlsim/workflows/run_baseline.py`  
  Deterministic baseline run path using local Ithaca files

- `src/ai_dlsim/workflows/run_query_pipeline.py`  
  End-to-end query -> pre-agent -> simulation -> post-agent flow

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If using OpenAI pre/post agents:

```bash
export OPENAI_API_KEY="your_key_here"
```

## Run Baseline Workflows

### 1) Baseline deterministic workflow

```bash
python3 src/ai_dlsim/workflows/run_baseline.py
```

### 2) AI query pipeline workflow

```bash
python3 src/ai_dlsim/workflows/run_query_pipeline.py \
  --query "Estimate travel time from Cornell University to Ithaca Commons at 08:30"
```

## Notes

- Baseline is currently configured around Ithaca data.
- DLSim external integration will replace the adapter stub in Phase 4.
- Preprocessing interfaces and required schemas are defined in `InterfaceSpecification.md`.
