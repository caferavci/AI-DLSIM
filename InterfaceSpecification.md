# AI-DLSIM Interface Specification

## Purpose

This document defines module boundaries, input/output specifications, and shared data schemas for AI-DLSIM.
The goal is to enable parallel development across preprocessing, simulation, and postprocessing with minimal integration rework.

## System Modules

- `preprocessing`: Converts user/query context into simulation-ready input files.
- `simulation`: Executes DLSim with validated inputs and returns run outputs.
- `postprocessing`: Parses simulation outputs and computes summarized metrics.
- `agents`:
  - pre-processing agent: natural language query -> structured request
  - post-processing agent: summarized metrics -> user-facing explanation

## 1) Pre-Agent Interface Specification

### Input
- `user_query: str`

### Output Schema (`QueryRequest`)
- `region: str` (required)
- `origin: str | null`
- `destination: str | null`
- `departure_time: str | null` (format `HH:MM`, 24h)
- `mode: str` (default: `car`)
- `scenario: str` (default: `baseline`)
- `analysis_type: str | null` (optional extension)
- `time_window: str | null` (optional extension)
- `constraints: dict | null` (optional extension)

### Notes
- This schema should stay stable so preprocessing logic does not depend on prompt phrasing.

## 2) Preprocessing Interface Specification

### Input
- `QueryRequest`
- static config (API keys, provider options, area defaults)

### Output Files (required)
- `node.csv`
- `link.csv`
- `input_agent.csv` (or generated from an intermediate demand representation)

### Required `node.csv` fields (minimum)
- `node_id`
- `zone_id`

### Required `link.csv` fields (minimum)
- `from_node_id`
- `to_node_id`
- `length`
- `lanes`
- `free_speed`
- `capacity`
- `link_type`
- `VDF_alpha1`
- `VDF_beta1`

### Required `input_agent.csv` fields (minimum)
- `o_node_id`
- `d_node_id`
- `departure_time_in_min`
- `PCE`
- `path_fixed_flag`
- `path_node_sequence`

### Behavior
- If intermediate demand files are used (e.g., `demand.csv`), preprocessing must convert them into DLSim-ready `input_agent.csv`.

## 3) Simulation Interface Specification

### Input
- `node_csv_path`
- `link_csv_path`
- `input_agent_csv_path`
- run settings:
  - `simulation_horizon_seconds`
  - `time_step_seconds`
  - `capacity_multiplier`
  - working/output directory

### Output Files (required)
- `link_performance.csv`
- `agent.csv`
- `solution.csv`

### Output Metadata (recommended)
- `status: "success" | "failed"`
- `runtime_seconds: float`
- `output_dir: str`
- `error_message: str | null`

## 4) Postprocessing Interface Specification

### Input
- simulation output file paths
- optional run metadata

### Output Schema (`SimulationSummary`)
- `travel_time_minutes: float | null`
- `route_descriptor: str | null`
- `peak_congestion_link: str | null`
- `peak_congestion_window: str | null`
- `avg_network_speed: float | null`
- `key_notes: list[str]`

### Notes
- Post-agent should consume this summary object, not full raw CSV tables.

## 5) Post-Agent Interface Specification

### Input
- `user_query: str`
- `SimulationSummary` (structured)

### Output
- concise user-facing explanation (text), for example:
  - expected travel time
  - most likely route/route descriptor
  - congestion hotspot and time window

## 6) Error Handling Conventions

- Missing required fields: fail fast with explicit validation messages.
- File I/O failures: return structured error metadata from module boundary.
- Unsupported region/mode in baseline: return clear `not_supported` reason.

## 7) Versioning Guidance

- If any schema field is renamed/removed, increment an interface version and update all dependent modules in the same PR.
- Backward-compatible field additions are allowed if defaults are defined.

