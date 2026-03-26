from __future__ import annotations

import argparse
from pathlib import Path

from ai_dlsim.postprocessing.llm_result_interpreter import LlmResultInterpreter
from ai_dlsim.preprocessing.llm_query_parser import LlmQueryParser
from ai_dlsim.schemas.scenario import ScenarioPaths
from ai_dlsim.simulation.simulation_service import SimulationService, ensure_parent_dir


def build_default_ithaca_paths(repo_root: Path) -> ScenarioPaths:
    return ScenarioPaths(
        raw_osm_path=repo_root / "data" / "Ithaca" / "14850_roads.osm",
        node_csv_path=repo_root / "data" / "Ithaca" / "node.csv",
        link_csv_path=repo_root / "data" / "Ithaca" / "link.csv",
        demand_csv_path=repo_root / "data" / "Ithaca" / "demand.csv",
        output_dir=repo_root / "outputs" / "runs" / "ithaca_baseline",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-driven traffic simulation baseline.")
    parser.add_argument("--query", type=str, required=True, help="Natural language query.")
    parser.add_argument(
        "--llm-model",
        type=str,
        default="openai.gpt-5-mini",
        help="Pre/post-processing model name (wired: openai.gpt-5-mini).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]

    pre = LlmQueryParser(model=args.llm_model)
    interpreter = LlmResultInterpreter(model=args.llm_model)

    request = pre.parse(args.query)
    print("Parsed request:", request)

    if request.region.strip().lower() not in {"ithaca, ny", "ithaca"}:
        raise ValueError(
            f"Baseline currently supports Ithaca only; got region='{request.region}'. "
            "Implement Phase 3 region selection next."
        )

    paths = build_default_ithaca_paths(repo_root)
    ensure_parent_dir(paths.output_dir)

    # Baseline simulation adapter is stubbed (Phase 4 later).
    dlsim_result = SimulationService().run(paths)

    final_text = interpreter.interpret(user_query=args.query, dlsim_result=dlsim_result)
    print(final_text)


if __name__ == "__main__":
    main()

