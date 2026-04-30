from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    uvicorn.run(
        "ai_dlsim.api.dashboard_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()

