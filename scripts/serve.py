#!/usr/bin/env python3
"""Launch the FastAPI server for the firewall."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from src.utils.io import load_multi_config
from src.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    config = load_multi_config("configs")
    serve_cfg = config.get("deployment", {}).get("serving", {})

    host = serve_cfg.get("host", "0.0.0.0")
    port = serve_cfg.get("port", 8000)

    from src.deployment.serving import create_app
    app = create_app(config)

    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
