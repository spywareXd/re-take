"""python -m service  [--host 127.0.0.1] [--port 8765]  (loads .env for LLM keys; never prints them)"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from .app import create_app
from .config import ServiceConfig, load_dotenv
from .refine import refine_available

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    load_dotenv(REPO / ".env")
    cfg = ServiceConfig.from_env()
    ap = argparse.ArgumentParser(description="Local retake service for SimpleCut")
    ap.add_argument("--host", default=cfg.host)
    ap.add_argument("--port", type=int, default=cfg.port)
    args = ap.parse_args()
    cfg.host, cfg.port = args.host, args.port
    app = create_app(cfg)
    from models.final.final_scorer import FinalScorer
    thr = FinalScorer(mock=True)                    # thresholds only: no weights are loaded here
    logging.getLogger("service").info("thresholds version %s (unsure band %s..%s) | adjudicate: %s (%s) | refine: %s | listening on http://%s:%d",
                                      thr.thresholds_version, thr.t_lo, thr.t_hi, app.state.llm.available, app.state.llm.provider, refine_available(), cfg.host, cfg.port)
    from pipeline.step_06_llm_adjudication.keeper_policy import KeeperConfig
    from pipeline.step_06_llm_adjudication.keeper_providers import FinetunedProvider
    kcfg = KeeperConfig.from_env(provider=cfg.keeper_provider)
    ok, why = FinetunedProvider(kcfg).availability()
    logging.getLogger("service").info("keeper default '%s' | finetuned available: %s (%s)", kcfg.provider, ok, why)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
