"""Allow ``python -m mctracker --config ...``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ConfigError
from .logging_utils import configure_logging
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Multi-camera person tracking.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--log-format",
        default="json",
        choices=["json", "text"],
    )
    p.add_argument(
        "--metrics-port",
        type=int,
        default=0,
        help="If >0, expose Prometheus /metrics on this TCP port "
             "(requires the [prometheus] extra).",
    )
    args = p.parse_args(argv)

    if args.log_format == "json":
        configure_logging(level=args.log_level)
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    log = logging.getLogger("mctracker.cli")
    try:
        pipeline = Pipeline(args.config, metrics_port=args.metrics_port)
        pipeline.build()
    except ConfigError as e:
        log.error("config error", extra={"error": str(e)})
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        log.exception("failed to start pipeline")
        print(f"failed to start pipeline: {e}", file=sys.stderr)
        return 1
    log.info("pipeline ready", extra={"streams": [s.id for s in pipeline.streams]})
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())