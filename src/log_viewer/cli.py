"""Command-line entry point for the offline log viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

from log_viewer.data import discover_sessions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect completed Vex SDK log chunks in a local browser")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing session folders (default: logs)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="HTTP port (default: 8050)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.log_dir.exists():
        parser.error(f"log directory does not exist: {args.log_dir}")
    if not args.log_dir.is_dir():
        parser.error(f"log path is not a directory: {args.log_dir}")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")

    try:
        from log_viewer.app import create_app
    except ModuleNotFoundError as exc:
        if exc.name in {"dash", "plotly"}:
            parser.error("viewer dependencies are missing; run with: uv run --extra viewer vex-log-viewer")
        raise

    catalog = discover_sessions(args.log_dir)
    app = create_app(catalog)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
