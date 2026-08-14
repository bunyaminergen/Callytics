"""Callytics CLI: ``python -m callytics <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from .bootstrap import create_all, ensure_user, seed_all
from .db.session import session_scope
from .settings import get_settings


def _cmd_seed(args: argparse.Namespace) -> int:
    """Seed stages, sources, field definitions and teams. Idempotent."""
    if args.create_tables:
        create_all()
    with session_scope() as session:
        counts = seed_all(session)
    print(json.dumps({"seeded": counts}, indent=2))
    return 0


def _cmd_add_user(args: argparse.Namespace) -> int:
    with session_scope() as session:
        user = ensure_user(session, args.name, role=args.role, email=args.email, key=args.key)
        print(json.dumps({"id": str(user.id), "key": user.key, "name": user.name, "role": user.role}))
    return 0


def _cmd_config(_: argparse.Namespace) -> int:
    """Print effective configuration, with secrets redacted."""
    settings = get_settings()
    redacted = {"api_token", "finecho_token", "finecho_webhook_secret", "lsq_access_key", "lsq_secret_key",
                "meta_access_token", "google_developer_token", "mysql_dsn"}
    out = {}
    for key, value in settings.model_dump().items():
        out[key] = ("<set>" if value else "<unset>") if key in redacted else value
    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed; install the 'api' extra", file=sys.stderr)
        return 1
    settings = get_settings()
    uvicorn.run(
        "callytics.api.app:app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="callytics", description="Callytics CRM")
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="seed vocabulary and teams (idempotent)")
    seed.add_argument(
        "--create-tables",
        action="store_true",
        help="create tables directly instead of using Alembic (dev only)",
    )
    seed.set_defaults(func=_cmd_seed)

    user = sub.add_parser("add-user", help="create or fetch a user")
    user.add_argument("name")
    user.add_argument("--role", default="counsellor", choices=["counsellor", "manager", "admin", "system"])
    user.add_argument("--email")
    user.add_argument("--key", help="folder-safe key shared with FinEcho recordings")
    user.set_defaults(func=_cmd_add_user)

    cfg = sub.add_parser("config", help="print effective configuration (secrets redacted)")
    cfg.set_defaults(func=_cmd_config)

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
