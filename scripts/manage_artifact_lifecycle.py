"""CLI for the versioned model artifact lifecycle registry.

Complements ``scripts/publish_model_artifact.py`` (signing + transparency
log) and ``scripts/list_model_versions.py`` (transparency-log listing) with
operational lifecycle commands backed by ``detection.artifact_lifecycle``.

``promote`` and ``rollback`` are authenticated, audited actions (Grand 2 /
issue #671): both require ``--actor`` and ``--credential`` (or the
``MODEL_PROMOTION_ACTOR``/``MODEL_PROMOTION_CREDENTIAL`` environment
variables), checked against ``MODEL_PROMOTION_AUTHORIZED_ACTORS`` /
``MODEL_PROMOTION_SECRET`` via ``detection.model_governance.authorize_actor``.
Every attempt — authorized or not — is written to the ``promotion_audit_log``
table. ``promote`` additionally requires the target artifact to pass the full
Ed25519 + transparency-log trust chain before ``PROMOTED`` is reached; a
failure raises rather than silently promoting.

Usage:
    python -m scripts.manage_artifact_lifecycle register --name rf --artifact-path models/rf.joblib
    python -m scripts.manage_artifact_lifecycle validate --name rf --version <version>
    python -m scripts.manage_artifact_lifecycle promote --name rf --version <version> --actor alice --credential <hmac>
    python -m scripts.manage_artifact_lifecycle rollback --name rf --reason "AUC regression" --actor alice --credential <hmac>
    python -m scripts.manage_artifact_lifecycle status --name rf
    python -m scripts.manage_artifact_lifecycle verify --name rf --version <version>
"""

import argparse
import json
import os
import sys

from detection.artifact_lifecycle import ArtifactLifecycleError, ModelArtifactRegistry
from detection.model_governance import (
    PromotionError,
    UnauthorizedPromotionError,
    authorize_actor,
    make_trust_verifier,
)
from detection.persistence import PromotionAuditLog, get_engine, get_session_factory


def _authorize_from_args(args: argparse.Namespace, model_name: str, action: str) -> str:
    """Authorize --actor/--credential (or the env-var fallback), audit the
    attempt regardless of outcome, and return the authorized actor name."""
    actor = args.actor or os.environ.get("MODEL_PROMOTION_ACTOR", "")
    credential = args.credential or os.environ.get("MODEL_PROMOTION_CREDENTIAL", "")
    audit_log = PromotionAuditLog(get_session_factory(get_engine()))
    try:
        authorize_actor(actor, credential)
    except UnauthorizedPromotionError as exc:
        audit_log.record(
            actor=actor or "(none)",
            action=action,
            model_name=model_name,
            success=False,
            reason=str(exc),
        )
        raise
    return actor


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser (split from main() so
    tests/test_docs_cli_consistency.py can introspect the real per-subcommand
    flag set without invoking the CLI)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default="models/artifact_manifest.json")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Preview planned lifecycle actions without modifying the manifest (register, validate, promote, deprecate, rollback).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_register = sub.add_parser("register")
    p_register.add_argument("--name", required=True)
    p_register.add_argument("--artifact-path", required=True)
    p_register.add_argument("--metrics-json", default=None, help="JSON string of metrics")

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("--name", required=True)
    p_validate.add_argument("--version", required=True)

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("--name", required=True)
    p_promote.add_argument("--version", required=True)
    p_promote.add_argument("--actor", default=None, help="Falls back to $MODEL_PROMOTION_ACTOR")
    p_promote.add_argument(
        "--credential", default=None, help="Falls back to $MODEL_PROMOTION_CREDENTIAL"
    )

    p_deprecate = sub.add_parser("deprecate")
    p_deprecate.add_argument("--name", required=True)
    p_deprecate.add_argument("--version", required=True)
    p_deprecate.add_argument("--reason", default=None)

    p_rollback = sub.add_parser("rollback")
    p_rollback.add_argument("--name", required=True)
    p_rollback.add_argument("--version", default=None)
    p_rollback.add_argument("--reason", default=None)
    p_rollback.add_argument("--actor", default=None, help="Falls back to $MODEL_PROMOTION_ACTOR")
    p_rollback.add_argument(
        "--credential", default=None, help="Falls back to $MODEL_PROMOTION_CREDENTIAL"
    )

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--name", required=True)
    p_verify.add_argument("--version", required=True)

    p_status = sub.add_parser("status")
    p_status.add_argument("--name", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # `promote` requires a trust_verifier configured — build one bound to the
    # directory the target artifact actually lives in, resolved from the
    # manifest once it's loaded below.
    registry = ModelArtifactRegistry(manifest_path=args.manifest_path)

    try:
        if args.command == "register":
            if args.check_only:
                print(f"[CHECK-ONLY] Would register {args.name} from {args.artifact_path} (stage=staged)")
            else:
                metrics = json.loads(args.metrics_json) if args.metrics_json else {}
                version = registry.register(args.name, args.artifact_path, metrics=metrics)
                print(f"Registered {args.name}:{version} (stage=staged)")
        elif args.command == "validate":
            if args.check_only:
                print(f"[CHECK-ONLY] Would validate {args.name}:{args.version}")
            else:
                registry.validate(args.name, args.version)
                print(f"{args.name}:{args.version} -> validated")
        elif args.command == "promote":
            actor = _authorize_from_args(args, args.name, "promote")
            audit_log = PromotionAuditLog(get_session_factory(get_engine()))
            record = registry._get(args.name, args.version)
            model_dir = os.path.dirname(os.path.abspath(record.artifact_path)) or "."
            registry._trust_verifier = make_trust_verifier(model_dir)
            try:
                registry.promote(args.name, args.version)
            except Exception as exc:
                audit_log.record(
                    actor=actor,
                    action="promote",
                    model_name=args.name,
                    version_id=args.version,
                    success=False,
                    reason=str(exc),
                )
                raise
            audit_log.record(
                actor=actor,
                action="promote",
                model_name=args.name,
                version_id=args.version,
                success=True,
            )
            print(f"{args.name}:{args.version} -> promoted (active); actor={actor}")
        elif args.command == "deprecate":
            if args.check_only:
                reason_str = f" (reason: {args.reason})" if args.reason else ""
                print(f"[CHECK-ONLY] Would deprecate {args.name}:{args.version}{reason_str}")
            else:
                registry.deprecate(args.name, args.version, reason=args.reason)
                print(f"{args.name}:{args.version} -> deprecated")
        elif args.command == "rollback":
            actor = _authorize_from_args(args, args.name, "rollback")
            audit_log = PromotionAuditLog(get_session_factory(get_engine()))
            try:
                record = registry.rollback(args.name, args.version, reason=args.reason)
            except Exception as exc:
                audit_log.record(
                    actor=actor,
                    action="rollback",
                    model_name=args.name,
                    success=False,
                    reason=str(exc),
                )
                raise
            audit_log.record(
                actor=actor,
                action="rollback",
                model_name=args.name,
                version_id=record.version,
                success=True,
                reason=args.reason,
            )
            print(
                f"Rolled back {args.name}:{record.version}; "
                f"reactivated parent={record.parent_version}; actor={actor}"
            )
        elif args.command == "verify":
            registry.verify_integrity(args.name, args.version)
            print(f"{args.name}:{args.version} integrity OK")
        elif args.command == "status":
            versions = registry.list_versions(args.name)
            if not versions:
                print(f"No versions registered for {args.name}")
            for record in versions:
                marker = " <- active" if record.stage.value == "promoted" else ""
                print(f"  {record.version}  stage={record.stage.value}{marker}")
    except (ArtifactLifecycleError, PromotionError, UnauthorizedPromotionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
