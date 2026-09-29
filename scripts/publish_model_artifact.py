"""Sign a new model artifact and append its hash to the transparency log.

Usage:
    python -m scripts.publish_model_artifact \\
        --model-name rf \\
        --model-dir ./models \\
        --private-key-path /secrets/signing_key.pem \\
        --db-url sqlite:///ledgerlens.db

Security requirements:
    - The signing private key must be stored in an HSM or encrypted secrets
      manager (AWS Secrets Manager, HashiCorp Vault, etc.).  Never commit the
      key to source control or store it on disk unencrypted in production.
    - The transparency log DB must be backed up separately from the model
      artifact store so a coordinated attack cannot tamper with both.
"""

import argparse
import sys


def publish(
    model_name: str,
    model_dir: str,
    private_key_path: str,
    db_url: str,
) -> str:
    """Sign *model_name* artifact, record in transparency log, return SHA-256.

    Delegates to ``detection.persistence.sign_and_register_artifact`` — the
    same signing routine ``detection.model_governance.promote_candidate``
    uses for automated promotions, so a manually-published artifact and an
    automatically-promoted one are indistinguishable to the trust chain.
    """
    from detection.persistence import (
        TransparencyLog,
        get_engine,
        get_session_factory,
        sign_and_register_artifact,
    )

    engine = get_engine(db_url)
    session_factory = get_session_factory(engine)
    log = TransparencyLog(session_factory)
    return sign_and_register_artifact(model_name, model_dir, private_key_path, log)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Model name (e.g. rf, xgb)")
    parser.add_argument(
        "--model-dir", default="./models", help="Directory containing model artifacts"
    )
    parser.add_argument(
        "--private-key-path", required=True, help="Path to Ed25519 private key (PEM)"
    )
    parser.add_argument("--db-url", default=None, help="SQLAlchemy DB URL (defaults to config)")
    args = parser.parse_args()

    if args.db_url is None:
        from config import config

        db_url = config.RISK_SCORE_DB_URL
    else:
        db_url = args.db_url

    try:
        sha = publish(
            model_name=args.model_name,
            model_dir=args.model_dir,
            private_key_path=args.private_key_path,
            db_url=db_url,
        )
        print(f"Published {args.model_name}: sha256={sha}")
        print(f"Transparency log updated in {db_url}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
