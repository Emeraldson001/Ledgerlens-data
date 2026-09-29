"""Tests for run_pipeline.py's --checkpoint-file / --fresh resume semantics."""

import json
import logging
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import run_pipeline
from utils.checkpoint import CheckpointMismatchError

USDC_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
BTC_ISSUER = "GBVOL67TMUQBGL4TZYNMY3ZQ5WGQYFPFD5VJRWXR72VA33VFNL225PL5"
USDC_PAIR_ID = f"USDC:{USDC_ISSUER}/XLM:native"
BTC_PAIR_ID = f"BTC:{BTC_ISSUER}/XLM:native"

TS = pd.Timestamp("2024-01-01", tz="UTC")


def _trades(w1, w2):
    return pd.DataFrame(
        {
            "base_account": [w1],
            "counter_account": [w2],
            "ledger_close_time": [TS],
            "amount": [100.0],
        }
    )


def _scored(wallets, score=10):
    return pd.DataFrame(
        {
            "wallet": wallets,
            "score": [score] * len(wallets),
            "benford_flag": [False] * len(wallets),
            "ml_flag": [False] * len(wallets),
            "confidence": [50] * len(wallets),
        }
    )


def _fake_build_feature_matrix(trades_df, **kwargs):
    wallets = list(pd.unique(trades_df[["base_account", "counter_account"]].values.ravel()))
    return pd.DataFrame({"wallet": wallets, "benford_mad_1h": [0.0] * len(wallets)})


@pytest.fixture
def two_pairs():
    return [("USDC", USDC_ISSUER), ("BTC", BTC_ISSUER)]


def _run_pipeline(argv, load_side_effect, two_pairs, scorer=None):
    """Run run_pipeline.main() with ingestion/scoring stubbed out."""
    if scorer is None:
        scorer = MagicMock()
        scorer.score_matrix.side_effect = lambda fm: _scored(list(fm["wallet"]))

    with ExitStack() as stack:
        stack.enter_context(patch("sys.argv", ["run_pipeline.py", *argv]))
        stack.enter_context(
            patch.object(run_pipeline, "load_pair_to_dataframe", side_effect=load_side_effect)
        )
        stack.enter_context(
            patch.object(
                run_pipeline, "build_feature_matrix", side_effect=_fake_build_feature_matrix
            )
        )
        stack.enter_context(patch("detection.model_inference.RiskScorer", return_value=scorer))
        stack.enter_context(patch.object(run_pipeline.config, "WATCHED_ASSET_PAIRS", two_pairs))
        run_pipeline.main()


def test_resume_skips_pairs_already_completed(tmp_path, two_pairs):
    ckpt_file = tmp_path / "ckpt.json"
    load_calls = []

    def fake_load(asset, xlm, start_time=None):
        load_calls.append(asset.code)
        wallets = ("GA", "GB") if asset.code == "USDC" else ("GC", "GD")
        return _trades(*wallets)

    argv = [
        "--no-orderbook",
        "--no-graph",
        "--no-persist",
        "--checkpoint-file",
        str(ckpt_file),
    ]

    _run_pipeline(argv, fake_load, two_pairs)

    assert load_calls == ["USDC", "BTC"]
    data = json.loads(ckpt_file.read_text())
    assert set(data["completed"]) == {USDC_PAIR_ID, BTC_PAIR_ID}

    # Second invocation: both pairs already done — load_pair_to_dataframe must
    # not be called again for either pair.
    load_calls.clear()
    _run_pipeline(argv, fake_load, two_pairs)
    assert load_calls == []


def test_failed_pair_is_retried_and_succeeding_pair_is_not_reprocessed(tmp_path, two_pairs):
    ckpt_file = tmp_path / "ckpt.json"
    attempt = {"BTC": 0}
    load_call_order = []

    def flaky_load(asset, xlm, start_time=None):
        load_call_order.append(asset.code)
        if asset.code == "USDC":
            return _trades("GA", "GB")
        attempt["BTC"] += 1
        if attempt["BTC"] == 1:
            raise ConnectionError("horizon unreachable")
        return _trades("GC", "GD")

    argv = [
        "--no-orderbook",
        "--no-graph",
        "--no-persist",
        "--checkpoint-file",
        str(ckpt_file),
    ]

    # First run: USDC succeeds, BTC fails. main() must not raise.
    _run_pipeline(argv, flaky_load, two_pairs)

    data = json.loads(ckpt_file.read_text())
    assert list(data["completed"]) == [USDC_PAIR_ID]
    assert list(data["failed"]) == [BTC_PAIR_ID]

    # Second run (resume is the default when --checkpoint-file exists): USDC is
    # skipped entirely, BTC is retried and succeeds this time.
    load_call_order.clear()
    _run_pipeline(argv, flaky_load, two_pairs)

    assert load_call_order == ["BTC"]
    data = json.loads(ckpt_file.read_text())
    assert set(data["completed"]) == {USDC_PAIR_ID, BTC_PAIR_ID}
    assert data["failed"] == {}


def test_mismatched_since_argument_raises_actionable_error(tmp_path, two_pairs):
    ckpt_file = tmp_path / "ckpt.json"

    _run_pipeline(
        [
            "--no-orderbook",
            "--no-graph",
            "--no-persist",
            "--checkpoint-file",
            str(ckpt_file),
            "--since",
            "2024-01-01",
        ],
        lambda asset, xlm, start_time=None: _trades("GA", "GB"),
        two_pairs,
    )

    with pytest.raises(CheckpointMismatchError, match="since"):
        _run_pipeline(
            [
                "--no-orderbook",
                "--no-graph",
                "--no-persist",
                "--checkpoint-file",
                str(ckpt_file),
                "--since",
                "2024-06-01",
            ],
            lambda asset, xlm, start_time=None: _trades("GA", "GB"),
            two_pairs,
        )


def test_fresh_flag_ignores_prior_checkpoint(tmp_path, two_pairs):
    ckpt_file = tmp_path / "ckpt.json"
    load_calls = []

    def fake_load(asset, xlm, start_time=None):
        load_calls.append(asset.code)
        wallets = ("GA", "GB") if asset.code == "USDC" else ("GC", "GD")
        return _trades(*wallets)

    base_argv = [
        "--no-orderbook",
        "--no-graph",
        "--no-persist",
        "--checkpoint-file",
        str(ckpt_file),
    ]

    _run_pipeline(base_argv, fake_load, two_pairs)
    assert load_calls == ["USDC", "BTC"]

    load_calls.clear()
    _run_pipeline([*base_argv, "--fresh"], fake_load, two_pairs)
    assert load_calls == ["USDC", "BTC"]  # both reprocessed after --fresh


def test_no_checkpoint_file_preserves_prior_all_or_nothing_behavior(tmp_path, two_pairs):
    """Without --checkpoint-file, a pair failure must still propagate and abort main()."""

    def failing_load(asset, xlm, start_time=None):
        if asset.code == "USDC":
            return _trades("GA", "GB")
        raise ConnectionError("horizon unreachable")

    with pytest.raises(ConnectionError):
        _run_pipeline(["--no-orderbook", "--no-graph", "--no-persist"], failing_load, two_pairs)


def test_dry_run_ignores_checkpoint_file(tmp_path, two_pairs, caplog):
    ckpt_file = tmp_path / "ckpt.json"

    with caplog.at_level(logging.WARNING):
        _run_pipeline(
            ["--no-orderbook", "--no-graph", "--dry-run", "--checkpoint-file", str(ckpt_file)],
            lambda asset, xlm, start_time=None: _trades("GA", "GB"),
            two_pairs[:1],
        )

    assert not ckpt_file.exists()


def test_since_with_invalid_date_fails_at_arg_parsing(capsys):
    """An invalid --since value must be rejected by argparse itself, with a clear
    message, before any pipeline stage (e.g. a network call) runs."""
    with patch("sys.argv", ["run_pipeline.py", "--since", "not-a-date"]):
        with pytest.raises(SystemExit) as exc_info:
            run_pipeline.parse_args()

    assert exc_info.value.code == 2
    assert "invalid --since value" in capsys.readouterr().err
    assert "--checkpoint-file has no effect with --dry-run" in caplog.text
