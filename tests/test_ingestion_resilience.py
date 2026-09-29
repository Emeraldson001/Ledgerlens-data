"""Resilience tests for the untrusted-input validation contract (see
`ingestion/untrusted_input.py`).

A single malformed record from Horizon/AMM (bad account ID, NaN amount,
zero-denominator price fraction, etc.) must be logged and skipped -- it
must never crash the rest of the page or the whole stream.
"""

from unittest.mock import MagicMock, patch

from stellar_sdk import Asset as SdkAsset

from ingestion.account_activity_loader import load_accounts_activity
from ingestion.amm_pool_loader import load_amm_pool_trades
from ingestion.historical_loader import load_trades
from ingestion.orderbook_loader import load_orderbook_events
from ingestion.untrusted_input import UntrustedInputError

VALID_BASE = "GCGPQMCLRXCUPCL3AVMYUUQML2WVC7A5M6HO5RKYSU4CIA7O7SI4VKWE"
VALID_COUNTER = "GB2HHLFDCBSBDAMU2QRDU4AJV63WQE2DWT7MBZWZRQDFYUXJXIPPUG7M"
VALID_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"


def _good_trade_record(trade_id="good-1"):
    return {
        "id": trade_id,
        "paging_token": trade_id,
        "ledger_close_time": "2024-01-01T00:00:00Z",
        "base_account": VALID_BASE,
        "counter_account": VALID_COUNTER,
        "base_asset_code": "USDC",
        "base_asset_issuer": VALID_ISSUER,
        "counter_asset_code": "",
        "counter_asset_issuer": None,
        "base_amount": "100.0",
        "counter_amount": "50.0",
        "price": {"n": 1, "d": 2},
    }


def test_load_trades_skips_malformed_record_without_crashing():
    """A page with one poisoned record (bad account ID) still yields the
    good trades either side of it instead of raising or dropping the page."""
    bad_record = _good_trade_record("bad-1")
    bad_record["base_account"] = "NOT-A-REAL-ACCOUNT"

    zero_price_record = _good_trade_record("bad-2")
    zero_price_record["price"] = {"n": 1, "d": 0}  # would ZeroDivisionError pre-fix

    records = [
        _good_trade_record("good-1"),
        bad_record,
        zero_price_record,
        _good_trade_record("good-2"),
    ]

    with patch("ingestion.historical_loader._fetch_page") as mock_fetch:
        mock_fetch.side_effect = [
            {"_embedded": {"records": records}, "_links": {"next": {"href": ""}}},
        ]
        trades = list(load_trades(SdkAsset("USDC", VALID_ISSUER), SdkAsset.native()))

    assert [t.trade_id for t in trades] == ["good-1", "good-2"]


def test_load_orderbook_events_skips_malformed_record():
    good = {
        "id": "op-good",
        "type": "manage_sell_offer",
        "source_account": VALID_BASE,
        "created_at": "2024-01-01T00:00:00Z",
        "selling_asset_type": "native",
        "buying_asset_type": "credit_alphanum4",
        "buying_asset_code": "USDC",
        "buying_asset_issuer": VALID_ISSUER,
        "amount": "100.0",
        "offer_id": "0",
        "price": "0.5",
    }
    bad = dict(good, id="op-bad", source_account="short-and-invalid")

    with patch("ingestion.orderbook_loader._fetch_page") as mock_fetch:
        mock_fetch.side_effect = [
            {
                "_embedded": {"records": [good, bad]},
                "_links": {"next": {"href": ""}},
            }
        ]
        events = list(load_orderbook_events(VALID_BASE))

    assert [e.event_id for e in events] == ["op-good"]


def test_load_amm_pool_trades_skips_malformed_record():
    pool_id = "a" * 64
    good = _good_trade_record("amm-good")
    bad = _good_trade_record("amm-bad")
    bad["base_amount"] = "nan"

    page = {
        "_embedded": {"records": [good, bad]},
        "_links": {"next": {"href": ""}},
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = page
    mock_resp.raise_for_status = MagicMock()

    import datetime

    since = datetime.datetime(2023, 1, 1, tzinfo=datetime.UTC)
    until = datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)

    with patch("requests.Session.get", return_value=mock_resp):
        df = load_amm_pool_trades(pool_id, since, until)

    assert list(df["trade_id"]) == ["amm-good"]


def test_load_accounts_activity_tolerates_untrusted_input_error():
    """Batch loading must skip a wallet whose record fails validation, same
    as it already tolerates network errors."""
    good_activity = MagicMock(account_id=VALID_BASE)

    with patch("ingestion.account_activity_loader.load_account_activity") as mock_load:
        mock_load.side_effect = [
            good_activity,
            UntrustedInputError("account_id", "not a valid account", source="test"),
        ]
        results = load_accounts_activity([VALID_BASE, "bogus-wallet"])

    assert results == [good_activity]


def test_horizon_endpoint_pool_probe_logs_network_error():
    """Health check probe must log network errors rather than crash,
    allowing pool to mark endpoint unhealthy and try another."""
    from ingestion.horizon_streamer import HorizonEndpointPool

    pool = HorizonEndpointPool(urls=["https://horizon.stellar.org"])

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = OSError("Connection refused")
        with patch("ingestion.horizon_streamer.logger") as mock_logger:
            healthy, latency = pool._probe("https://horizon.stellar.org")

    assert healthy is False
    assert latency > 0
    mock_logger.debug.assert_called()
    # Verify exception message is logged
    call_args = mock_logger.debug.call_args[0]
    assert "Connection refused" in str(call_args)


def test_rate_limiter_degradation_on_redis_unavailable():
    """Rate limiter must degrade gracefully when Redis unavailable,
    not crash ingestion."""
    from ingestion.rate_limiter import TokenBucketLimiter

    limiter = TokenBucketLimiter(redis_url="redis://invalid:9999")
    # Should not raise even though Redis is unreachable
    assert limiter._client is None
    assert limiter._warned is True

    # try_acquire should return True (grant all) when Redis unavailable
    result = limiter.try_acquire()
    assert result is True


def test_kafka_producer_serialization_failure_logs_and_routes_to_dlq():
    """Serialization failure must be logged and routed to DLQ, not crash."""
    from ingestion.data_models import Asset, Trade
    from ingestion.kafka_producer import HorizonKafkaProducer

    producer = HorizonKafkaProducer()
    # Create a trade with invalid data that will fail serialization
    bad_trade = Trade(
        trade_id="test",
        ledger_close_time="not-a-datetime",  # type: ignore
        base_account=VALID_BASE,
        counter_account=VALID_COUNTER,
        base_asset=Asset(code="USDC"),
        counter_asset=Asset(code="XLM"),
        base_amount=100.0,
        counter_amount=50.0,
        price=0.5,
    )

    with patch("ingestion.kafka_producer.logger") as mock_logger:
        with patch.object(producer, "_produce_to_dlq") as mock_dlq:
            # This should log and route to DLQ, not raise
            producer.produce_trade(bad_trade)

            # Verify DLQ was called
            mock_dlq.assert_called()
            # Verify error was logged
            assert mock_logger.error.called
