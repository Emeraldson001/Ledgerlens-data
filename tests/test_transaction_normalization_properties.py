"""Property-based tests for transaction normalization.

Two code paths normalize raw transaction data into the shared `Trade`
contract (`ingestion/data_models.py`):

  - `ingestion.avro_codec.trade_to_record` / `record_to_trade` — the Kafka
    wire-format round trip (asset_pair string encode/decode).
  - `ingestion.horizon_streamer._to_trade` — raw Horizon SSE record -> `Trade`.

`tests/fuzz/fuzz_avro_codec.py` already fuzzes the Avro *byte* deserializer
for crashes. These tests instead check semantic invariants over generated
`Trade`/record values: round-tripping shouldn't silently corrupt data, and
the Stellar "native asset" (`issuer=None`) convention must survive both
normalization paths.
"""

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from ingestion.avro_codec import record_to_trade, trade_to_record
from ingestion.data_models import Asset, Trade
from ingestion.horizon_streamer import _to_trade

# Stellar asset codes are 1-12 alphanumeric characters; issuers are account IDs.
# Exclude ":" and "/" since avro_codec's asset_pair format uses them as separators.
asset_codes = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=1, max_size=12
)
issuers = st.one_of(
    st.none(), st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", min_size=5, max_size=56)
)
account_ids = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", min_size=5, max_size=56)
amounts = st.decimals(
    min_value="0.0000001",
    max_value="1000000",
    places=7,
    allow_nan=False,
    allow_infinity=False,
)
close_times = st.datetimes(min_value=datetime(2015, 1, 1), max_value=datetime(2100, 1, 1)).map(
    lambda dt: dt.replace(tzinfo=UTC)
)


@st.composite
def assets(draw):
    return Asset(code=draw(asset_codes), issuer=draw(issuers))


@st.composite
def trades(draw):
    base_asset = draw(assets())
    counter_asset = draw(assets())
    return Trade(
        trade_id=draw(st.text(min_size=1, max_size=20)),
        ledger_close_time=draw(close_times),
        base_account=draw(account_ids),
        counter_account=draw(account_ids),
        base_asset=base_asset,
        counter_asset=counter_asset,
        base_amount=draw(amounts),
        counter_amount=draw(amounts),
        price=draw(amounts),
    )


@given(trades())
def test_avro_round_trip_preserves_trade_fields(trade: Trade):
    record = trade_to_record(trade)
    rebuilt = record_to_trade(record)

    assert rebuilt.trade_id == trade.trade_id
    assert rebuilt.base_account == trade.base_account
    assert rebuilt.counter_account == trade.counter_account
    assert rebuilt.base_amount == trade.base_amount
    assert rebuilt.counter_amount == trade.counter_amount
    assert rebuilt.price == trade.price
    assert rebuilt.ledger_close_time == trade.ledger_close_time


@given(trades())
def test_avro_round_trip_preserves_asset_identity(trade: Trade):
    """The asset_pair string encoding must not conflate a real issuer with 'native'."""
    record = trade_to_record(trade)
    rebuilt = record_to_trade(record)

    assert rebuilt.base_asset.code == trade.base_asset.code
    assert rebuilt.counter_asset.code == trade.counter_asset.code
    assert (rebuilt.base_asset.issuer is None) == (trade.base_asset.issuer is None)
    if trade.base_asset.issuer is not None:
        assert rebuilt.base_asset.issuer == trade.base_asset.issuer
    if trade.counter_asset.issuer is not None:
        assert rebuilt.counter_asset.issuer == trade.counter_asset.issuer


@given(trades())
def test_avro_round_trip_is_idempotent(trade: Trade):
    """Normalizing an already-normalized trade a second time is a no-op."""
    once = record_to_trade(trade_to_record(trade))
    twice = record_to_trade(trade_to_record(once))
    assert once == twice


def _raw_horizon_record(
    trade_id,
    close_time,
    base_account,
    counter_account,
    base,
    counter,
    base_amount,
    counter_amount,
    n,
    d,
):
    return {
        "id": trade_id,
        "ledger_close_time": close_time,
        "base_account": base_account,
        "counter_account": counter_account,
        "base_asset_code": base.code,
        "base_asset_issuer": base.issuer,
        "counter_asset_code": counter.code,
        "counter_asset_issuer": counter.issuer,
        "base_amount": base_amount,
        "counter_amount": counter_amount,
        "price": {"n": n, "d": d},
    }


@given(
    trade_id=st.text(min_size=1, max_size=20),
    close_time=close_times,
    base_account=account_ids,
    counter_account=account_ids,
    base=assets(),
    counter=assets(),
    base_amount=amounts,
    counter_amount=amounts,
    n=st.integers(min_value=1, max_value=10_000_000),
    d=st.integers(min_value=1, max_value=10_000_000),
)
def test_horizon_normalization_computes_price_from_fraction(
    trade_id,
    close_time,
    base_account,
    counter_account,
    base,
    counter,
    base_amount,
    counter_amount,
    n,
    d,
):
    record = _raw_horizon_record(
        trade_id,
        close_time,
        base_account,
        counter_account,
        base,
        counter,
        base_amount,
        counter_amount,
        n,
        d,
    )
    trade = _to_trade(record)

    expected_price = Decimal(str(float(n) / float(d))).quantize(Decimal("0.0000001"))
    assert trade.price == expected_price
    assert trade.base_amount == base_amount
    assert trade.counter_amount == counter_amount


@given(
    trade_id=st.text(min_size=1, max_size=20),
    close_time=close_times,
    base_account=account_ids,
    counter_account=account_ids,
    counter=assets(),
    base_amount=amounts,
    counter_amount=amounts,
)
def test_horizon_normalization_defaults_empty_base_code_to_xlm(
    trade_id, close_time, base_account, counter_account, counter, base_amount, counter_amount
):
    """A falsy base_asset_code (Horizon's representation of the native asset) normalizes to 'XLM'."""
    record = _raw_horizon_record(
        trade_id,
        close_time,
        base_account,
        counter_account,
        Asset(code="", issuer=None),
        counter,
        base_amount,
        counter_amount,
        1,
        1,
    )
    trade = _to_trade(record)
    assert trade.base_asset.code == "XLM"
