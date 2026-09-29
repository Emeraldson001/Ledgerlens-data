from datetime import UTC, datetime, timedelta
import logging

from ingestion.watermark_tracker import WatermarkTracker


def test_watermark_does_not_move_backwards_for_out_of_order_trade(tmp_path, caplog):
    """Older timestamps should not regress the stored watermark, but are still logged."""
    tracker = WatermarkTracker(store_path=str(tmp_path / "watermarks.json"))

    first_time = datetime(2024, 1, 1, tzinfo=UTC)
    second_time = first_time + timedelta(minutes=5)
    older_time = first_time + timedelta(minutes=3)

    tracker.advance("USDC:XLM", "token-1", first_time)
    tracker.advance("USDC:XLM", "token-2", second_time)

    with caplog.at_level(logging.WARNING):
        tracker.advance("USDC:XLM", "token-3", older_time)

    wm = tracker.get("USDC:XLM")
    assert wm is not None
    assert wm.ledger_close_time == second_time
    assert wm.paging_token == "token-2"
    assert wm.trade_count == 2
    assert "out-of-order" in caplog.text.lower()
