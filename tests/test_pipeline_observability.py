"""Tests for utils.pipeline_observability — structured observability for pipeline execution."""

import pytest

from utils.pipeline_observability import PipelineRun


def test_run_id_is_generated_and_stable():
    run = PipelineRun("test_pipeline")
    assert run.run_id
    assert run.run_id == run.run_id


def test_stage_records_success_outcome():
    run = PipelineRun("test_pipeline")
    with run.stage("load_data", pair_id="USDC:XLM"):
        pass

    assert len(run.stages) == 1
    record = run.stages[0]
    assert record["stage"] == "load_data"
    assert record["pair_id"] == "USDC:XLM"
    assert record["outcome"] == "ok"
    assert record["run_id"] == run.run_id
    assert record["duration_ms"] >= 0


def test_stage_records_failure_outcome_and_reraises():
    run = PipelineRun("test_pipeline")

    with pytest.raises(ValueError, match="boom"):
        with run.stage("score_wallets"):
            raise ValueError("boom")

    assert len(run.stages) == 1
    assert run.stages[0]["outcome"] == "failed"
    assert run.stages[0]["stage"] == "score_wallets"


def test_multiple_stages_accumulate_in_order():
    run = PipelineRun("test_pipeline")
    with run.stage("stage_one"):
        pass
    with run.stage("stage_two"):
        pass

    assert [s["stage"] for s in run.stages] == ["stage_one", "stage_two"]


def test_summary_reports_failed_stages():
    run = PipelineRun("test_pipeline")
    with run.stage("ok_stage"):
        pass
    try:
        with run.stage("bad_stage"):
            raise RuntimeError("nope")
    except RuntimeError:
        pass

    summary = run.summary()
    assert summary["run_id"] == run.run_id
    assert summary["stage_count"] == 2
    assert summary["failed_stages"] == ["bad_stage"]


def test_summary_with_no_stages_is_empty_but_valid():
    run = PipelineRun("test_pipeline")
    summary = run.summary()
    assert summary["stage_count"] == 0
    assert summary["failed_stages"] == []
    assert summary["stages"] == []
