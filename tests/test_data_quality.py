import json

from utils.data_quality import (
    DataQualityValidator,
    RangeRule,
    RegexRule,
    RequiredFieldRule,
    TypeRule,
)


def test_valid_record_passes_all_rules():
    validator = DataQualityValidator(
        [
            RequiredFieldRule("wallet"),
            RequiredFieldRule("score"),
            TypeRule("score", (int, float)),
            RangeRule("score", minimum=0, maximum=100),
        ]
    )
    report = validator.validate({"wallet": "GABCDEF", "score": 42})
    assert report.passed
    assert report.issues == []


def test_missing_required_field_reported():
    validator = DataQualityValidator([RequiredFieldRule("wallet")])
    report = validator.validate({"score": 42})
    assert not report.passed
    assert report.issues[0].rule_name == "required:wallet"
    assert report.issues[0].field == "wallet"


def test_null_required_field_reported():
    validator = DataQualityValidator([RequiredFieldRule("wallet")])
    report = validator.validate({"wallet": None})
    assert not report.passed


def test_type_rule_rejects_wrong_type():
    validator = DataQualityValidator([TypeRule("score", (int, float))])
    report = validator.validate({"score": "not-a-number"})
    assert not report.passed
    assert "type:score" in report.issues[0].rule_name


def test_type_rule_rejects_bool_for_numeric():
    validator = DataQualityValidator([TypeRule("score", (int, float))])
    report = validator.validate({"score": True})
    assert not report.passed


def test_range_rule_flags_out_of_bounds():
    validator = DataQualityValidator([RangeRule("score", minimum=0, maximum=100)])
    below = validator.validate({"score": -5})
    above = validator.validate({"score": 150})
    within = validator.validate({"score": 50})
    assert not below.passed
    assert not above.passed
    assert within.passed


def test_regex_rule_validates_pattern():
    validator = DataQualityValidator([RegexRule("wallet", r"^G[A-Z0-9]{5,}$")])
    ok = validator.validate({"wallet": "GABCDEF123"})
    bad = validator.validate({"wallet": "not-an-address"})
    assert ok.passed
    assert not bad.passed


def test_absent_field_does_not_trigger_type_or_range_rules():
    # RequiredFieldRule owns "missing" diagnostics; type/range rules should
    # be silent on absence to avoid duplicate/confusing reports.
    validator = DataQualityValidator([TypeRule("score", int), RangeRule("score", minimum=0)])
    report = validator.validate({})
    assert report.passed


def test_validate_batch_aggregates_with_record_index():
    validator = DataQualityValidator([RequiredFieldRule("wallet")])
    records = [{"wallet": "G1"}, {}, {"wallet": "G3"}, {}]
    report = validator.validate_batch(records)
    assert report.records_checked == 4
    assert len(report.issues) == 2
    assert [i.record_index for i in report.issues] == [1, 3]


def test_validate_batch_fail_fast_stops_early():
    validator = DataQualityValidator([RequiredFieldRule("wallet")])
    records = [{}, {"wallet": "G1"}, {}]
    report = validator.validate_batch(records, fail_fast=True)
    assert report.records_checked == 1
    assert len(report.issues) == 1


def test_range_rule_from_feature_ranges_file(tmp_path):
    ranges_path = tmp_path / "feature_ranges.json"
    ranges_path.write_text(
        json.dumps({"velocity": {"min": 0, "max": 1000}, "not_a_range": "ignored"})
    )
    rules = RangeRule.from_feature_ranges(str(ranges_path))
    assert len(rules) == 1
    assert rules[0].field == "velocity"
    assert rules[0].minimum == 0
    assert rules[0].maximum == 1000


def test_range_rule_from_feature_ranges_missing_file_returns_empty():
    assert RangeRule.from_feature_ranges("/nonexistent/path/feature_ranges.json") == []


def test_report_as_dict_shape():
    validator = DataQualityValidator([RequiredFieldRule("wallet")])
    report = validator.validate({})
    d = report.as_dict()
    assert d["passed"] is False
    assert d["issue_count"] == 1
    assert d["issues"][0]["rule_name"] == "required:wallet"
