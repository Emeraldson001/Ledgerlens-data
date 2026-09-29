"""Tests for `config.settings_validator` — the runtime settings validation service."""

from __future__ import annotations

import pytest

from config.settings_validator import (
    DEFAULT_LEDGERLENS_SPECS,
    SettingSpec,
    SettingsValidationError,
    SettingsValidator,
    is_non_empty,
    is_url,
)


class TestPredicates:
    def test_is_url_accepts_valid_url(self):
        assert is_url("https://horizon.stellar.org") is True

    def test_is_url_rejects_missing_scheme(self):
        assert is_url("horizon.stellar.org") is False

    def test_is_url_rejects_non_string(self):
        assert is_url(123) is False

    def test_is_non_empty(self):
        assert is_non_empty("x") is True
        assert is_non_empty("") is False
        assert is_non_empty(None) is False
        assert is_non_empty([1]) is True
        assert is_non_empty([]) is False


class TestSettingSpec:
    def test_required_missing_is_error(self):
        spec = SettingSpec("FOO", str, required=True)
        issues = spec.check(None, present=False)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "missing" in issues[0].message

    def test_optional_missing_produces_no_issue(self):
        spec = SettingSpec("FOO", str, required=False)
        assert spec.check(None, present=False) == []

    def test_wrong_type_is_error(self):
        spec = SettingSpec("FOO", int, required=True)
        issues = spec.check("not-an-int", present=True)
        assert any("expected type" in i.message for i in issues)

    def test_bool_not_accepted_as_int(self):
        spec = SettingSpec("FOO", int, required=True)
        issues = spec.check(True, present=True)
        assert len(issues) == 1

    def test_min_max_bounds(self):
        spec = SettingSpec("FOO", float, min_value=0.0, max_value=1.0)
        assert spec.check(0.5, present=True) == []
        assert len(spec.check(-0.1, present=True)) == 1
        assert len(spec.check(1.1, present=True)) == 1

    def test_allowed_values(self):
        spec = SettingSpec("FOO", str, allowed_values=("A", "B"))
        assert spec.check("A", present=True) == []
        assert len(spec.check("C", present=True)) == 1

    def test_custom_validator_failure(self):
        spec = SettingSpec("FOO", str, validator=lambda v: v.startswith("x"))
        assert spec.check("xyz", present=True) == []
        assert len(spec.check("abc", present=True)) == 1

    def test_validator_exception_is_captured_as_issue(self):
        def bad_validator(v):
            raise RuntimeError("boom")

        spec = SettingSpec("FOO", str, validator=bad_validator)
        issues = spec.check("abc", present=True)
        assert len(issues) == 1
        assert "boom" in issues[0].message


class TestSettingsValidator:
    def test_valid_source_produces_ok_report(self):
        specs = [SettingSpec("PORT", int, min_value=1, max_value=65535)]
        report = SettingsValidator(specs).validate({"PORT": 8080})
        assert report.ok is True
        assert report.errors == []

    def test_invalid_source_reports_named_errors(self):
        specs = [
            SettingSpec("PORT", int, min_value=1, max_value=65535),
            SettingSpec("NAME", str, required=True),
        ]
        report = SettingsValidator(specs).validate({"PORT": 999999})
        assert report.ok is False
        names = {i.setting_name for i in report.errors}
        assert names == {"PORT", "NAME"}

    def test_supports_attribute_style_source(self):
        class Source:
            HORIZON_URL = "https://horizon.stellar.org"

        specs = [SettingSpec("HORIZON_URL", str, validator=is_url)]
        report = SettingsValidator(specs).validate(Source)
        assert report.ok is True

    def test_validate_or_raise_raises_with_report(self):
        specs = [SettingSpec("MISSING", str, required=True)]
        validator = SettingsValidator(specs)
        with pytest.raises(SettingsValidationError) as exc:
            validator.validate_or_raise({})
        assert "MISSING" in str(exc.value)
        assert exc.value.report.ok is False

    def test_duplicate_spec_names_rejected(self):
        with pytest.raises(ValueError):
            SettingsValidator([SettingSpec("X", str), SettingSpec("X", int)])

    def test_report_render_lists_all_issues(self):
        specs = [SettingSpec("A", str, required=True), SettingSpec("B", str, required=True)]
        report = SettingsValidator(specs).validate({})
        rendered = report.render()
        assert "A" in rendered and "B" in rendered


class TestRiskThresholdSpecs:
    def _spec(self, name: str) -> SettingSpec:
        return next(s for s in DEFAULT_LEDGERLENS_SPECS if s.name == name)

    def test_risk_score_flag_threshold_rejects_out_of_range(self):
        spec = self._spec("RISK_SCORE_FLAG_THRESHOLD")
        issues = spec.check(700, present=True)
        assert len(issues) == 1
        assert "RISK_SCORE_FLAG_THRESHOLD" in str(issues[0])
        assert "700" in issues[0].message

    def test_risk_score_flag_threshold_accepts_in_range(self):
        spec = self._spec("RISK_SCORE_FLAG_THRESHOLD")
        assert spec.check(70, present=True) == []
        assert spec.check(0, present=True) == []
        assert spec.check(100, present=True) == []

    def test_mad_nonconformity_threshold_rejects_negative(self):
        spec = self._spec("MAD_NONCONFORMITY_THRESHOLD")
        issues = spec.check(-0.5, present=True)
        assert len(issues) == 1
        assert "MAD_NONCONFORMITY_THRESHOLD" in str(issues[0])

    def test_mad_nonconformity_threshold_accepts_default(self):
        spec = self._spec("MAD_NONCONFORMITY_THRESHOLD")
        assert spec.check(0.015, present=True) == []


def test_default_ledgerlens_specs_validate_against_live_config():
    from config import Config

    report = SettingsValidator(DEFAULT_LEDGERLENS_SPECS).validate(Config)
    # The live Config module ships sane defaults; this is a regression guard
    # so a future default change gets caught here with a named setting.
    assert report.ok, report.render()


class TestThresholdBoundsValidation:
    """Tests for negative and out-of-range threshold rejection."""

    def test_negative_threshold_rejected(self):
        """Negative threshold should fail validation with named variable and range."""
        specs = [
            SettingSpec(
                "RISK_SCORE_FLAG_THRESHOLD",
                int,
                required=True,
                min_value=0,
                max_value=100,
                description="must be in range [0, 100]",
            )
        ]
        report = SettingsValidator(specs).validate({"RISK_SCORE_FLAG_THRESHOLD": -5})
        assert not report.ok
        assert len(report.errors) == 1
        error = report.errors[0]
        assert error.setting_name == "RISK_SCORE_FLAG_THRESHOLD"
        assert "-5" in error.message
        assert "0" in error.message

    def test_above_max_threshold_rejected(self):
        """Threshold above 100 should fail validation."""
        specs = [
            SettingSpec(
                "RISK_SCORE_FLAG_THRESHOLD",
                int,
                required=True,
                min_value=0,
                max_value=100,
                description="must be in range [0, 100]",
            )
        ]
        report = SettingsValidator(specs).validate({"RISK_SCORE_FLAG_THRESHOLD": 105})
        assert not report.ok
        assert len(report.errors) == 1
        error = report.errors[0]
        assert error.setting_name == "RISK_SCORE_FLAG_THRESHOLD"
        assert "105" in error.message
        assert "100" in error.message

    def test_zero_threshold_passes(self):
        """Boundary value 0 should be valid."""
        specs = [
            SettingSpec(
                "RISK_SCORE_FLAG_THRESHOLD",
                int,
                required=True,
                min_value=0,
                max_value=100,
            )
        ]
        report = SettingsValidator(specs).validate({"RISK_SCORE_FLAG_THRESHOLD": 0})
        assert report.ok

    def test_hundred_threshold_passes(self):
        """Boundary value 100 should be valid."""
        specs = [
            SettingSpec(
                "RISK_SCORE_FLAG_THRESHOLD",
                int,
                required=True,
                min_value=0,
                max_value=100,
            )
        ]
        report = SettingsValidator(specs).validate({"RISK_SCORE_FLAG_THRESHOLD": 100})
        assert report.ok

    def test_negative_float_threshold_rejected(self):
        """Negative float threshold (e.g., for MAD_NONCONFORMITY_THRESHOLD) rejected."""
        specs = [
            SettingSpec(
                "MAD_NONCONFORMITY_THRESHOLD",
                float,
                required=True,
                min_value=0.0,
                description="must be non-negative",
            )
        ]
        report = SettingsValidator(specs).validate({"MAD_NONCONFORMITY_THRESHOLD": -0.5})
        assert not report.ok
        error = report.errors[0]
        assert error.setting_name == "MAD_NONCONFORMITY_THRESHOLD"
        assert "-0.5" in error.message
