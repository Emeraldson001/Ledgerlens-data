# Issue #734 — Validate `DP_EPSILON` and `DP_DELTA` Are Within Sane Bounds at Startup

## Summary

`detection/differential_privacy.py` implements the Gaussian and Laplace
mechanisms used to protect per-feature SHAP attributions against model
inversion. Both mechanisms share a mathematical dependency on two privacy
parameters supplied via environment variables:

| Variable | Default | Role |
|---|---|---|
| `DP_EPSILON` | `1.0` | Privacy budget — controls noise magnitude |
| `DP_DELTA` | `1e-5` | Failure probability — must stay in the open interval `(0, 1)` |

If either value is misconfigured (e.g. `DP_EPSILON=0`, `DP_EPSILON=-2`,
`DP_DELTA=0`, `DP_DELTA=1.5`) the formula

```
σ = Δ · √(2 · ln(1.25 / δ)) / ε
```

produces a division-by-zero, a logarithm of a non-positive number, or an
imaginary result. The failure currently surfaces deep inside NumPy's noise
sampling call as a `ZeroDivisionError` or `ValueError: math domain error`,
with no reference to which environment variable caused the problem. That makes
operator triage significantly harder.

---

## Current Behaviour

### `gaussian_sigma` (lines 37–44 in `differential_privacy.py`)

```python
def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if sensitivity < 0:
        raise ValueError("sensitivity must be >= 0")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
```

The parameter guards already exist **inside** `gaussian_sigma`. The gap is that
`gaussian_sigma` is a low-level utility called deep in the explain path.
Nothing validates `DP_EPSILON` and `DP_DELTA` at the point where the
`ShapExplainer` (or any caller) reads them from `config` — so an invalid env
var silently reaches `gaussian_sigma` only at explain time, not at import /
construction time.

### `laplace_scale` (lines 89–96)

```python
def laplace_scale(sensitivity: float, epsilon: float) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    ...
    return sensitivity / epsilon
```

Same pattern: the guard is post-hoc.

### `add_laplace_noise`

Delegates to `laplace_scale`; inherits the same late-failure behaviour.

---

## Root Cause

`config.DP_EPSILON` and `config.DP_DELTA` are read at call time, not validated
at the module boundary. There is no construction-time or import-time check that
rejects values that would make the privacy math undefined.

---

## Risk

| Scenario | Impact |
|---|---|
| `DP_EPSILON = 0` | `ZeroDivisionError` inside noise sampling; SHAP explanations unavailable |
| `DP_EPSILON < 0` | Negative sigma; noise goes in the wrong direction; DP guarantee void |
| `DP_DELTA = 0` | `math.log(1.25 / 0)` → domain error |
| `DP_DELTA >= 1` | `ln(1.25 / δ)` approaches zero or negative; σ underflows; DP guarantee void |
| `DP_DELTA > 1` | `(ε, δ)`-DP is meaningless for δ ≥ 1; the mechanism provides no privacy |

In all cases the error message given to the operator does not reference the
environment variable, making the misconfiguration hard to diagnose in a
deployed container where the stack trace may be truncated.

---

## Acceptance Criteria

1. Constructing the DP mechanism with `DP_EPSILON <= 0` raises a `ValueError`
   that names `DP_EPSILON` explicitly — e.g.:

   ```
   ValueError: DP_EPSILON must be > 0 (got 0). Set a positive value in your
   environment or .env file. See docs/issues/issue-734-dp-epsilon-delta-validation.md.
   ```

2. Constructing with `DP_DELTA` outside `(0, 1)` raises a `ValueError` that
   names `DP_DELTA` explicitly.

3. Validation fires at **construction time** (i.e. when `ShapExplainer` or any
   mechanism class is instantiated), not at the first noise-sampling call.

4. A test in `tests/test_dp_shap.py` or `tests/test_dp_training.py` covers at
   minimum the `DP_EPSILON = 0` case and confirms the error message contains
   the string `"DP_EPSILON"`.

---

## Proposed Implementation

### Option A — Validate in `gaussian_sigma` / `laplace_scale` with richer messages (minimal diff)

Change the existing guards to include the env-var name:

```python
def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    if epsilon <= 0:
        raise ValueError(
            f"DP_EPSILON must be > 0 (got {epsilon!r}). "
            "Check your environment configuration."
        )
    if not 0 < delta < 1:
        raise ValueError(
            f"DP_DELTA must be in the open interval (0, 1) (got {delta!r}). "
            "Check your environment configuration."
        )
    ...
```

Callers would then pass `config.DP_EPSILON` / `config.DP_DELTA` and the error
names the culprit.

### Option B — Validate at `ShapExplainer.__init__` (preferred for early failure)

Add a `_validate_dp_params` helper that is called in `ShapExplainer.__init__`:

```python
def _validate_dp_params(epsilon: float, delta: float) -> None:
    if epsilon <= 0:
        raise ValueError(
            f"DP_EPSILON must be > 0 (got {epsilon!r}). "
            "Update DP_EPSILON in your .env file or environment and restart."
        )
    if not 0 < delta < 1:
        raise ValueError(
            f"DP_DELTA must be in the open interval (0, 1) (got {delta!r}). "
            "Valid range: 0 < DP_DELTA < 1 (typical: 1e-5)."
        )
```

Option B is preferred because it surfaces the misconfiguration at startup /
object-creation time, before any wallet is scored, reducing the blast radius.

---

## Suggested Test

```python
# tests/test_dp_shap.py

def test_gaussian_sigma_rejects_zero_epsilon():
    """Issue #734 — DP_EPSILON = 0 must raise ValueError naming the parameter."""
    with pytest.raises(ValueError, match="DP_EPSILON"):
        gaussian_sigma(0.3, 0.0, 1e-5)

def test_gaussian_sigma_rejects_negative_epsilon():
    with pytest.raises(ValueError, match="DP_EPSILON"):
        gaussian_sigma(0.3, -1.0, 1e-5)

def test_gaussian_sigma_rejects_delta_out_of_range():
    with pytest.raises(ValueError, match="DP_DELTA"):
        gaussian_sigma(0.3, 1.0, 0.0)
    with pytest.raises(ValueError, match="DP_DELTA"):
        gaussian_sigma(0.3, 1.0, 1.0)
    with pytest.raises(ValueError, match="DP_DELTA"):
        gaussian_sigma(0.3, 1.0, 1.5)
```

---

## Affected Files

| File | Change type |
|---|---|
| `detection/differential_privacy.py` | Add/enrich parameter validation with env-var names |
| `tests/test_dp_shap.py` | Add rejection tests for `DP_EPSILON = 0`, negative epsilon, out-of-range delta |

---

## Related

- Issue #59 — original DP-SHAP implementation
- `docs/privacy.md` — privacy architecture overview
- `config.py` — `DP_EPSILON`, `DP_DELTA`, `DP_DEFAULT_SENSITIVITY`, `DP_RENYI_*` settings
- Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014) §3.3
- Mironov, "Rényi Differential Privacy of the Gaussian Mechanism" (CSF 2017)

