# Results

**Generated 2026-09-29 by executing this repository's own test suite.**

## Status: 10 passed, 1 FAILED

| Metric | Value |
|---|---|
| Tests passed | 10 |
| Tests failed | **1** |
| Skipped | 1 |
| Wall clock | 102.39 s |
| Environment | Python 3.11, CPU-only, no GPU (torch 2.14.0+cu130) |

Reproduce with:

```bash
PYTHONPATH=$PWD:$PWD/src pytest -q
```

## ⚠️ Failing test

- `tests/test_benchmark.py::test_run_benchmark_table`

**This is a negative result and is reported as such.** The repository's own test
suite asserts a performance ordering that does not hold in this environment. The
failing assertion compares two benchmark numbers:

```
assert pre > scr
E   assert np.float64(0.42615042058386937) > np.float64(0.4951756556160317)
```

The test expects the `pre` method to score above `scr`; it does not. This is
exactly the kind of self-verifying claim that should be surfaced rather than
hidden, so it is recorded here in full.

### What is established

- 10 of 11 tests pass. The code imports cleanly and the bulk of the
  suite behaves as specified.

### What is NOT established

- **The `pre` > `scr` performance claim is not reproducible here.** Whether this
  is a seed effect, a hardware difference, a genuine regression, or a
  over-specified test has not been determined by this run.
- No independent benchmark, metric, or comparative claim was reproduced.
- Coverage was not measured.

## Recommended next step

Determine whether `tests/test_benchmark.py::test_run_benchmark_table` is
flaky (unseeded randomness), environment-dependent, or asserting a claim that no
longer holds. Until then, treat the ordering as **unverified**.

**No quantitative performance claims were detected in the README**, so there is
no benchmark figure here that this run confirms or contradicts.
## Negative and unverified results

- **1 test failure** — see above. Not hidden, not averaged away.
- **No experimental result was independently reproduced.** The suite passing
  elsewhere is not evidence the research is correct.
