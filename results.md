# Results

**Generated 2026-11-02 by executing this repository's own test suite against the
current `main`.**

## Status: 12 passed, 1 FAILED

| Metric | Value |
|---|---|
| Tests passed | 12 |
| Tests failed | **1** |
| Skipped | 1 |
| Wall clock | 75.2 s |
| Environment | Python 3.11, CPU-only, no GPU |

Reproduce with:

```bash
PYTHONPATH=$PWD:$PWD/src pytest -q
```

## ⚠️ Failing test

- (see pytest output for the failing node id)

**This is a negative result and is reported as such.** It is not hidden, and it
is not averaged away by the passing tests.

### What is established

- 12 of 13 tests pass. The code imports cleanly and the bulk of the
  suite behaves as specified.

### What is NOT established

- The failing assertion does not hold in this environment.
- Whether that is flakiness (unseeded randomness), an environment dependency,
  or a genuine regression has **not** been determined by this run.
- No independent benchmark, metric, or comparative claim was reproduced.
- Coverage was not measured.

## Recommended next step

Determine whether the failing test is flaky, environment-dependent, or asserting
a claim that no longer holds. Until then, treat the behaviour it guards as
**unverified**.

## Negative and unverified results

- **1 test failure** -- see above.
- **No experimental result was independently reproduced.**
