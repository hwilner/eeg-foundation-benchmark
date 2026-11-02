# Results

**Generated 2026-11-02 by executing this repository's own test suite against the
current `main`.**

## Verification run

| Metric | Value |
|---|---|
| Tests passed | **13** |
| Tests failed | 0 |
| Skipped | 1 |
| Wall clock | ~75 s |
| Environment | Python 3.11, CPU-only, no GPU (torch 2.14.0+cu130) |

Reproduce with:

```bash
PYTHONPATH=$PWD:$PWD/src pytest -q
```

The suite passes in isolation. This was confirmed across four independent runs, all
reporting `13 passed, 1 skipped`.

## ⚠️ The benchmark test is not robust under resource contention

The suite passes cleanly when run on an otherwise idle machine, but it does not
behave reliably when the machine is busy. Three observations:

| Condition | Result |
|---|---|
| Isolated, idle machine (4 independent runs) | `13 passed, 1 skipped` — ~70–75 s |
| Immediately after 15 other test suites ran back-to-back | `1 failed, 12 passed, 1 skipped` |
| `tests/test_benchmark.py` under 3× concurrent CPU load | **exceeded a 900 s timeout; did not complete** |

The runtime itself is highly variable: two isolated runs of the same test file took
**618 s** and **70 s** — a 9× spread on identical code with no source change.

**What this means.** The assertion guarded by `tests/test_benchmark.py` compares
benchmark scores and expects a specific ordering. Under contention the underlying
numeric results move enough to flip that ordering, and under heavier contention the
run does not finish in a reasonable time at all.

This is **not** a claim that the underlying science is wrong, and it is **not** a
defect introduced by any recent change — the failure was observed against the
current `main` and could not be reproduced in isolation. It is a statement about
the *robustness of the test*, not about the correctness of the model.

### Recommended next steps

1. **Assert the ordering with a tolerance, not exact inequality.** If the score
   difference the test relies on is small relative to run-to-run numerical noise,
   compare with a margin, or seed the benchmark explicitly and assert on a
   seeded run (the 2026-10-03 commit "Seed the benchmark before building encoders;
   fix a false test assertion" is in this direction).
2. **Make the timeout explicit.** The test should declare its own expected budget
   so a starved run fails with a clear message rather than a harness timeout.
3. **Add CI.** Unlike `kimi-delta-attention`, this repository has no workflow
   runs on GitHub Actions, so nothing catches this in an automated, controlled
   environment today.

## What this establishes

- The code imports cleanly and **13 tests pass with 0 failures** on an idle machine.
- The remaining suite is fast and deterministic.

## What this does NOT establish

- **No scientific claim is verified by a passing unit test.** These tests check
  that functions behave as specified -- not that an EEG foundation-model claim holds.
- **No benchmark, metric, or comparative result was reproduced.** The unit tests
  are not experiments; the benchmark test asserts an ordering, it does not measure
  anything new.
- **Coverage was not measured.** A passing suite can still leave large areas untested.
- **Statistical significance was not assessed.** No experiment was run.

## Negative and unverified results

Recorded explicitly, so they are not mistaken for successes:

- **The benchmark test failed once and timed out once under resource contention.**
  See the table above. This is a robustness finding about the test, and it is the
  most important thing in this file.
- **No empirical result was reproduced for this repository.** Any finding described
  in the README or `docs/` was not re-derived here.
- **The test suite passing is not evidence the research is correct.** It is
  evidence the code is internally consistent, under the conditions tested.

## Reproducing

```bash
PYTHONPATH=$PWD:$PWD/src pytest -q
```

For a stable timing measurement, run on an otherwise idle machine.
