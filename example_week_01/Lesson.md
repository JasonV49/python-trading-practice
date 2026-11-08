# Python Trading Practice 01 | Validate OHLCV and split chronologically

This exercise starts the series.

This week's code extends the previous research components. Allow about 45–60 minutes. The exercise uses Python 3.12 and its standard library.

## Concept

Validate before transforming. A close return becomes available at the second close. Split the bars before fitting any parameters.

## Coding challenge

Validate dated OHLCV rows, compute simple close-to-close returns, and split the bars by chronological index.

Complete `validate(bars)`, `close_returns(bars)`, `split(rows, train_end, validation_end)` in `exercise/trading.py`. Read the test cases for input and output examples.

**Success criteria.** Reject duplicate dates and impossible ranges. The first return equals close[1] / close[0] − 1. Splits are disjoint, ordered, and exhaustive.

**Hint.** Start with the simplest test case and calculate the expected result by hand.

## Bias and overfitting checks

- **Look-ahead.** Perturb future prices, then compare decisions through the current open. The deliberately leaky control must fail the invariance assertion.
- **Overfitting.** Perturb the holdout and check that it cannot change development-only parameter selection. The deliberately contaminated selector must fail. The seeded null experiment also compares the effects of selecting from one versus twenty candidates. Neither check establishes that a strategy generalizes.
- Use the lesson-specific test alongside these pipeline checks.

Run from the extracted `exercise/` folder.

```sh
python3 -m unittest -v test_trading.TradingTests.test_open_positions_ignore_future_closes test_trading.TradingTests.test_leakage_positive_control test_trading.TradingTests.test_selection_ignores_holdout test_trading.TradingTests.test_overfitting_positive_control test_trading.TradingTests.test_null_search test_trading.TradingTests.test_data_contract
```

**Reflection.** What data is available when the decision is made, and what data is reserved for evaluation?

## Worked solution

Validate before transforming. A close return becomes available at the second close. Split the bars before fitting any parameters.

The runnable implementation is in the separate `solution/` folder. Run the same command there to reproduce the test checks, then run this command to reproduce the experiment.

```sh
python3 -c "import json; from curriculum import experiment; print(json.dumps(experiment(1)[2], indent=2))"
```

### Verified on this run

The reviewed Python implementation passed 6 named test checks for this lesson. The reported experiment ran with synthetic fixture seed 20261110 and the parameters in `curriculum.py`. Returns are decimal fractions.

```json
{
  "bars": 321,
  "first_close_return": -0.0014158736163758512,
  "split_sizes": [
    160,
    80,
    81
  ]
}
```

### Assumptions and limits

The synthetic bars model an exercise and do not follow an exchange calendar. Strategy fills are modeled at the open, with whole-sleeve exposure, no market impact, and linear per-turnover costs where applicable. The default is 5 bps in fees plus 5 bps in slippage. The gross buy-and-hold benchmark excludes entry costs. These simulated observations provide no evidence of a live trading edge.

### Handoff

Validate OHLCV and split chronologically added `validate(bars)`, `close_returns(bars)`, `split(rows, train_end, validation_end)`. Next exercise: Build a causal momentum signal.
