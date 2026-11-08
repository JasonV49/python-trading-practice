"""Authored curriculum, acceptance criteria and deterministic experiment inputs."""
from dataclasses import dataclass

import trading as t


@dataclass(frozen=True)
class Unit:
    title: str
    functions: tuple
    challenge: str
    criteria: str
    explanation: str


UNITS = (
    Unit("Validate OHLCV and split chronologically", (t.validate, t.close_returns, t.split),
         "Validate dated OHLCV rows, compute simple close-to-close returns, and split the bars by chronological index.",
         "Reject duplicate dates and impossible ranges. The first return equals close[1] / close[0] − 1. Splits are disjoint, ordered, and exhaustive.",
         "Validate before transforming. A close return becomes available at the second close. Split the bars before fitting any parameters."),
    Unit("Build a causal momentum signal", (t.positions,),
         "Return one long-or-flat decision for each bar. At open i, compare the close at i − delay with the close lookback bars before it.",
         "Return zero through warm-up. A large change at close i cannot change the position at open i. Test both one- and two-bar delays.",
         "The close that determines a position predates its open fill. The next open-to-open return is the outcome, not a signal input."),
    Unit("Account for costs and compare a benchmark", (t.backtest, t.total),
         "Compute gross and net open-to-open returns, a buy-and-hold return series, and absolute position turnover. Compound the returns.",
         "Use one hand-calculated round trip. Charge fees and slippage on position changes, including entry from flat. Check flat gross P&L and a zero-cost benchmark.",
         "The model starts flat and skips a final fill without a following open. Fees and slippage are linear per unit of turnover. There is no forced liquidation."),
    Unit("Freeze parameters before the holdout", (t.select_window,),
         "Rank candidate windows on training mean net return. Keep the best two, choose between them on validation, and evaluate the frozen winner on the holdout.",
         "Changing the holdout must leave the winner unchanged. Break score ties by shorter lookback. Exclude returns that cross the train or validation endpoint during selection.",
         "The holdout is only for evaluation. The validation series may use earlier training bars as signal history."),
    Unit("Implement walk-forward evaluation", (t.walk_forward,),
         "Use expanding training history and a fixed-length validation window. Record the winner and compounded net return in each later block.",
         "Each fold selects before its test block. Fold test blocks are disjoint. A later shock cannot alter earlier folds.",
         "Each fold is an independent simulation of a frozen parameter. Report fold results separately. They do not constitute a continuous traded portfolio."),
    Unit("Estimate uncertainty with a block bootstrap", (t.bootstrap_interval,),
         "Use a seeded circular block bootstrap to estimate an interval for mean daily strategy return in excess of buy-and-hold.",
         "Same seed gives the same interval. A constant series returns its constant value at both endpoints. Compare two block lengths.",
         "Sample contiguous blocks to retain some serial dependence. The percentile interval is approximate and depends on block length and the suitability of resampling the chosen series."),
    Unit("Measure selection effects under a null", (t.null_search,),
         "Simulate independent, zero-mean candidate returns. Compare the selected winner's mean return in development and on an independent holdout.",
         "The seeded experiment is reproducible. Candidate selection sees only the first sample. Compare one candidate with a 20-candidate search.",
         "Selecting the largest noisy estimate biases the development score upward. The simulated candidates are independent by construction, unlike many real trading rules."),
    Unit("Stress costs and predefined regimes", (t.stress,),
         "Evaluate one pre-specified momentum window across three calendar blocks and a grid of transaction costs.",
         "Cut points and cost grid are fixed before evaluation. Verify that higher costs cannot increase any positive-wealth regime result.",
         "Calendar blocks are scenario labels. Their results are descriptive. Costs are model inputs, and each segment inherits the preceding strategy position."),
    Unit("Combine multiple assets", (t.portfolio,),
         "Align three synthetic assets by date and aggregate equal-weight strategy-sleeve returns.",
         "Reject mismatched dates. Check that identical sleeves reproduce the single-sleeve result. Report the assumptions behind equal weighting.",
         "The example holds equal sleeves each period. Sleeve turnover costs are included. Transfers required to restore portfolio weights are assumed costless."),
    Unit("Test execution delay", (t.delay_sensitivity,),
         "Compare one frozen signal with several open-fill delays and the same cost assumption.",
         "The delay-one case matches the original backtest. Future-close changes cannot affect prior fills at any delay.",
         "Delay is measured in bars after the signal close. A delay sensitivity table measures the model's dependence on the execution assumption."),
    Unit("Inspect the parameter neighborhood", (t.neighborhood,),
         "Freeze the development winner and report holdout results for candidate lookbacks within five bars of it.",
         "Changing holdout prices can change reported performance but cannot change either the winner or the neighborhood.",
         "Treat the holdout neighborhood as a diagnostic for fragility. Any decision based on that table requires a fresh final holdout."),
    Unit("Run a joint robustness experiment", (t.robustness,),
         "Freeze the training/validation winner and evaluate a pre-specified grid of costs and delays on the holdout.",
         "Return one result per scenario. Verify every scenario uses the same frozen winner and only holdout-period returns in its final score.",
         "Costs and delays vary together so their interaction is visible. Select neither from the holdout grid."),
)


def experiment(number):
    if number < 1:
        raise ValueError("Lesson number must be positive")
    seed = 20261109 + number
    bars = t.fixture(seed)
    candidates = [5, 10, 15, 20, 30]
    if number > 12:
        # Subsequent lessons reuse the audited capstone with increasing cost
        # stress and a rotating pre-specified delay range.
        costs = (0, 10, min(100, 15 + number))
        delays = (1, 2, 3 + number % 4)
        winner, grid = t.robustness(bars, candidates, 160, 240, costs, delays)
        return UNITS[-1], seed, {"frozen_lookback": winner, "holdout_scenarios": grid}
    unit = UNITS[number - 1]
    experiments = (
        lambda: {"bars": len(bars), "first_close_return": t.close_returns(bars)[0],
                 "split_sizes": list(map(len, t.split(bars, 160, 240)))},
        lambda: {"long_opens": sum(t.positions(bars, 10)), "lookback_bars": 10},
        lambda: {"strategy_net": t.total(t.backtest(bars, t.positions(bars, 10))["net"]),
                 "buy_and_hold_gross": t.total(t.backtest(bars, t.positions(bars, 10))["benchmark"])},
        lambda: _selection_results(bars, candidates),
        lambda: {"folds": t.walk_forward(bars, candidates)},
        lambda: _bootstrap_results(bars),
        lambda: {"one_candidate_train_test_means": t.null_search(candidates=1, seed=seed),
                 "twenty_candidate_train_test_means": t.null_search(candidates=20, seed=seed)},
        lambda: {"cost_bps_to_regime_returns": t.stress(bars, 10, (0, 10, 25), (0, 107, 214, len(bars)))},
        lambda: {"equal_sleeve_net": t.total(t.portfolio([t.fixture(seed+i) for i in range(3)], 10))},
        lambda: {"delay_bars_to_net": t.delay_sensitivity(bars, 10, (1, 2, 3, 5))},
        lambda: dict(zip(("frozen_lookback", "neighbor_holdout_net"),
                         t.neighborhood(bars, candidates, 160, 240))),
        lambda: dict(zip(("frozen_lookback", "holdout_scenarios"),
                         t.robustness(bars, candidates, 160, 240, (0, 10, 25), (1, 2, 4)))),
    )
    return unit, seed, experiments[number - 1]()


def _selection_results(bars, candidates):
    winner = t.select_window(bars, candidates, 160, 240)
    returns = t.backtest(bars, t.positions(bars, winner))["net"]
    return {"frozen_lookback": winner, "holdout_net": t.total(returns[240:])}


def _bootstrap_results(bars):
    result = t.backtest(bars, t.positions(bars, 10))
    excess = [strategy - benchmark for strategy, benchmark in zip(result["net"], result["benchmark"])]
    return {f"block_{block}_95pct": t.bootstrap_interval(excess, block) for block in (4, 12)}
