"""Independent checks for the reference implementations."""
import unittest
from math import isclose
from statistics import mean

import trading as t
from curriculum import experiment


def shock(bars, start):
    """Change both prices and returns from start, keeping valid OHLCV ranges."""
    return [row if i < start else (row[0], *(x * (8 + i % 3) for x in row[1:5]), row[5])
            for i, row in enumerate(bars)]


def check_causal(position_fn, bars, cutoff, delay):
    altered = shock(bars, cutoff)
    # Positions through the open of the changed bar may not use that close.
    assert position_fn(bars, 10, delay)[:cutoff + 1] == position_fn(
        altered, 10, delay)[:cutoff + 1]


def check_holdout(selector, bars):
    candidates = [5, 10, 15, 20, 30]
    winner = selector(bars, candidates, 160, 240)
    assert winner == selector(shock(bars, 240), candidates, 160, 240)


class TradingTests(unittest.TestCase):
    def setUp(self):
        self.bars = t.fixture()

    def test_data_contract(self):
        t.validate(self.bars)
        with self.assertRaises(ValueError):
            t.validate([self.bars[0], self.bars[0]])
        day, opening, high, low, close, volume = self.bars[0]
        with self.assertRaises(ValueError):
            t.validate([(day, opening, low, high, close, volume)])
        self.assertAlmostEqual(t.close_returns(self.bars)[0],
                               self.bars[1][4] / self.bars[0][4] - 1)
        self.assertEqual(t.close_returns(self.bars)[:120],
                         t.close_returns(shock(self.bars, 121))[:120])
        pieces = t.split(self.bars, 160, 240)
        self.assertEqual(tuple(map(len, pieces)), (160, 80, 81))
        self.assertEqual(sum(pieces, []), self.bars)
        self.assertEqual(pieces[:2], t.split(shock(self.bars, 240), 160, 240)[:2])

    def test_open_positions_ignore_future_closes(self):
        for delay in (1, 2, 5):
            for cutoff in (75, 159, 240):
                check_causal(t.positions, self.bars, cutoff, delay)
            exposure = t.positions(self.bars, 10, delay)
            self.assertEqual(exposure[:10 + delay], [0] * (10 + delay))

    def test_leakage_positive_control(self):
        def leaky(bars, lookback, delay):
            return [int(row[4] > 500) for row in bars]

        with self.assertRaises(AssertionError):
            check_causal(leaky, self.bars, 75, 1)

    def test_backtest_costs_and_alignment(self):
        bars = [(str(i), opening, opening, opening, opening, 1)
                for i, opening in enumerate((100, 110, 99))]
        result = t.backtest(bars, [1, 0, 1], fee_bps=7, slippage_bps=3)
        self.assertEqual(result["turnover"], [1, 1])
        self.assertAlmostEqual(result["gross"][0], .1)
        self.assertAlmostEqual(result["net"][0], .099)
        self.assertAlmostEqual(result["net"][1], -.001)
        self.assertAlmostEqual(result["benchmark"][1], -.1)
        self.assertAlmostEqual(t.total(result["net"]), 1.099 * .999 - 1)
        buy_hold = t.backtest(bars, [1, 1, 1], 0, 0)
        self.assertEqual(buy_hold["net"], buy_hold["benchmark"])

    def test_selection_ignores_holdout(self):
        check_holdout(t.select_window, self.bars)

        class NoHoldout(list):
            def __getitem__(self, key):
                if isinstance(key, slice) and key.start == 240:
                    return ["withheld"]
                return super().__getitem__(key)

        candidates = [5, 10, 15, 20, 30]
        self.assertEqual(t.select_window(NoHoldout(self.bars), candidates, 160, 240),
                         t.select_window(self.bars, candidates, 160, 240))

    def test_overfitting_positive_control(self):
        def leaky(bars, candidates, train_end, validation_end):
            return candidates[int(bars[validation_end][4] > 500)]

        with self.assertRaises(AssertionError):
            check_holdout(leaky, self.bars)

    def test_walk_forward_folds(self):
        folds = t.walk_forward(self.bars, [5, 10, 15, 20, 30])
        changed = t.walk_forward(shock(self.bars, 241), [5, 10, 15, 20, 30])
        self.assertEqual(folds[:2], changed[:2])
        self.assertEqual([(a, b) for a, b, _, _ in folds],
                         [(160, 200), (200, 240), (240, 280), (280, 320)])
        for start, stop, winner, net in folds:
            self.assertEqual(winner, t.select_window(
                self.bars[:stop + 1], [5, 10, 15, 20, 30], start - 40, start))
            returns = t.backtest(self.bars, t.positions(self.bars, winner))["net"]
            self.assertAlmostEqual(net, t.total(returns[start:stop]))

    def test_bootstrap(self):
        self.assertEqual(t.bootstrap_interval([.25] * 40), (.25, .25))
        data = t.close_returns(self.bars)
        lower, upper = t.bootstrap_interval(data)
        self.assertLess(lower, upper)
        self.assertEqual((lower, upper), t.bootstrap_interval(data))

    def test_null_search(self):
        one = t.null_search(candidates=1)
        many = t.null_search(candidates=20)
        self.assertEqual(many, t.null_search(candidates=20))
        self.assertGreater(many[0], one[0] + .001)
        self.assertLess(abs(many[1]), .0005)

    def test_cost_and_delay_sensitivity(self):
        grid = t.stress(self.bars, 10, (0, 10, 25), (0, 107, 214, 321))
        for free, mid, high in zip(grid["0"], grid["10"], grid["25"]):
            self.assertGreaterEqual(free, mid)
            self.assertGreaterEqual(mid, high)
        original = t.backtest(self.bars, t.positions(self.bars, 10))["net"]
        self.assertEqual(t.delay_sensitivity(self.bars, 10, (1, 2))["1"],
                         t.total(original))

    def test_equal_sleeve_portfolio(self):
        assets = [self.bars, t.fixture(17)]
        sleeves = [t.backtest(asset, t.positions(asset, 10))["net"] for asset in assets]
        self.assertEqual(t.portfolio(assets, 10), list(map(mean, zip(*sleeves))))
        self.assertEqual(t.portfolio([self.bars] * 2, 10), sleeves[0])
        with self.assertRaises(ValueError):
            t.portfolio([self.bars, self.bars[1:]], 10)

    def test_neighborhood_and_robustness_freeze_winner(self):
        candidates = [5, 10, 15, 20, 30]
        altered = shock(self.bars, 240)
        winner, neighbors = t.neighborhood(self.bars, candidates, 160, 240)
        changed_winner, changed_neighbors = t.neighborhood(altered, candidates, 160, 240)
        self.assertEqual((winner, neighbors.keys()), (changed_winner, changed_neighbors.keys()))
        costs, delays = (0, 10, 25), (1, 2, 4)
        selected, grid = t.robustness(self.bars, candidates, 160, 240, costs, delays)
        self.assertEqual(selected, winner)
        self.assertEqual(len(grid), 9)
        baseline = t.backtest(self.bars, t.positions(self.bars, winner))["net"]
        self.assertTrue(isclose(grid["delay=1, cost=10bps"], t.total(baseline[240:])))

    def test_all_lessons_replay(self):
        for number in range(1, 15):
            with self.subTest(number=number):
                first = experiment(number)
                self.assertEqual(first, experiment(number))
                self.assertTrue(first[2])


if __name__ == "__main__":
    unittest.main()
