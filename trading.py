"""Pure Python reference implementation for the trading exercises."""
from datetime import date, timedelta
from math import isfinite, prod
from random import Random
from statistics import mean


def fixture(seed=20261110, count=321):
    """Synthetic daily OHLCV bars. Dates are labels, not an exchange calendar."""
    rng = Random(seed)
    previous_close = 100.0
    bars = []
    for i in range(count):
        drift = (0.0008, -0.0005, 0.0003)[min(2, i * 3 // count)]
        open_price = previous_close * (1 + rng.uniform(-0.008, 0.008))
        close = open_price * (1 + drift + rng.uniform(-0.023, 0.023))
        bars.append((
            (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
            open_price,
            max(open_price, close) * 1.003,
            min(open_price, close) * 0.997,
            close,
            rng.randint(1000, 10000),
        ))
        previous_close = close
    return bars


def validate(bars):
    previous = None
    for timestamp, opening, high, low, close, volume in bars:
        day = date.fromisoformat(timestamp)
        if previous is not None and day <= previous:
            raise ValueError("Dates must be strictly increasing")
        if not all(isfinite(x) and x > 0 for x in (opening, high, low, close)):
            raise ValueError("Prices must be positive and finite")
        if low > min(opening, close) or high < max(opening, close):
            raise ValueError("Open and close must lie within low and high")
        if not isinstance(volume, int) or isinstance(volume, bool) or volume < 0:
            raise ValueError("Volume must be a nonnegative integer")
        previous = day


def close_returns(bars):
    validate(bars)
    return [today[4] / yesterday[4] - 1 for yesterday, today in zip(bars, bars[1:])]


def split(rows, train_end, validation_end):
    if not 0 < train_end < validation_end < len(rows):
        raise ValueError("Training, validation and test must all be nonempty")
    return rows[:train_end], rows[train_end:validation_end], rows[validation_end:]


def positions(bars, lookback, delay=1):
    if lookback < 1 or delay < 1:
        raise ValueError("Lookback and delay must be positive")
    # An order at open i may only use close i - delay or earlier.
    return [
        int(bars[i - delay][4] > bars[i - delay - lookback][4])
        if i >= delay + lookback else 0
        for i in range(len(bars))
    ]


def backtest(bars, exposure, fee_bps=5, slippage_bps=5):
    if len(exposure) != len(bars):
        raise ValueError("Supply one open-time position per bar")
    gross, net, benchmark, turnover = [], [], [], []
    previous_position = 0
    for i, (opening_bar, closing_bar) in enumerate(zip(bars, bars[1:])):
        position = exposure[i]
        move = closing_bar[1] / opening_bar[1] - 1
        trade = abs(position - previous_position)
        gross.append(position * move)
        net.append(position * move - trade * (fee_bps + slippage_bps) / 10_000)
        benchmark.append(move)
        turnover.append(trade)
        previous_position = position
    return dict(gross=gross, net=net, benchmark=benchmark, turnover=turnover)


def total(returns):
    return prod(1 + move for move in returns) - 1


def select_window(bars, candidates, train_end, validation_end):
    # Omit the last row of each segment as a return start if its outcome
    # needs an opening price from the next segment.
    train, validation, _ = split(bars, train_end, validation_end)
    available = train + validation
    ranked = sorted(
        candidates,
        key=lambda n: (-mean(backtest(train, positions(train, n))["net"]), n),
    )
    finalists = ranked[:2]
    scores = {
        n: mean(backtest(available, positions(available, n))["net"][train_end:])
        for n in finalists
    }
    return min(finalists, key=lambda n: (-scores[n], n))


def walk_forward(bars, candidates, first_end=160, step=40):
    folds = []
    for boundary in range(first_end, len(bars) - 1, step):
        end = min(boundary + step, len(bars) - 1)
        selection = select_window(bars[:boundary + 1], candidates, boundary - step, boundary)
        visible = bars[:end + 1]
        out_of_sample = backtest(visible, positions(visible, selection))["net"][boundary:]
        folds.append((boundary, end, selection, total(out_of_sample)))
    return folds


def bootstrap_interval(excess_returns, block=8, samples=500, seed=123):
    if not 0 < block <= len(excess_returns) or samples < 40:
        raise ValueError("Provide a valid block size and at least 40 draws")
    rng = Random(seed)
    size = len(excess_returns)
    estimates = []
    for _ in range(samples):
        draw = []
        for _ in range((size + block - 1) // block):
            start = rng.randrange(size)
            draw.extend(excess_returns[(start + j) % size] for j in range(block))
        estimates.append(mean(draw[:size]))
    estimates.sort()
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def null_search(trials=100, candidates=20, sample_size=80, seed=73):
    """A deliberately simplified null with independent candidate returns."""
    rng = Random(seed)
    train_scores, test_scores = [], []
    for _ in range(trials):
        train = [mean(rng.choice((-0.01, 0.01)) for _ in range(sample_size))
                 for _ in range(candidates)]
        holdout = [mean(rng.choice((-0.01, 0.01)) for _ in range(sample_size))
                   for _ in range(candidates)]
        winner = max(range(candidates), key=train.__getitem__)
        train_scores.append(train[winner])
        test_scores.append(holdout[winner])
    return mean(train_scores), mean(test_scores)


def stress(bars, lookback, costs, cuts, delay=1):
    exposure = positions(bars, lookback, delay)
    return {
        str(cost): [
            total(backtest(bars, exposure, cost, 0)["net"][start:end - 1])
            for start, end in zip(cuts, cuts[1:])
        ]
        for cost in costs
    }


def portfolio(bars_by_asset, lookback, cost_bps=10):
    if not bars_by_asset:
        raise ValueError("Provide at least one asset")
    dates = [row[0] for row in bars_by_asset[0]]
    if any([row[0] for row in bars] != dates for bars in bars_by_asset):
        raise ValueError("Asset bars must align by date")
    sleeves = [backtest(bars, positions(bars, lookback), cost_bps, 0)["net"]
               for bars in bars_by_asset]
    return [mean(period) for period in zip(*sleeves)]


def delay_sensitivity(bars, lookback, delays, cost_bps=10):
    return {
        str(delay): total(backtest(bars, positions(bars, lookback, delay), cost_bps, 0)["net"])
        for delay in delays
    }


def neighborhood(bars, candidates, train_end, validation_end):
    winner = select_window(bars, candidates, train_end, validation_end)
    frozen = [n for n in candidates if abs(n - winner) <= 5]
    return winner, {
        str(n): total(backtest(bars, positions(bars, n))["net"][validation_end:])
        for n in frozen
    }


def robustness(bars, candidates, train_end, validation_end, costs, delays):
    winner = select_window(bars, candidates, train_end, validation_end)
    return winner, {
        f"delay={delay}, cost={cost}bps": total(
            backtest(bars, positions(bars, winner, delay), cost, 0)["net"][validation_end:]
        )
        for delay in delays for cost in costs
    }
