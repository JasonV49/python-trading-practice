"""Render a weekly lesson from authored code and optional model-written teaching notes."""
import base64
import io
import inspect
import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile

import trading
from curriculum import UNITS, experiment

ROOT = Path(__file__).resolve().parent
FIELDS = ("concept", "hint", "reflection")
REQUIRED_TESTS = (
    "test_open_positions_ignore_future_closes", "test_leakage_positive_control",
    "test_selection_ignores_holdout", "test_overfitting_positive_control", "test_null_search",
)
FOCUS_TESTS = (
    "test_data_contract", "test_open_positions_ignore_future_closes",
    "test_backtest_costs_and_alignment", "test_selection_ignores_holdout",
    "test_walk_forward_folds", "test_bootstrap", "test_null_search",
    "test_cost_and_delay_sensitivity", "test_equal_sleeve_portfolio",
    "test_cost_and_delay_sensitivity", "test_neighborhood_and_robustness_freeze_winner",
    "test_neighborhood_and_robustness_freeze_winner",
)


def post_json(url, key, payload, headers=None):
    request = Request(url, data=json.dumps(payload, sort_keys=True).encode(), headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **(headers or {}),
    })
    try:
        with urlopen(request, timeout=90) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"{request.host} returned HTTP {error.code}") from None
    except (URLError, TimeoutError):
        raise RuntimeError(f"{request.host} request failed") from None


def teach(unit, previous, key, model):
    result = post_json("https://api.openai.com/v1/responses", key, {
        "model": model,
        "store": False,
        "max_output_tokens": 2000,
        "instructions": (
            "Write short teaching notes for an algorithmic trading exercise. Use the provided "
            "facts. Explain the concept, offer one implementation hint, and ask one reflection "
            "question. Use plain text. Omit code, performance values, claims of test execution, "
            "and claims of trading efficacy."
        ),
        "input": json.dumps({
            "title": unit.title, "challenge": unit.challenge,
            "explanation": unit.explanation, "previous_handoff": previous,
        }),
        "text": {"format": {
            "type": "json_schema", "name": "teaching_notes", "strict": True,
            "schema": {"type": "object",
                       "properties": {field: {"type": "string"} for field in FIELDS},
                       "required": list(FIELDS), "additionalProperties": False},
        }},
    })
    if result.get("status") != "completed":
        raise RuntimeError("OpenAI response did not complete")
    chunks = [part["text"] for message in result.get("output", [])
              if message.get("type") == "message"
              for part in message.get("content", []) if part.get("type") == "output_text"]
    notes = json.loads("".join(chunks)) if chunks else {}
    if set(notes) != set(FIELDS) or any(not isinstance(notes[k], str) or not notes[k].strip()
                                        for k in FIELDS):
        raise RuntimeError("OpenAI response did not contain complete teaching notes")
    return notes


def verify(number):
    names = (*REQUIRED_TESTS, FOCUS_TESTS[min(number, len(FOCUS_TESTS)) - 1])
    # Run reviewed tests and reviewed source only. Model output is never executed.
    commands = [f"test_trading.TradingTests.{name}" for name in dict.fromkeys(names)]
    subprocess.run([sys.executable, "-m", "unittest", "-q", *commands],
                   cwd=ROOT, check=True, timeout=60, capture_output=True)
    return commands


def make_bundle(markdown, functions):
    reference = inspect.getsource(trading)
    starter = reference
    for function in functions:
        stub = f"def {function.__name__}{inspect.signature(function)}:\n    raise NotImplementedError\n"
        starter = starter.replace(inspect.getsource(function), stub, 1)
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr("Lesson.md", markdown)
        for folder, code in (("exercise", starter), ("solution", reference)):
            bundle.writestr(f"{folder}/trading.py", code)
            for filename in ("test_trading.py", "curriculum.py"):
                bundle.write(ROOT / filename, f"{folder}/{filename}")
    return base64.b64encode(output.getvalue()).decode()


def build(number, previous="", author=None):
    commands = verify(number)
    unit, seed, measured = experiment(number)
    previous_unit = UNITS[min(number - 1, len(UNITS)) - 1].title if number > 1 else ""
    prior = previous or previous_unit or "This exercise starts the series."
    notes = author(unit, prior) if author else {
        "concept": unit.explanation,
        "hint": "Start with the simplest test case and calculate the expected result by hand.",
        "reflection": "What data is available when the decision is made, and what data is reserved for evaluation?",
    }
    interfaces = ", ".join(f"`{fn.__name__}{inspect.signature(fn)}`" for fn in unit.functions)
    next_unit = UNITS[min(number, len(UNITS) - 1)].title
    handoff = f"{unit.title} added {interfaces}. Next exercise: {next_unit}."
    test_command = "python3 -m unittest -v " + " ".join(commands)
    week_notes = "This week's code extends the previous research components." if number <= 12 else (
        "This follow-up varies the synthetic sample and stress scenarios using the capstone harness."
    )
    title = f"Python Trading Practice {number:02d} | {unit.title}"
    markdown = f"""# {title}

{prior}

{week_notes} Allow about 45–60 minutes. The exercise uses Python 3.12 and its standard library.

## Concept

{notes['concept']}

## Coding challenge

{unit.challenge}

Complete {interfaces} in `exercise/trading.py`. Read the test cases for input and output examples.

**Success criteria.** {unit.criteria}

**Hint.** {notes['hint']}

## Bias and overfitting checks

- **Look-ahead.** Perturb future prices, then compare decisions through the current open. The deliberately leaky control must fail the invariance assertion.
- **Overfitting.** Perturb the holdout and check that it cannot change development-only parameter selection. The deliberately contaminated selector must fail. The seeded null experiment also compares the effects of selecting from one versus twenty candidates. Neither check establishes that a strategy generalizes.
- Use the lesson-specific test alongside these pipeline checks.

Run from the extracted `exercise/` folder.

```sh
{test_command}
```

**Reflection.** {notes['reflection']}

## Worked solution

{unit.explanation}

The runnable implementation is in the separate `solution/` folder. Run the same command there to reproduce the test checks, then run this command to reproduce the experiment.

```sh
python3 -c "import json; from curriculum import experiment; print(json.dumps(experiment({number})[2], indent=2))"
```

### Verified on this run

The reviewed Python implementation passed {len(commands)} named test checks for this lesson. The reported experiment ran with synthetic fixture seed {seed} and the parameters in `curriculum.py`. Returns are decimal fractions.

```json
{json.dumps(measured, indent=2, allow_nan=False)}
```

### Assumptions and limits

The synthetic bars model an exercise and do not follow an exchange calendar. Strategy fills are modeled at the open, with whole-sleeve exposure, no market impact, and linear per-turnover costs where applicable. The default is 5 bps in fees plus 5 bps in slippage. The gross buy-and-hold benchmark excludes entry costs. These simulated observations provide no evidence of a live trading edge.

### Handoff

{handoff}
"""
    return dict(title=title, markdown=markdown, bundle=make_bundle(markdown, unit.functions),
                handoff=handoff, seed=seed, measured=measured, checks=commands)
