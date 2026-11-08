# Python Trading Practice

A weekly Python exercise series for learning how to build and evaluate systematic trading strategies. Each 45–60 minute lesson includes a coding challenge, starter code, measurable success criteria, tests, and a worked solution.

## Curriculum

| Weeks | Topics |
| --- | --- |
| 1–4 | Validate OHLCV bars, build a momentum signal, model trading costs, and reserve a chronological holdout. |
| 5–8 | Run walk-forward tests, estimate uncertainty with a block bootstrap, measure selection bias, and stress-test costs and market regimes. |
| 9–12 | Combine assets, test execution delays, examine nearby parameters, and run a joint robustness experiment. |
| 13 onward | Repeat the capstone with new synthetic datasets and stress scenarios. |

The tests change future prices to check that earlier trading decisions stay fixed. They also perturb holdout data to check that it cannot influence parameter selection and use a simulated null to show how searching more candidates can inflate in-sample results. Lessons report measured results separately from assumptions about fills and costs. Synthetic results cannot establish a live trading edge.

## Project status

This import includes the tested Python source, root Render Blueprint, and Week 1 example. All 19 tests passed on Python 3.12.14, and the Week 1 preview was generated and checked for separate starter and worked-solution folders. Live deployment and email delivery remain unverified.

## Run locally

Use Python 3.12. The tests and local preview use the standard library.

```sh
python3 -m unittest -v
python3 job.py --preview 1
```

The preview writes a Markdown lesson and a ZIP containing separate starter and solution folders to `preview/`. Replace `1` with the lesson number you want to preview.

## Deploy on Render

Connect this repository as a Render Blueprint using the root `render.yaml`. The cron job is configured for Tuesdays at **8 a.m. Eastern**. It runs the reviewed Python reference code and tests, uses the OpenAI Responses API for teaching notes, saves the lesson and its test results in Neon, and delivers it by email through Resend.

Render cron uses UTC. The Blueprint runs at 12:00, 12:30, 13:00, and 13:30 UTC on Tuesdays; `job.py` admits only the 8 a.m. America/New_York window, accounting for daylight saving time. The half-hour run provides a retry opportunity, and persisted delivery IDs prevent a second accepted send.

Set these values in the Render service environment.

| Variable | Value |
| --- | --- |
| `DATABASE_URL` | Direct Neon Postgres connection URL with TLS |
| `OPENAI_API_KEY` | OpenAI project API key |
| `OPENAI_MODEL` | Start with `gpt-6-luna`; confirm project access and Structured Outputs support |
| `RESEND_API_KEY` | Resend sending key |
| `LESSON_FROM` | Sender address on a verified domain |
| `LESSON_TO` | Recipient email address |

Verify the first run in Render and confirm that the email arrives with the lesson ZIP attached. Keep credentials in the service environment.

## License

[MIT](LICENSE)
