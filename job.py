"""Run the Render cron job or generate a local, credential-free preview."""
import argparse
import base64
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from lesson import build, post_json, teach

EASTERN = ZoneInfo("America/New_York")
LOCK = 767262920261110
SCHEMA = """
CREATE TABLE IF NOT EXISTS python_trading_practice (
    lesson_date date PRIMARY KEY,
    lesson_number integer UNIQUE NOT NULL,
    artifact jsonb NOT NULL,
    envelope jsonb NOT NULL,
    attempted_at timestamptz,
    provider_id text
)
"""


def scheduled_date(now):
    local = now.astimezone(EASTERN)
    return local.date() if local.weekday() == 1 and local.hour == 8 else None


def email_payload(artifact, sender, recipient):
    return {
        "from": sender,
        "to": [recipient],
        "subject": artifact["title"],
        "text": artifact["markdown"],
        "attachments": [{"filename": "Python_Trading_Practice.zip", "content": artifact["bundle"]}],
    }


def run_job(connection, day, sender, recipient, author, deliver):
    # Session lock stays held across commits, including the outbound call.
    # Use a direct Neon connection because pooler transactions can move sessions.
    connection.execute("SELECT pg_advisory_lock(%s)", (LOCK,))
    connection.commit()
    try:
        return _run_job(connection, day, sender, recipient, author, deliver)
    finally:
        connection.rollback()
        connection.execute("SELECT pg_advisory_unlock(%s)", (LOCK,))
        connection.commit()


def _run_job(connection, day, sender, recipient, author, deliver):
    connection.execute(SCHEMA)
    row = connection.execute(
        "SELECT lesson_number, artifact, envelope, attempted_at, provider_id "
        "FROM python_trading_practice WHERE lesson_date = %s", (day,),
    ).fetchone()
    if row:
        number, artifact, envelope, attempted_at, provider_id = row
        if provider_id:
            return f"Lesson {number} was already accepted for delivery."
        if attempted_at and datetime.now(timezone.utc) - attempted_at >= timedelta(hours=22):
            raise RuntimeError("Check the previous delivery with the email provider before resending")
    else:
        previous = connection.execute(
            "SELECT lesson_date, lesson_number, artifact->>'handoff', provider_id "
            "FROM python_trading_practice ORDER BY lesson_date DESC LIMIT 1"
        ).fetchone()
        if previous and (day <= previous[0] or previous[3] is None):
            raise RuntimeError("Resolve the last lesson's delivery before creating another lesson")
        number = previous[1] + 1 if previous else 1
        artifact = build(number, previous[2] if previous else "", author)
        envelope = email_payload(artifact, sender, recipient)
        connection.execute(
            "INSERT INTO python_trading_practice (lesson_date, lesson_number, artifact, envelope) "
            "VALUES (%s, %s, %s::jsonb, %s::jsonb)",
            (day, number, json.dumps(artifact), json.dumps(envelope)),
        )

    connection.execute(
        "UPDATE python_trading_practice SET attempted_at = COALESCE(attempted_at, now()) "
        "WHERE lesson_date = %s", (day,),
    )
    connection.commit()
    provider_id = deliver(envelope, f"python-trading-practice/{day.isoformat()}")
    if not provider_id:
        raise RuntimeError("The email provider returned no message ID")
    connection.execute(
        "UPDATE python_trading_practice SET provider_id = %s WHERE lesson_date = %s",
        (provider_id, day),
    )
    connection.commit()
    return f"Lesson {number} was accepted by the email provider."


def preview(number, output):
    artifact = build(number)
    output.mkdir(parents=True, exist_ok=True)
    (output / "Lesson.md").write_text(artifact["markdown"], encoding="utf-8")
    (output / "Python_Trading_Practice.zip").write_bytes(base64.b64decode(artifact["bundle"]))
    print(f"Verified preview written to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=int, metavar="WEEK")
    parser.add_argument("--output", type=Path, default=Path("preview"))
    parser.add_argument("--date", type=date.fromisoformat, help="Retry or run a specific Tuesday")
    args = parser.parse_args()
    if args.preview is not None:
        if args.date is not None:
            parser.error("--preview and --date are mutually exclusive")
        if args.preview < 1:
            parser.error("--preview requires a positive week number")
        preview(args.preview, args.output)
        return

    day = args.date or scheduled_date(datetime.now(timezone.utc))
    if day is None:
        print("Outside the Tuesday 8 a.m. Eastern delivery window.")
        return
    if day.weekday() != 1:
        parser.error("--date must be a Tuesday")

    required = ("DATABASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "RESEND_API_KEY",
                "LESSON_FROM", "LESSON_TO")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        parser.error("Set these environment variables: " + ", ".join(missing))
    if "-pooler." in os.environ["DATABASE_URL"]:
        parser.error("DATABASE_URL must be a direct Neon connection")
    import psycopg

    def author(unit, previous):
        return teach(unit, previous, os.environ["OPENAI_API_KEY"], os.environ["OPENAI_MODEL"])

    def deliver(payload, key):
        result = post_json("https://api.resend.com/emails", os.environ["RESEND_API_KEY"],
                           payload, {"Idempotency-Key": key})
        return result.get("id")

    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        print(run_job(connection, day, os.environ["LESSON_FROM"], os.environ["LESSON_TO"],
                      author, deliver))


if __name__ == "__main__":
    main()
