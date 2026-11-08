"""Check scheduling, the offline artifact, and the OpenAI response contract."""
import ast
import base64
from datetime import date, datetime, timezone
from io import BytesIO
import json
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

from curriculum import UNITS
from job import email_payload, run_job, scheduled_date
from lesson import build, teach


class JobTests(unittest.TestCase):
    def test_eastern_schedule_across_daylight_saving(self):
        for month, day, scheduled_hour in ((1, 6, 13), (7, 7, 12), (11, 10, 13)):
            expected = date(2026, month, day)
            for minute in (0, 30):
                self.assertEqual(scheduled_date(
                    datetime(2026, month, day, scheduled_hour, minute, tzinfo=timezone.utc)),
                    expected)
            self.assertIsNone(scheduled_date(
                datetime(2026, month, day, 25 - scheduled_hour, tzinfo=timezone.utc)))
            self.assertIsNone(scheduled_date(
                datetime(2026, month, day + 1, scheduled_hour, tzinfo=timezone.utc)))

    def test_lesson_bundle_keeps_starter_and_solution_separate(self):
        lesson = build(1)
        self.assertIn("Verified on this run", lesson["markdown"])
        self.assertEqual(lesson["measured"]["split_sizes"], [160, 80, 81])
        with ZipFile(BytesIO(base64.b64decode(lesson["bundle"]))) as bundle:
            self.assertIsNone(bundle.testzip())
            starter = ast.parse(bundle.read("exercise/trading.py"))
            reference = ast.parse(bundle.read("solution/trading.py"))
            for name in ("validate", "close_returns", "split"):
                stub = next(fn for fn in starter.body if getattr(fn, "name", None) == name)
                answer = next(fn for fn in reference.body if getattr(fn, "name", None) == name)
                self.assertIsInstance(stub.body[0], ast.Raise)
                self.assertNotEqual(ast.dump(stub), ast.dump(answer))
            self.assertEqual(bundle.read("Lesson.md").decode(), lesson["markdown"])
        message = email_payload(lesson, "teacher@example.com", "student@example.com")
        self.assertEqual(message["from"], "teacher@example.com")
        self.assertEqual(message["to"], ["student@example.com"])
        self.assertEqual(message["attachments"][0]["content"], lesson["bundle"])

    def test_job_persists_before_delivery_and_reuses_saved_envelope(self):
        day = date(2026, 11, 10)
        connection = Mock()
        artifact = {"title": "Week 1", "markdown": "Lesson", "bundle": "emlw", "handoff": "Next"}
        expected = email_payload(artifact, "teacher@example.com", "student@example.com")
        connection.execute.return_value.fetchone.side_effect = [None, None]

        def fail_after_sending(payload, key):
            self.assertEqual(connection.commit.call_count, 2)
            self.assertEqual(payload, expected)
            self.assertEqual(key, "python-trading-practice/2026-11-10")
            raise RuntimeError("response lost")

        with patch("job.build", return_value=artifact) as builder:
            with self.assertRaisesRegex(RuntimeError, "response lost"):
                run_job(connection, day, expected["from"], expected["to"][0], None,
                        fail_after_sending)
            inserted = next(call.args[1] for call in connection.execute.call_args_list
                            if call.args[0].startswith("INSERT"))
            self.assertEqual(json.loads(inserted[3]), expected)
            connection.reset_mock()
            connection.execute.return_value.fetchone.side_effect = [
                (1, artifact, expected, datetime.now(timezone.utc), None)
            ]
            deliver = Mock(return_value="accepted-message")
            run_job(connection, day, "new@example.com", "new@example.com", None, deliver)
            deliver.assert_called_once_with(expected, "python-trading-practice/2026-11-10")
            builder.assert_called_once()

    def test_already_accepted_lesson_is_not_sent_again(self):
        connection = Mock()
        connection.execute.return_value.fetchone.return_value = (1, {}, {}, None, "accepted")
        deliver = Mock()
        response = run_job(connection, date(2026, 11, 10), "", "", None, deliver)
        self.assertIn("already accepted", response)
        deliver.assert_not_called()

    @patch("lesson.post_json")
    def test_openai_notes_use_previous_handoff_and_strict_schema(self, post):
        notes = {"concept": "Concept", "hint": "Hint", "reflection": "Reflection?"}
        post.return_value = {"status": "completed", "output": [
            {"type": "reasoning"},
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(notes)}]},
        ]}
        self.assertEqual(teach(UNITS[1], "Earlier work", "fake-key", "configured-model"), notes)
        url, _, payload = post.call_args.args
        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertEqual(json.loads(payload["input"])["previous_handoff"], "Earlier work")
        self.assertFalse(payload["store"])
        self.assertTrue(payload["text"]["format"]["strict"])

    @patch("lesson.post_json", return_value={"status": "incomplete"})
    def test_incomplete_model_output_fails_before_delivery(self, _):
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            teach(UNITS[0], "", "fake-key", "configured-model")


if __name__ == "__main__":
    unittest.main()
