"""Exercise 5 UI integration, without GitHub writes or LLM requests."""
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from test_workflows import URL, analysis, services, e4


class StreamlitTests(unittest.TestCase):
    def test_review_buttons_and_escalation_form(self):
        for confidence, choice in [(.65, 'Approve'), (.65, 'Reject'), (.65, 'Edit'), (.4, 'Submit answers'), (.9, None)]:
            with self.subTest(choice=choice), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, HITL_DB_PATH=str(Path(tmp) / 'ui.db')))
                _, _, post = services(e4, [analysis(confidence, ['Why MD5 in auth.py?']), analysis(.8)], stack)
                at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=20).run()
                self.assertFalse(at.exception)
                at.text_input[0].set_value(URL)
                next(b for b in at.button if b.label == 'Run review').click().run()
                self.assertFalse(at.exception)
                if choice:
                    post.assert_not_called()
                    if choice == 'Submit answers':
                        next(t for t in at.text_input if t.label == 'Why MD5 in auth.py?').set_value('Legacy migration')
                    next(b for b in at.button if b.label == choice).click().run()
                    self.assertFalse(at.exception)
                if choice in ['Reject', 'Edit']:
                    self.assertTrue(at.warning)
                    post.assert_not_called()
                else:
                    self.assertTrue(at.success)
                    self.assertEqual(at.metric[0].label, 'Confidence')
                    post.assert_called_once()
                self.assertTrue(at.dataframe)

    def test_select_saved_session_after_fresh_page(self):
        from streamlit.proto.WidgetStates_pb2 import WidgetStates
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, HITL_DB_PATH=str(Path(tmp) / 'ui.db')))
            fetch, _, post = services(e4, [analysis(.4, ['Why MD5?']), analysis(.8)], stack)
            app_path = str(Path(__file__).resolve().parents[1] / 'app.py')
            first = AppTest.from_file(app_path, default_timeout=20).run()
            first.text_input[0].set_value(URL)
            next(b for b in first.button if b.label == 'Run review').click().run()
            thread = first.session_state.thread_id
            fresh = AppTest.from_file(app_path, default_timeout=20).run()
            self.assertEqual(fresh.dataframe[0].value.iloc[0]['worst_risk'], 'high')
            widgets = WidgetStates()
            selection = widgets.widgets.add()
            selection.id = fresh.dataframe[0].proto.id
            selection.string_value = '{"selection":{"rows":[0],"columns":[],"cells":[]}}'
            fresh._run(widgets)
            self.assertFalse(fresh.exception)
            self.assertEqual(fresh.session_state.thread_id, thread)
            self.assertEqual(fresh.session_state.interrupt_payload['kind'], 'escalation')
            next(t for t in fresh.text_input if t.label == 'Why MD5?').set_value('Legacy migration')
            next(b for b in fresh.button if b.label == 'Submit answers').click().run()
            self.assertFalse(fresh.exception)
            self.assertTrue(fresh.success)
            fetch.assert_called_once()
            post.assert_called_once()

    def test_failed_post_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, HITL_DB_PATH=str(Path(tmp) / 'ui.db')))
            _, _, post = services(e4, [analysis(.9)], stack)
            post.side_effect = RuntimeError('GitHub unavailable')
            at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=20).run()
            at.text_input[0].set_value(URL)
            next(b for b in at.button if b.label == 'Run review').click().run()
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            self.assertFalse(at.success)


if __name__ == '__main__':
    unittest.main()
