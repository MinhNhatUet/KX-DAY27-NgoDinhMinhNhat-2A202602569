"""Offline integration checks: real graphs/checkpoints, mocked external services."""
import asyncio
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from common.db import replay_events
from common.schemas import PRAnalysis, ReviewComment
from exercises import exercise_1_confidence as e1, exercise_2_hitl as e2
from exercises import exercise_3_escalation as e3, exercise_4_audit as e4

URL = 'https://github.com/example/repo/pull/1'
PR = SimpleNamespace(title='Task priority', diff='diff migration\n' * 300,
                     files_changed=['storage.py'], head_sha='abcdef1234')


def analysis(confidence, questions=None):
    return PRAnalysis(summary='Review summary', confidence=confidence,
                      confidence_reasoning='Migration context is missing',
                      risk_factors=['Missing migration test'],
                      comments=[ReviewComment(file='storage.py', severity='issue', body='Add a test')],
                      escalation_questions=questions or [])


def services(module, outputs, stack):
    fetch = stack.enter_context(patch.object(module, 'fetch_pr', return_value=PR))
    llm = MagicMock()
    llm.with_structured_output.return_value = llm
    llm.invoke.side_effect = list(outputs)
    llm.ainvoke = AsyncMock(side_effect=list(outputs))
    stack.enter_context(patch.object(module, 'get_llm', return_value=llm))
    post = stack.enter_context(patch.object(module, 'post_review_comment')) if module != e1 else None
    return fetch, llm, post


class GraphTests(unittest.TestCase):
    def test_confidence_boundaries_and_terminal_nodes(self):
        for confidence, decision, action in [
            (0.0, 'escalate', 'pending_escalation'),
            (0.5799, 'escalate', 'pending_escalation'),
            (0.58, 'human_approval', 'pending_human_approval'),
            (0.7299, 'human_approval', 'pending_human_approval'),
            (0.73, 'auto_approve', 'auto_approved'),
            (1.0, 'auto_approve', 'auto_approved'),
        ]:
            with self.subTest(confidence=confidence), ExitStack() as stack:
                _, llm, _ = services(e1, [analysis(confidence)], stack)
                result = e1.build_graph().invoke({'pr_url': URL})
                self.assertEqual(result['decision'], decision)
                self.assertEqual(result['final_action'], action)
                llm.with_structured_output.assert_called_once_with(PRAnalysis)

    def test_approval_pauses_and_posts_only_once_on_approve(self):
        for choice in ['approve', 'reject', 'edit']:
            with self.subTest(choice=choice), ExitStack() as stack:
                fetch, llm, post = services(e2, [analysis(.65)], stack)
                app = e2.build_graph()
                cfg = {'configurable': {'thread_id': choice}}
                result = app.invoke({'pr_url': URL}, cfg)
                payload = result['__interrupt__'][0].value
                self.assertEqual(payload['kind'], 'approval_request')
                self.assertEqual(len(payload['diff_preview']), 2000)
                self.assertEqual(payload['comments'][0]['file'], 'storage.py')
                post.assert_not_called()
                final = app.invoke(Command(resume={'choice': choice, 'feedback': 'test feedback'}), cfg)
                self.assertEqual(final['final_action'], 'committed' if choice == 'approve' else 'rejected')
                self.assertEqual(post.call_count, int(choice == 'approve'))
                fetch.assert_called_once()
                llm.invoke.assert_called_once()

    def test_escalation_refines_with_original_analysis_and_answers(self):
        with ExitStack() as stack:
            _, llm, post = services(e3, [analysis(.4, ['Why MD5 in auth.py?']), analysis(.8)], stack)
            app = e3.build_graph()
            cfg = {'configurable': {'thread_id': 'escalation'}}
            result = app.invoke({'pr_url': URL}, cfg)
            self.assertEqual(result['__interrupt__'][0].value['questions'], ['Why MD5 in auth.py?'])
            post.assert_not_called()
            final = app.invoke(Command(resume={'Why MD5 in auth.py?': 'Legacy migration only'}), cfg)
            self.assertEqual(final['final_action'], 'committed_after_escalation')
            prompt = llm.invoke.call_args.args[0][1]['content']
            for expected in [PR.diff, 'Initial analysis', 'Legacy migration only']:
                self.assertIn(expected, prompt)
            self.assertIn('Reviewer answered escalation questions', post.call_args.args[1])
            post.assert_called_once()

    def test_sqlite_resume_after_reopen_and_audit(self):
        async def scenario(path, confidence, choice):
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, HITL_DB_PATH=path, GITHUB_USER='test-reviewer'))
                fetch, llm, post = services(e4, [analysis(confidence, ['Why MD5?']), analysis(.8)], stack)
                cfg = {'configurable': {'thread_id': choice}}
                async with AsyncSqliteSaver.from_conn_string(path) as cp:
                    await cp.setup()
                    result = await e4.build_graph(cp).ainvoke({'pr_url': URL, 'thread_id': choice}, cfg)
                if confidence < .73:
                    self.assertIn('__interrupt__', result)
                    post.assert_not_called()
                    async with AsyncSqliteSaver.from_conn_string(path) as cp:
                        graph = e4.build_graph(cp)
                        snapshot = await graph.aget_state(cfg)
                        self.assertTrue(snapshot.tasks[0].interrupts)
                        answer = {'Why MD5?': 'Legacy migration'} if choice == 'escalate' else {'choice': choice, 'feedback': 'Reviewed'}
                        result = await graph.ainvoke(Command(resume=answer), cfg)
                events = await replay_events(choice)
                self.assertEqual(events[0]['action'], 'fetch_pr')
                self.assertEqual(events[0]['risk_level'], 'med')
                self.assertEqual(events[1]['decision'], 'pending')
                self.assertFalse(any(e['risk_level'] == 'high' and e['decision'] == 'auto' for e in events))
                self.assertTrue(all(e['execution_time_ms'] >= 0 for e in events))
                fetch.assert_called_once()
                if choice == 'auto':
                    self.assertTrue(all(e['reviewer_id'] is None for e in events))
                else:
                    node = 'escalate' if choice == 'escalate' else 'human_approval'
                    rows = [e for e in events if e['action'] == node]
                    self.assertEqual([e['decision'] for e in rows], ['pending', 'pending', choice])
                    self.assertEqual(rows[-1]['reviewer_id'], 'test-reviewer')
                if choice == 'escalate':
                    refined = next(e for e in events if e['action'] == 'synthesize')
                    self.assertEqual(refined['confidence'], .8)
                    self.assertEqual(refined['risk_level'], 'low')
                self.assertEqual(post.call_count, int(choice not in ['reject', 'edit']))
        for confidence, choice in [(.65, 'approve'), (.65, 'reject'), (.65, 'edit'), (.4, 'escalate'), (.9, 'auto')]:
            with self.subTest(choice=choice), tempfile.TemporaryDirectory() as tmp:
                asyncio.run(scenario(str(Path(tmp) / 'audit.db'), confidence, choice))


if __name__ == '__main__':
    unittest.main()
