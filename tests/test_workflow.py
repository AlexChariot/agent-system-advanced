import importlib
import sys
import types
import unittest
from unittest.mock import patch


langchain_ollama = types.ModuleType("langchain_ollama")


class _PlaceholderChatOllama:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, _messages):
        raise NotImplementedError


langchain_ollama.ChatOllama = _PlaceholderChatOllama
sys.modules.setdefault("langchain_ollama", langchain_ollama)

langchain_core = types.ModuleType("langchain_core")
langchain_core_messages = types.ModuleType("langchain_core.messages")


class _HumanMessage:
    def __init__(self, content: str):
        self.content = content


langchain_core_messages.HumanMessage = _HumanMessage
sys.modules.setdefault("langchain_core", langchain_core)
sys.modules.setdefault("langchain_core.messages", langchain_core_messages)

analyst = importlib.import_module("agent_system.agents.analyst").analyst
manager = importlib.import_module("agent_system.agents.manager").manager


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, _messages):
        return _FakeResponse(self._content)


class WorkflowRoutingTests(unittest.TestCase):
    def test_manager_sends_last_researched_task_to_analyst(self):
        state = {
            "plan": [],
            "current_task": "Final task",
            "research": "Fresh research",
            "analysis": "",
            "result": "",
            "evaluation": "",
            "completed_tasks": [],
            "selected_model": "llama3.1",
        }

        self.assertEqual(manager(state)["next_agent"], "analyst")

    def test_manager_requests_next_research_after_prior_task_analysis(self):
        state = {
            "plan": ["Task 2"],
            "current_task": "Task 2",
            "research": "",
            "analysis": "Task: Task 1\nInsight",
            "result": "",
            "evaluation": "",
            "completed_tasks": ["Task 1"],
            "selected_model": "llama3.1",
        }

        self.assertEqual(manager(state)["next_agent"], "researcher")

    def test_manager_resets_failed_result_before_retry(self):
        state = {
            "plan": [],
            "current_task": None,
            "research": "",
            "analysis": "Task: Task 1\nInsight",
            "result": "Outdated result",
            "evaluation": "NO",
            "completed_tasks": ["Task 1"],
            "selected_model": "llama3.1",
        }

        update = manager(state)

        self.assertEqual(update["next_agent"], "executor")
        self.assertEqual(update["result"], "")
        self.assertEqual(update["evaluation"], "")


class AnalystProgressionTests(unittest.TestCase):
    @patch("agent_system.agents.analyst.ChatOllama", return_value=_FakeLLM("Insight A"))
    def test_analyst_advances_to_next_task_and_accumulates_analysis(self, _mock_llm):
        state = {
            "research": "Research for task 1",
            "current_task": "Task 1",
            "plan": ["Task 1", "Task 2"],
            "completed_tasks": [],
            "analysis": "",
            "selected_model": "llama3.1",
        }

        update = analyst(state)

        self.assertEqual(update["plan"], ["Task 2"])
        self.assertEqual(update["current_task"], "Task 2")
        self.assertEqual(update["completed_tasks"], ["Task 1"])
        self.assertEqual(update["research"], "")
        self.assertIn("Task: Task 1", update["analysis"])
        self.assertIn("Insight A", update["analysis"])

    @patch("agent_system.agents.analyst.ChatOllama", return_value=_FakeLLM("Insight B"))
    def test_analyst_exhausts_plan_then_leaves_aggregated_analysis_for_executor(self, _mock_llm):
        state = {
            "research": "Research for task 2",
            "current_task": "Task 2",
            "plan": ["Task 2"],
            "completed_tasks": ["Task 1"],
            "analysis": "Task: Task 1\nInsight A",
            "selected_model": "llama3.1",
        }

        update = analyst(state)

        self.assertEqual(update["plan"], [])
        self.assertIsNone(update["current_task"])
        self.assertEqual(update["completed_tasks"], ["Task 1", "Task 2"])
        self.assertIn("Task: Task 1", update["analysis"])
        self.assertIn("Task: Task 2", update["analysis"])
        self.assertIn("Insight B", update["analysis"])


if __name__ == "__main__":
    unittest.main()
