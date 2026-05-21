import unittest
import sys
import types
from types import SimpleNamespace

if "tenacity" not in sys.modules:
    tenacity = types.ModuleType("tenacity")
    tenacity.retry = lambda *args, **kwargs: (lambda fn: fn)
    tenacity.stop_after_attempt = lambda *args, **kwargs: None
    tenacity.wait_exponential = lambda *args, **kwargs: None
    sys.modules["tenacity"] = tenacity

if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    openai.api_key = None
    sys.modules["openai"] = openai

if "google.generativeai" not in sys.modules:
    google = types.ModuleType("google")
    generativeai = types.ModuleType("google.generativeai")
    generativeai.configure = lambda *args, **kwargs: None
    generativeai.GenerativeModel = object
    generativeai.types = SimpleNamespace(GenerationConfig=object)
    sys.modules.setdefault("google", google)
    sys.modules["google.generativeai"] = generativeai

from models.completion_api import BaseLLMAPI, CompletionAPI, LLMConfig


class DummyLLM(BaseLLMAPI):
    def _make_completion_request(self, prompt, stop=None):
        return prompt

    def _make_chat_request(self, messages, stop=None):
        return messages[0]["content"], ""


class FakeChatCompletions:
    def __init__(self, content):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class TestThinkingModelSupport(unittest.TestCase):
    def make_dummy(self):
        return DummyLLM(LLMConfig(provider="completion_api", model="dummy"))

    def test_extract_complete_thinking_trace(self):
        model = self.make_dummy()

        trace = model._extract_thinking("<think>reason\nstep</think>\nAnswer")

        self.assertEqual(trace, "reason\nstep")

    def test_extract_partial_thinking_trace(self):
        model = self.make_dummy()

        trace = model._extract_thinking("<think>unfinished reasoning")

        self.assertEqual(trace, "unfinished reasoning")

    def test_strip_thinking_before_applying_stop_tokens(self):
        model = self.make_dummy()
        raw = "<think>line one\nline two</think>\nTeacher: visible answer\nextra"

        visible = model._strip_thinking(raw, stop=["\n"])

        self.assertEqual(visible, "visible answer")

    def test_completion_chat_does_not_send_stop_tokens_to_api(self):
        api = CompletionAPI.__new__(CompletionAPI)
        api.config = SimpleNamespace(model="dummy", temperature=0.0, max_tokens=128)
        fake_chat = FakeChatCompletions("<think>reason\nwith stop</think>\nFinal answer\nignored")
        api.client = SimpleNamespace(chat=SimpleNamespace(completions=fake_chat))

        visible, trace = api._make_chat_request(
            [{"role": "system", "content": "prompt"}],
            stop=["\n"],
        )

        self.assertNotIn("stop", fake_chat.kwargs)
        self.assertEqual(trace, "reason\nwith stop")
        self.assertEqual(visible, "Final answer")

    def test_generate_can_return_trace(self):
        class TraceLLM(DummyLLM):
            def _make_chat_request(self, messages, stop=None):
                return "answer", "reason"

        model = TraceLLM(LLMConfig(provider="completion_api", model="dummy", is_chat=True))

        response, trace = model.generate([], "prompt", return_trace=True)

        self.assertEqual(response, "answer")
        self.assertEqual(trace, "reason")


if __name__ == "__main__":
    unittest.main()
