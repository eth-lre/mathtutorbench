from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import jinja2
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
from openai import OpenAI, api_key
from enum import Enum


class ProviderType(Enum):
    COMPLETION_API = "completion_api"
    OLLAMA = "ollama"
    GEMINI = "gemini"


@dataclass
class LLMConfig:
    provider: str  # Will be converted to ProviderType
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None  # For Gemini
    temperature: float = 0.0
    max_tokens: int = 8196
    is_chat: bool = False
    is_thinking: bool = False
    enable_thinking: bool = False
    prompt_as_user: bool = False
    max_retries: int = 3

    def __post_init__(self):
        self.provider = ProviderType(self.provider.lower())
        if self.is_thinking and self.enable_thinking:
            raise ValueError("is_thinking (suppress thinking) and enable_thinking are mutually exclusive")
        # Set default base_url for Ollama if not provided
        if self.provider == ProviderType.OLLAMA and not self.base_url:
            raise ValueError("base_url is required for ollama provider")

        # Validate configuration
        # if self.provider == ProviderType.COMPLETION_API and not self.base_url:
        #     raise ValueError("base_url is required for CompletionAPI provider")
        if self.provider == ProviderType.GEMINI and not self.api_key:
            raise ValueError("api_key is required for Gemini provider")


THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
NOTHINK = f"{THINK_OPEN_TAG}\n\n{THINK_CLOSE_TAG}\n\n"


def strip_thinking(text: str) -> str:
    """Drop the reasoning trace from a thinking model's output.

    Qwen-style chat templates open the block in the prompt itself, so the
    generation usually contains only the closing tag: everything up to the
    last one is reasoning and the answer follows it.
    """
    if not text:
        return text
    if THINK_CLOSE_TAG in text:
        return text.rpartition(THINK_CLOSE_TAG)[2].strip()
    return text.strip()


def truncate_at_stop(text: str, stop: Optional[List[str]] = None) -> str:
    """Cut text at the earliest stop string, mimicking server-side stop."""
    if not text or not stop:
        return text
    cut = len(text)
    for stop_str in stop:
        index = text.find(stop_str)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].strip()


def create_llm_model(config: LLMConfig) -> 'BaseLLMAPI':
    """Factory function to create LLM model instance based on provider"""
    if config.provider == ProviderType.COMPLETION_API:
        return CompletionAPI(config)
    elif config.provider == ProviderType.OLLAMA:
        return OllamaAPI(config)
    elif config.provider == ProviderType.GEMINI:
        return GeminiAPI(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


class BaseLLMAPI(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format messages into a conversation string"""
        # print("Formatting conversation from messages: " + "|EOM|".join(messages))
        conversation = ""
        for msg in messages:
            role = msg["user"]
            content = msg["text"]
            conversation += f"{role}: {content}\n"

        formatted_conversation = conversation.strip()
        # print("Formatted conversation: " + formatted_conversation)
        return formatted_conversation

    def _format_prompt(self, system_prompt: str, messages: List[Dict]) -> str:
        """Format prompt for non-chat completions"""
        # print("Formatting prompt with system_prompt: " + system_prompt)
        conversation = self._format_conversation(messages)

        if "{{conversation}}" in system_prompt:
            template = jinja2.Template(system_prompt)
            prompt = template.render(conversation=conversation)
        else:
            prompt = system_prompt

        # print("Final formatted prompt: " + prompt)
        return prompt

    def _format_chat_messages(self, system_prompt: str, messages: List[Dict]) -> List[Dict]:
        """Format messages for chat completions"""
        # print("Formatting chat messages with system_prompt: " + system_prompt)
        # Tasks put the whole prompt in the system message and send no user turn, which
        # some chat templates (e.g. Qwen3) reject with "No user query found in messages".
        role = "user" if self.config.prompt_as_user else "system"
        formatted_messages = [
            {"role": role, "content": system_prompt}
        ]
        formatted_messages.extend(messages)
        # print("Formatted chat messages: " + str(formatted_messages))
        return formatted_messages

    @abstractmethod
    def _make_completion_request(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Make a completion request - to be implemented by specific providers"""
        pass

    @abstractmethod
    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> str:
        """Make a chat request - to be implemented by specific providers"""
        pass

    def generate(
            self,
            messages: List[Dict],
            system_prompt: str,
            stop: Optional[List[str]] = None,
    ) -> str:
        """Generate completion using either chat or completion API"""
        print("==============================================================")
        print("Generating completion with model: " + self.config.model)

        try:
            if self.config.is_chat:
                formatted_messages = self._format_chat_messages(system_prompt, messages)
                return self._make_chat_request(formatted_messages, stop)
            else:
                prompt = self._format_prompt(system_prompt, messages)
                return self._make_completion_request(prompt, stop)
        except Exception as e:
            print("Failed to generate completion after retries: " + str(e))
            raise


class CompletionAPI(BaseLLMAPI):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if config.base_url:
            openai_api_key = "EMPTY" if config.api_key is None else config.api_key
            print(f"Using base_url: {config.base_url} and api_key: {openai_api_key}")
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=config.base_url,
            )
            models = self.client.models.list()
            print(f"Available models {models}")
            # pick the model with the config.model
            if config.model is None:
                # pick the first available one
                self.config.model = models.data[0].id
            else:
                # pick based on the name
                self.config.model = next((model.id for model in models.data if config.model in model.id), None)
            if self.config.model is None:
                raise ValueError(f"Model {config.model} not found in {models}")
        else:
            self.client = OpenAI(api_key=config.api_key)

    def _postprocess_thinking(
            self,
            completion: str,
            stop: Optional[List[str]] = None,
            finish_reason: Optional[str] = None,
            trace_already_split: bool = False,
    ) -> str:
        """Remove the reasoning trace and apply stop strings to the answer."""
        # A reasoning parser hands back the answer already split from the trace, and
        # sends None instead of a string when the trace consumed the whole generation.
        completion = completion or ""
        # With the trace split out a missing closing tag no longer signals truncation,
        # so fall back to what the server reports.
        if finish_reason == "length" or (not trace_already_split and THINK_CLOSE_TAG not in completion):
            print(f"Warning: enable_thinking is set but the response looks truncated "
                  f"(finish_reason={finish_reason}). The model likely ran out of budget while "
                  f"reasoning - consider raising max_tokens.")
        answer = completion.strip() if trace_already_split else strip_thinking(completion)
        answer = truncate_at_stop(answer, stop)
        if not answer:
            print("Warning: empty answer left after removing the reasoning trace - this example "
                  "will score as an empty prediction.")
        return answer

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_completion_request(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # print("Making completion request with prompt: " + prompt)
            print("========================(Prompt-completion-start)======================================")
            print(prompt)
            print("========================(Prompt-completion-end)======================================")
            # /v1/completions applies no chat template, so chat_template_kwargs has no
            # effect here - thinking is steered by what we prefill onto the prompt. An
            # empty block suppresses it, an open tag forces the model into it.
            if self.config.is_thinking:
                prompt = prompt + NOTHINK
            elif self.config.enable_thinking:
                prompt = prompt + THINK_OPEN_TAG# + "\n"
            # Stop strings are applied after the trace is removed, since they would
            # otherwise fire inside the reasoning.
            request_stop = None if self.config.enable_thinking else stop
            response = self.client.completions.create(
                model=self.config.model,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stop=request_stop
            )
            choice = response.choices[0]
            completion = (choice.text or "").strip()

            # print("Received completion response: " + completion)
            print("===========================(Response-completion-start)===================================")
            if self.config.enable_thinking:
                print("Stripping thinking:")
                print(completion)
                completion = self._postprocess_thinking(completion, stop, choice.finish_reason)
            print(completion)
            print("===========================(Response-completion-end)===================================")
            return completion
        except Exception as e:
            print("Error in completion request: " + str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> str:
        try:
            # print("Making chat request with messages: " + str(messages))
            print("========================(Prompt-chat-start)======================================")
            # print(messages)
            print(messages[0]["content"])
            print("========================(Prompt-chat-end)======================================")
            extra_body = {}
            request_stop = stop
            if self.config.is_thinking:
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            elif self.config.enable_thinking:
                extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
                request_stop = None
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stop=request_stop,
                extra_body=extra_body
            )
            choice = response.choices[0]
            completion = choice.message.content
            # A server started with --reasoning-parser splits the trace into its own
            # field; without one the whole trace stays inline in the content.
            reasoning = getattr(choice.message, "reasoning_content", None) or \
                        getattr(choice.message, "reasoning", None)
            if self.config.enable_thinking:
                completion = self._postprocess_thinking(
                    completion, stop, choice.finish_reason, trace_already_split=bool(reasoning)
                )
            elif reasoning and not (completion or "").strip():
                print("Warning: the reasoning parser returned an empty answer with a non-empty "
                      "reasoning field while thinking is disabled - reading the answer from the "
                      "reasoning field instead.")
                completion = truncate_at_stop(strip_thinking(reasoning), stop)
            print("===========================(Response-chat-start)===================================")
            print(completion)
            print("===========================(Response-chat-end)===================================")
            # print("Received chat response: " + completion)
            return completion
        except Exception as e:
            print("Error in chat request: " + str(e))
            raise


class OllamaAPI(BaseLLMAPI):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_completion_request(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Make a completion request to Ollama"""
        try:
            print("========================(Prompt-start)======================================")
            print(prompt)
            print("========================(Prompt-end)======================================")
            response = requests.post(
                f"{self.base_url}",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                        "stop": stop or []
                    },
                    "stream": False
                }
            )
            response.raise_for_status()
            completion = response.json()["response"].strip()
            print("===========================(Response-start)===================================")
            print(completion)
            print("===========================(Response-end)===================================")
            return completion
        except Exception as e:
            print("Error in completion request: " + str(e))
            raise

    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> str:
        """Make a chat request to Ollama"""
        # Ollama doesn't have a separate chat endpoint, so we'll format messages into a prompt
        formatted_prompt = ""
        for message in messages:
            formatted_prompt += f"{message['role']}: {message['content']}\nassistant: "

        return self._make_completion_request(formatted_prompt, stop)


class GeminiAPI(BaseLLMAPI):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_completion_request(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            print("========================(Prompt-start)======================================")
            print(prompt)
            print("========================(Prompt-end)======================================")
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    stop_sequences=stop or []
                )
            )
            completion = response.text
            print("===========================(Response-start)===================================")
            print(completion)
            print("===========================(Response-end)===================================")
            return completion
        except Exception as e:
            print("Error in completion request: " + str(e))
            raise

    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> str:
        print("========================(Prompt-chat-start)======================================")
        # print(messages)
        print(messages[0]["content"])
        print("========================(Prompt-chat-end)======================================")
        chat = self.model.start_chat()
        # for message in messages:
        if messages[0]["role"] in ("system", "user"):
            # Add system prompt as first user message
            response = chat.send_message(messages[0]["content"], stream=False,
                                         generation_config=genai.types.GenerationConfig(
                                             temperature=self.config.temperature,
                                             max_output_tokens=self.config.max_tokens,
                                             stop_sequences=stop or []
                                         ))
        else:
            # chat.send_message(message["content"], role=message["role"])
            raise ValueError("No system prompt")

        completion = response.text
        print("===========================(Response-chat-start)===================================")
        print(completion)
        print("===========================(Response-chat-end)===================================")
        return completion
