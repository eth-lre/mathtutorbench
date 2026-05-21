from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import jinja2
import re
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
    max_tokens: int = 2048
    is_chat: bool = False
    max_retries: int = 3

    def __post_init__(self):
        self.provider = ProviderType(self.provider.lower())
        # Set default base_url for Ollama if not provided
        if self.provider == ProviderType.OLLAMA and not self.base_url:
            raise ValueError("base_url is required for ollama provider")

        # Validate configuration
        # if self.provider == ProviderType.COMPLETION_API and not self.base_url:
        #     raise ValueError("base_url is required for CompletionAPI provider")
        if self.provider == ProviderType.GEMINI and not self.api_key:
            raise ValueError("api_key is required for Gemini provider")


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

    def _extract_thinking(self, response: str) -> str:
        """Extract hidden reasoning from <think> blocks, if present."""
        match = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
        if match:
            return match.group(1).strip()

        # Preserve partial traces when generation is cut off before </think>.
        match = re.search(r'<think>(.*?)$', response, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

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

    def _strip_thinking(self, response: str, stop: Optional[List[str]] = None) -> str:
        """Remove <think> blocks before applying task stop tokens.

        Chat reasoning models can emit a reasoning block before the visible answer.
        Completion-style stop tokens such as "\\n" or "\\n\\n" can appear inside
        that reasoning block, so passing them directly to chat completion APIs may
        truncate the output before the answer is generated. For chat mode, callers
        should request the full response, strip reasoning locally, then apply stop
        tokens to the visible answer only.
        """
        text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL).strip()

        # Some chat models echo completion-style role prefixes from prompts.
        text = re.sub(r'^Teacher\s*\([^)]*\)\s*:\s*', '', text)
        text = re.sub(r'^Teacher\s*:\s*', '', text).strip()

        if stop:
            for token in stop:
                idx = text.find(token)
                if idx != -1:
                    text = text[:idx]
        return text.strip()

    def _format_chat_messages(self, system_prompt: str, messages: List[Dict]) -> List[Dict]:
        """Format messages for chat completions"""
        # print("Formatting chat messages with system_prompt: " + system_prompt)
        formatted_messages = [
            {"role": "system", "content": system_prompt}
        ]
        formatted_messages.extend(messages)
        # print("Formatted chat messages: " + str(formatted_messages))
        return formatted_messages

    @abstractmethod
    def _make_completion_request(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Make a completion request - to be implemented by specific providers"""
        pass

    @abstractmethod
    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> tuple:
        """Make a chat request - to be implemented by specific providers.

        Returns (visible_response, thinking_trace).
        """
        pass

    def generate(
            self,
            messages: List[Dict],
            system_prompt: str,
            stop: Optional[List[str]] = None,
            return_trace: bool = False,
    ) -> Union[str, tuple]:
        """Generate completion using either chat or completion API.

        When return_trace is true, return (visible_response, thinking_trace).
        Non-chat providers return an empty trace.
        """
        print("==============================================================")
        print("Generating completion with model: " + self.config.model)

        try:
            if self.config.is_chat:
                formatted_messages = self._format_chat_messages(system_prompt, messages)
                response, thinking_trace = self._make_chat_request(formatted_messages, stop)
                if return_trace:
                    return response, thinking_trace
                return response
            else:
                prompt = self._format_prompt(system_prompt, messages)
                response = self._make_completion_request(prompt, stop)
                if return_trace:
                    return response, ""
                return response
        except Exception as e:
            print("Failed to generate completion after retries: " + str(e))
            raise


class CompletionAPI(BaseLLMAPI):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.api_key:
            openai_api_key = "EMPTY"
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=config.base_url,
            )
            models = self.client.models.list()
            print(f"Available models {models}")
            self.config.model = models.data[0].id
        else:
            self.client = OpenAI(api_key=config.api_key)


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
            response = self.client.completions.create(
                model=self.config.model,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stop=stop
            )
            completion = response.choices[0].text.strip()
            # print("Received completion response: " + completion)
            print("===========================(Response-completion-start)===================================")
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
    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> tuple:
        try:
            # print("Making chat request with messages: " + str(messages))
            print("========================(Prompt-chat-start)======================================")
            # print(messages)
            print(messages[0]["content"])
            print("========================(Prompt-chat-end)======================================")
            # Do not pass task stop tokens to chat APIs. Reasoning models can emit
            # <think> blocks before their answer, and stop tokens may appear inside
            # the reasoning. Strip thinking and apply stops to visible text below.
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            raw_response = response.choices[0].message.content or ""
            thinking_trace = self._extract_thinking(raw_response)
            completion = self._strip_thinking(raw_response, stop)
            print("===========================(Response-chat-start)===================================")
            print(completion)
            print("===========================(Response-chat-end)===================================")
            # print("Received chat response: " + completion)
            return completion, thinking_trace
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

    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> tuple:
        """Make a chat request to Ollama"""
        # Ollama doesn't have a separate chat endpoint, so we'll format messages into a prompt
        formatted_prompt = ""
        for message in messages:
            formatted_prompt += f"{message['role']}: {message['content']}\nassistant: "

        return self._make_completion_request(formatted_prompt, stop), ""


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

    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> tuple:
        print("========================(Prompt-chat-start)======================================")
        # print(messages)
        print(messages[0]["content"])
        print("========================(Prompt-chat-end)======================================")
        chat = self.model.start_chat()
        # for message in messages:
        if messages[0]["role"] == "system":
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
        return completion, ""
