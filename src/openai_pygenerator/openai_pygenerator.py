#  Copyright (c) 2023 Steve Phelps.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import logging
import os
from enum import Enum, auto
from functools import lru_cache
from typing import Callable, Iterator, List, NewType, Optional, TypeVar

import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

Completion = ChatCompletionMessageParam
Seconds = NewType("Seconds", int)
Completions = Iterator[Completion]
History = List[Completion]
Completer = Callable[[History, int], Completions]
T = TypeVar("T")


class Role(Enum):
    USER = auto()
    ASSISTANT = auto()
    SYSTEM = auto()


def var(name: str, to_type: Callable[[str], T], default: T) -> T:
    result = os.environ.get(name)
    if result is None:
        return default
    return to_type(result)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = var("GPT_MODEL", str, "gpt-3.5-turbo")
GPT_TEMPERATURE = var("GPT_TEMPERATURE", float, 0.2)
GPT_MAX_TOKENS = var("GPT_MAX_TOKENS", int, 500)
GPT_MAX_RETRIES = var("GPT_MAX_RETRIES", int, 5)
GPT_REQUEST_TIMEOUT_SECONDS = Seconds(var("GPT_REQUEST_TIMEOUT_SECONDS", int, 20))

logger = logging.getLogger(__name__)


def completer(
    model: str = GPT_MODEL,
    max_tokens: int = GPT_MAX_TOKENS,
    temperature: float = GPT_TEMPERATURE,
    max_retries: int = GPT_MAX_RETRIES,
    request_timeout: Seconds = GPT_REQUEST_TIMEOUT_SECONDS,
) -> Completer:
    @lru_cache()
    def get_client() -> openai.OpenAI:
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=request_timeout,
            max_retries=max_retries,
        )

    def f(messages: History, n: int = 1) -> Completions:
        return generate_completions(
            get_client, messages, model, max_tokens, temperature, n
        )

    return f


gpt_completions = completer()


def generate_completions(
    client: Callable[[], openai.OpenAI],
    messages: History,
    model: str,
    max_tokens: int,
    temperature: float,
    n: int = 1,
) -> Completions:
    client_instance = client()
    logger.debug("client_instance = %s", client_instance)
    result = client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=n,
        temperature=temperature,
    )
    logger.debug("result = %s", result)
    for choice in result.choices:  # type: ignore
        yield to_message_param(choice.message)


def to_message_param(message: ChatCompletionMessage) -> Completion:
    return ChatCompletionAssistantMessageParam(
        {"role": message.role, "content": message.content}
    )


def next_completion(completions: Completions) -> Optional[Completion]:
    try:
        return next(completions)
    except StopIteration:
        return None


def user_message(text: str) -> Completion:
    return ChatCompletionUserMessageParam({"role": "user", "content": text})


def assistant_message(text: str) -> Completion:
    return ChatCompletionAssistantMessageParam({"role": "assistant", "content": text})


def content(completion: Completion) -> str:
    return str(completion["content"])


def role(completion: Completion) -> Role:
    r = completion["role"]
    if r == "user":
        return Role.USER
    elif r == "assistant":
        return Role.ASSISTANT
    elif r == "system":
        return Role.SYSTEM
    else:
        raise ValueError(f"Cannot determine role from {r}")


def is_user_role(completion: Completion) -> bool:
    return role(completion) == Role.USER


def is_assistant_role(completion: Completion) -> bool:
    return role(completion) == Role.ASSISTANT


def transcript(messages: History) -> List[str]:
    return [content(r) for r in messages]


class ChatSession:
    """Encapsulates chat session state

    :param generate: A function to generate completions from a history
    """

    def __init__(self, generate: Completer = gpt_completions):
        self._messages: List[Completion] = []
        self._generate = generate

    def ask(self, prompt: str) -> str:
        """
        Submit a message in the user-role to the chatbot,
        and record both the user message and assistant
        response in the chat history.
        :param prompt: The user message
        :return: The assistant response
        """
        message = user_message(prompt)
        self._messages.append(message)
        completions = self._generate(self.messages, 1)
        response = next(completions)
        self._messages.append(response)
        return content(response)

    @property
    def transcript(self) -> List[str]:
        return transcript(self.messages)

    @property
    def messages(self) -> History:
        return self._messages
