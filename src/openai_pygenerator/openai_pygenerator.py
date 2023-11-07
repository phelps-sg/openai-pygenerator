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
import time
from enum import Enum, auto
from typing import Callable, Dict, Iterable, Iterator, List, NewType, Optional, TypeVar

import openai
import urllib3.exceptions
from openai.error import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
)

Completion = Dict[str, str]
Seconds = NewType("Seconds", int)
Completions = Iterator[Completion]
History = Iterable[Completion]
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


GPT_MODEL = var("GPT_MODEL", str, "gpt-3.5-turbo")
GPT_TEMPERATURE = var("GPT_TEMPERATURE", float, 0.2)
GPT_MAX_TOKENS = var("GPT_MAX_TOKENS", int, 100)
GPT_MAX_RETRIES = var("GPT_MAX_RETRIES", int, 5)
GPT_RETRY_EXPONENT_SECONDS = Seconds(var("GPT_RETRY_EXPONENT_SECONDS", int, 2))
GPT_RETRY_BASE_SECONDS = Seconds(var("GPT_RETRY_BASE_SECONDS", int, 20))
GPT_REQUEST_TIMEOUT_SECONDS = Seconds(var("GPT_REQUEST_TIMEOUT_SECONDS", int, 60))

logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


def completer(
    model: str = GPT_MODEL,
    max_tokens: int = GPT_MAX_TOKENS,
    temperature: float = GPT_TEMPERATURE,
    max_retries: int = GPT_MAX_RETRIES,
    retry_base: Seconds = GPT_RETRY_BASE_SECONDS,
    retry_exponent: Seconds = GPT_RETRY_EXPONENT_SECONDS,
    request_timeout: Seconds = GPT_REQUEST_TIMEOUT_SECONDS,
) -> Completer:
    def f(messages: History, n: int = 1) -> Completions:
        return generate_completions(
            messages,
            model,
            max_tokens,
            temperature,
            max_retries,
            retry_base,
            retry_exponent,
            request_timeout,
            n,
        )

    return f


gpt_completions = completer()


def generate_completions(
    messages: History,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_base: Seconds,
    retry_exponent: Seconds,
    request_timeout: Seconds,
    n: int = 1,
    retries: int = 0,
) -> Completions:
    logger.debug("messages = %s", messages)
    try:
        result = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
            request_timeout=request_timeout,
        )
        logger.debug("response = %s", result)
        for choice in result.choices:  # type: ignore
            yield choice.message
    except (
        openai.error.Timeout,  # type: ignore
        urllib3.exceptions.TimeoutError,
        RateLimitError,
        APIConnectionError,
        APIError,
        ServiceUnavailableError,
    ) as err:
        if isinstance(err, APIError) and not (err.http_status in [524, 502, 500]):
            raise
        logger.warning("Error returned from openai API: %s", err)
        logger.debug("retries = %d", retries)
        if retries < max_retries:
            logger.info("Retrying... ")
            time.sleep(retry_base + retry_exponent**retries)
            for completion in generate_completions(
                messages,
                model,
                max_tokens,
                temperature,
                max_retries,
                retry_base,
                retry_exponent,
                request_timeout,
                n,
                retries + 1,
            ):
                yield completion
        else:
            logger.error("Maximum retries reached, aborting.")
            raise


def next_completion(completions: Completions) -> Optional[Completion]:
    try:
        return next(completions)
    except StopIteration:
        return None


def user_message(text: str) -> Completion:
    return {"role": "user", "content": text}


def content(completion: Completion) -> str:
    return completion["content"]


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
