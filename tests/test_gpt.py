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

from typing import Iterable

import openai.error
import pytest
import urllib3.exceptions as urlex
from openai.error import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
)
from openai.openai_object import OpenAIObject

from openai_pygenerator import (
    GPT_MAX_RETRIES,
    ChatSession,
    Completer,
    Completion,
    Completions,
    History,
    Role,
    content,
    gpt_completions,
    role,
    transcript,
    user_message,
)


def aio(text: str) -> OpenAIObject:
    result = OpenAIObject()
    result["message"] = text
    return result


# pylint: disable=too-few-public-methods
class MockChoices:
    def __init__(self, responses: Iterable[str]):
        self.choices = [aio(text) for text in responses]


@pytest.fixture
def mock_openai(mocker):
    return mocker.patch("openai.ChatCompletion.create")


@pytest.fixture
def mock_sleep(mocker):
    return mocker.patch("time.sleep", return_value=None)


def make_test_completion(role: str) -> Completion:
    return {"role": role, "content": "testing"}


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError("rate limited", http_status=429),
        APIConnectionError("connection timeout"),
        APIError("Gateway Timeout", http_status=524),
        ServiceUnavailableError(
            message=(
                "openai.error.ServiceUnavailableError:"
                " The server is overloaded or not ready yet"
            ),
            http_status=503,
        ),
    ],
)
def test_generate_completion(mock_openai, mock_sleep, error):
    mock_openai.side_effect = [
        error,
        error,
        MockChoices(["Test completion 1", "Test completion 2"]),
    ]

    completions = list(gpt_completions([]))  # type: ignore

    assert completions == ["Test completion 1", "Test completion 2"]
    assert mock_sleep.call_count == 2


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError("rate limited", http_status=429),
        APIError("Gateway Timeout", http_status=524),
        APIError("Server shutdown", http_status=500),
        ServiceUnavailableError("Service unavailable"),
        urlex.ReadTimeoutError("test-pool", "http://test", "read timeout"),  # type: ignore
        openai.error.Timeout,
    ],
)
def test_generate_completion_error(mock_openai, mock_sleep, error):
    mock_openai.side_effect = [error] * GPT_MAX_RETRIES

    with pytest.raises(Exception):
        _ = list(gpt_completions([]))  # type: ignore

    assert mock_sleep.call_count == GPT_MAX_RETRIES


def test_user_message():
    test_message = "test"
    result = user_message(test_message)
    assert result["role"] == "user"
    assert result["content"] == test_message


def test_transcript():
    def test_message(i: int) -> str:
        return f"message{i}"

    test_messages = [user_message(test_message(i)) for i in range(10)]
    result = list(transcript(iter(test_messages)))
    for i in range(10):
        assert result[i] == test_message(i)


def test_chat_session():
    def completer(response: str) -> Completer:
        def mock_complete(_history: History, _n: int) -> Completions:
            yield {"role": "assistant", "content": response}

        return mock_complete

    session = ChatSession(completer("response1"))
    result = session.ask("First question")
    assert result == "response1"
    session._generate = completer("response2")  # pylint: disable=protected-access
    result = session.ask("Second question")
    assert result == "response2"
    assert session.transcript == [
        "First question",
        "response1",
        "Second question",
        "response2",
    ]


@pytest.mark.parametrize("role", ["user", "system", "assistant"])
def test_content(role: str):
    completion = make_test_completion(role)
    assert content(completion) == "testing"


@pytest.mark.parametrize(
    "test_role_str, expected",
    [("user", Role.USER), ("system", Role.SYSTEM), ("assistant", Role.ASSISTANT)],
)
def test_role(test_role_str: str, expected: Role):
    completion = make_test_completion(test_role_str)
    assert role(completion) == expected
