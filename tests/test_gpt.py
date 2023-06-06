# pylint: disable=redefined-outer-name

from typing import Iterable

import pytest
from openai.error import APIError, RateLimitError
from openai.openai_object import OpenAIObject

from openai_pygenerator import (
    GPT_MAX_RETRIES,
    ChatSession,
    Completer,
    Completion,
    Completions,
    Role,
    content,
    gpt_completions,
    is_assistant_role,
    is_user_role,
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


@pytest.fixture
def assistant_completion():
    return {"role": "assistant", "content": "testing"}


@pytest.fixture
def user_completion():
    return {"role": "user", "content": "testing"}


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError("rate limited", http_status=429),
        APIError("Gateway Timeout", http_status=524),
    ],
)
def test_generate_completion(mock_openai, mock_sleep, error):
    mock_openai.side_effect = [
        error,
        error,
        MockChoices(["Test completion 1", "Test completion 2"]),
    ]

    completions = list(gpt_completions([]))

    assert completions == ["Test completion 1", "Test completion 2"]
    assert mock_sleep.call_count == 2


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError("rate limited", http_status=429),
        APIError("Gateway Timeout", http_status=524),
    ],
)
def test_generate_completion_error(mock_openai, mock_sleep, error):
    mock_openai.side_effect = [error] * GPT_MAX_RETRIES

    with pytest.raises(Exception):
        _ = list(gpt_completions([]))

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
        def mock_complete(_history: Completions, _n: int) -> Completions:
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


def test_content(user_completion: Completion, assistant_completion: Completion):
    assert content(user_completion) == "testing"
    assert content(assistant_completion) == "testing"


def test_role(user_completion: Completion, assistant_completion: Completion):
    assert role(user_completion) == Role.USER
    assert role(assistant_completion) == Role.ASSISTANT
    assert is_user_role(user_completion)
    assert is_assistant_role(assistant_completion)
