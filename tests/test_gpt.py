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

import pytest
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from openai_pygenerator import (
    ChatSession,
    Completer,
    Completion,
    Completions,
    History,
    Role,
    content,
    role,
    transcript,
    user_message,
)


def aio(text: str) -> ChatCompletionMessage:
    result = ChatCompletionMessage(role="assistant", content=text)
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


def test_user_message():
    test_message = "test"
    result = user_message(test_message)
    assert result["role"] == "user"
    assert result["content"] == test_message


def test_transcript():
    def test_message(i: int) -> str:
        return f"message{i}"

    test_messages = [user_message(test_message(i)) for i in range(10)]
    result = list(transcript(test_messages))
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


@pytest.mark.parametrize(
    "message",
    [
        ChatCompletionAssistantMessageParam(
            {"role": "assistant", "content": "testing"}
        ),
        ChatCompletionUserMessageParam({"role": "user", "content": "testing"}),
        ChatCompletionSystemMessageParam({"role": "system", "content": "testing"}),
    ],
)
def test_content(message: ChatCompletionMessageParam):
    assert content(message) == "testing"


@pytest.mark.parametrize(
    "completion, expected",
    [
        (
            ChatCompletionAssistantMessageParam(
                {"role": "assistant", "content": "testing"}
            ),
            Role.ASSISTANT,
        ),
        (
            ChatCompletionUserMessageParam({"role": "user", "content": "testing"}),
            Role.USER,
        ),
        (
            ChatCompletionSystemMessageParam({"role": "system", "content": "testing"}),
            Role.SYSTEM,
        ),
    ],
)
def test_role(completion: Completion, expected: Role):
    assert role(completion) == expected
