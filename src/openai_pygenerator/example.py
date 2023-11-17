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

from openai_pygenerator import (
    ChatSession,
    Completions,
    completer,
    content,
    next_completion,
    user_message,
)

high_temp_completions = completer(temperature=1.0)


def heading(message: str, margin: int = 80) -> None:
    print()
    print("-" * margin)
    print(message)
    print("-" * margin)
    print()


def example_square_root(session: ChatSession) -> None:
    solution = session.ask("What is the square root of 256?")
    print(solution)
    working = session.ask("Show your working")
    print(working)

    heading("Session transcript:")
    print(session.transcript)


def creative_answer(prompt: str, num_completions: int = 1) -> Completions:
    return high_temp_completions([user_message(prompt)], num_completions)


def pick_color(num_completions: int) -> Completions:
    return creative_answer(
        "Pick a color at random and then just tell me your choice, e.g. 'red'",
        num_completions,
    )


def generate_sentence(color_completions: Completions) -> Iterable[str]:
    for color_completion in color_completions:
        color = content(color_completion)
        result = next_completion(
            creative_answer(f"Write a sentence about the color {color}.")
        )
        if result is not None:
            yield content(result)


if __name__ == "__main__":
    heading("Find square root - using environment variables for parameters")
    example_square_root(session=ChatSession())

    heading("Find square root - overriding temperature, max_tokens, max_retries")
    example_square_root(
        session=ChatSession(
            generate=completer(temperature=0.5, max_tokens=300, max_retries=5)
        )
    )

    heading("Example completion pipeline")
    for sentence in generate_sentence(pick_color(num_completions=10)):
        print(sentence)
