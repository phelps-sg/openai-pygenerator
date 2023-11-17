# openai-pygenerator

[![GitHub Workflow Status](https://github.com/phelps-sg/openai-pygenerator/actions/workflows/python-package.yml/badge.svg)](https://github.com/phelps-sg/openai-pygenerator/actions/workflows/python-package.yml)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/phelps-sg/openai-pygenerator)
![GitHub](https://img.shields.io/github/license/phelps-sg/openai-pygenerator?color=blue)

## Import Note

The openai Python API versions 1.0.0 and later now incorporate retry functionality and type annotations. This package has been migrated to the new API, but should only be used as a temporary measure to ensure backward compatibility.  If you are using this package in production, consider rewriting your code so that it uses the new openai API directly.

## Overview

This is a simple type-annotated wrapper around the OpenAI Python API which:
- provides configurable retry functionality,
- reduces the default timeout from 10 minutes to 20 seconds (configurable), 
- provides a simple class to manage chat session state, and
- provides a generator over completions.
  
It can also be used to chain together completions from different prompts in a very straightforward [functional-programming](https://docs.python.org/3/howto/functional.html) style using [Python generators](https://docs.python.org/3/howto/functional.html#generators).

## Installation

~~~bash
pip install openai-pygenerator
~~~

## Basic usage

In the example below we will retry automatically if there is a `RateLimitError`.

~~~python
from openai_pygenerator import ChatSession
 
session = ChatSession()
solution = session.ask("What is the square root of 256?")
print(solution)
working = session.ask("Show your working")
print(working)
print("Transcript:")
print(session.transcript)
~~~

## Completion pipelines and overriding parameters

~~~python
from typing import Iterable

from openai_pygenerator import (
    ChatSession,
    Completions,
    completer,
    content,
    next_completion,
    user_message,
)

high_temp_completions = completer(temperature=0.8)


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
    return high_temp_completions([user_message(prompt)], n=num_completions)


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
~~~

## Running 

~~~bash
export OPENAI_API_KEY=<key>
python src/openai_pygenerator/example.py
~~~

## Configuration

To override default parameters use the following shell environment variables:

~~~bash
export GPT_MODEL=gpt-3.5-turbo
export GPT_TEMPERATURE=0.2
export GPT_MAX_TOKENS=500
export GPT_MAX_RETRIES=5
export GPT_REQUEST_TIMEOUT_SECONDS=20
export OPENAI_API_KEY=<key>
python src/openai_pygenerator/example.py
~~~
