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
