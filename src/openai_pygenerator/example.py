from openai_pygenerator.openai_pygenerator import ChatSession, completer


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


if __name__ == "__main__":
    heading("Find square root - using environment variables for parameters")
    example_square_root(session=ChatSession())

    heading("Find square root - overriding temperature, max_tokens, max_retries")
    example_square_root(
        session=ChatSession(
            generate=completer(temperature=0.5, max_tokens=300, max_retries=5)
        )
    )
