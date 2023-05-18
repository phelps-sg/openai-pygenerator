import logging
import os
import time
from typing import Callable, Dict, Iterable, List, NewType, TypeVar

import openai
from openai.error import APIError, RateLimitError

Completion = Dict[str, str]
Seconds = NewType("Seconds", int)
Conversation = Iterable[Completion]
T = TypeVar("T")


def var(name: str, to_type: Callable[[str], T], default: T) -> T:
    result = os.environ.get(name)
    if result is None:
        return default
    else:
        return to_type(result)


GPT_MODEL = var("GPT_MODEL", str, "gpt-3.5-turbo")
GPT_TEMPERATURE = var("GPT_TEMPERATURE", float, 0.2)
GPT_MAX_TOKENS = var("GPT_MAX_TOKENS", int, 100)
GPT_MAX_RETRIES = var("GPT_MAX_RETRIES", int, 5)
GPT_RETRY_EXPONENT_SECONDS = Seconds(var("GPT_RETRY_EXPONENT_SECONDS", int, 2))
GPT_RETRY_BASE_SECONDS = Seconds(var("GPT_RETRY_BASE_SECONDS", int, 20))

logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


def completer(
    model: str = GPT_MODEL,
    max_tokens: int = GPT_MAX_TOKENS,
    temperature: float = GPT_TEMPERATURE,
    max_retries: int = GPT_MAX_RETRIES,
    retry_base: Seconds = GPT_RETRY_BASE_SECONDS,
    retry_exponent: Seconds = GPT_RETRY_EXPONENT_SECONDS,
) -> Callable[[Conversation, int], Conversation]:
    def f(messages: Conversation, n: int = 1) -> Conversation:
        return generate_completions(
            messages,
            model,
            max_tokens,
            temperature,
            max_retries,
            retry_base,
            retry_exponent,
            n,
        )

    return f


gpt_completions = completer()


def generate_completions(
    messages: Conversation,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_base: Seconds,
    retry_exponent: Seconds,
    n: int = 1,
    retries: int = 0,
) -> Conversation:
    logger.debug("messages = %s", messages)
    try:
        result = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
        )
        logger.debug("response = %s", result)
        for choice in result.choices:
            yield choice.message
    except (RateLimitError, APIError) as err:
        if isinstance(err, APIError) and not (err.http_status in [524, 502]):
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
                n,
                retries + 1,
            ):
                yield completion
        else:
            logger.error("Maximum retries reached, aborting.")
            raise


def user_message(content: str) -> Completion:
    return Completion({"role": "user", "content": content})


def transcript(messages: Conversation) -> List[str]:
    return [r["content"] for r in messages]
