# openai-pygenerator

This is a simple wrapper around the OpenAI Python API which provides
type annotations, retry functionality and a generator over completions.


## Installation

~~~bash
pip install openai-pygenerator
~~~

## Usage

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

For an example of overriding parameters see [src/example.py](src/example.py).

## Running 

~~~bash
export OPENAI_API_KEY=<key>
python src/example.py
~~~

## Configuration

~~~bash
export GPT_MODEL=gpt-3.5-turbo
export GPT_TEMPERATURE=0.2
export GPT_MAX_TOKENS=100
export GPT_MAX_RETRIES=5
export GPT_RETRY_EXPONENT_SECONDS=2
export GPT_RETRY_BASE_SECONDS=20
export OPENAI_API_KEY=<key>
python src/example.py
~~~

