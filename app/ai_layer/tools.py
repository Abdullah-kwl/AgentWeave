# search tools
"""
tools.py defines the tools that the agent can use to perform tasks.
The tools are defined as functions that take in arguments and return a result.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

search = DuckDuckGoSearchRun()
# print(search.invoke("can you tell me who is the primininster and presidend of pakistan"))


@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools_list = [add, multiply, divide, search]
