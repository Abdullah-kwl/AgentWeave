"""
This module defines the prompts used by the agent.
The prompts are used to set the context for the agent
and to provide instructions on how to use the tools.
"""

from langchain_core.messages import SystemMessage


def get_system_message():
    """Returns a system message that sets the context for the agent."""
    content = "You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs."
    return SystemMessage(content=content)
