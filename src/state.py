"""
Define state structures for the agent.
"""
from langgraph.graph import MessagesState
from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage

class MessageState(MessagesState):
    """
    This is built-in chat history
    """
    llm: Runnable
