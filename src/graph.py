"""
React-agent workflow implementation using langgraph
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import src.tools as agent_tools
from src.config.config import config
from src.state import MessageState
from src.prompts import AGENT_SYSTEM_PROMPT
from src.config.logging_config import get_logger


logger = get_logger(__name__)

# Node 1
def initialize_llm_node(state: MessageState) -> MessageState:
    try:
        llm = ChatOpenAI(
            model=config.openai_model,
            api_key=config.openai_api_key.get_secret_value(),
            temperature=config.openai_temperature,
            max_tokens=config.openai_max_tokens
        )

        tools = [
            agent_tools.add,
            agent_tools.divide,
            agent_tools.multiply
        ]

        llm_with_tool = llm.bind_tools(tools)

        state['llm'] = llm_with_tool

        return state

    except Exception as e:
        logger.error(f"LLM initialization failed: {e}", exc_info=True)
        raise


# Node 2
def assistant_node(state: MessageState) -> MessageState:
    llm = state['llm']
    try:
        response = llm.invoke(state['messages'])
        state['messages'] = add_messages(state['messages'], response)

        return state

    except Exception as e:
        logger.error(f"LLM failed to generate response: {e}")
        raise



def build_graph():
    try:
        # Create builder
        builder = StateGraph(MessageState)

        # Add nodes
        builder.add_node("initialize_llm_node", initialize_llm_node)
        builder.add_node("assistant_node", assistant_node)
        builder.add_node("tools", ToolNode(
            [
                agent_tools.add,
                agent_tools.divide,
                agent_tools.multiply
            ]
        ))

        # Add edges
        builder.add_edge(START, "initialize_llm_node")
        builder.add_edge('initialize_llm_node', "assistant_node")
        builder.add_conditional_edges("assistant_node", tools_condition)
        builder.add_edge("tools", "assistant_node")

        react_graph = builder.compile()

        return react_graph

    except Exception as e:
        logger.error(f"Failed to create graph: {e}")
        raise

# Make sure the graph is accessible
__all__ = ['build_graph']
