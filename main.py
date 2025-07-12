"""
Main application entry point for React Agent Template
Demonstrates the React agent workflow with professional logging and configuration
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import add_messages

from src.graph import build_graph
from src.state import MessageState
from src.config.config import config
from src.prompts import AGENT_SYSTEM_PROMPT
from src.config.logging_config import setup_logging, get_logger


def main():
    """Main application entry point"""
    # Setup logging first
    setup_logging(config)
    logger = get_logger(__name__)

    # Log application startup
    logger.info("React Agent Template starting...")
    logger.info(f"App: {config.app_name} | V{config.app_version}")
    logger.info(f"Environment: {config.environment}")

    try:
        # Build the React agent graph
        logger.info("Building React agent graph...")
        agent_graph = build_graph()
        logger.info("React agent graph built successfully")

        # 4. Create initial state with user message
        user_query = "what is twice of 10 + 10?"

        initial_state = MessageState(messages=[
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=user_query)]
        )

        logger.info(f"Processing user query: {user_query}")

        # 5. Run the agent
        logger.info("Agent processing started...")
        result = agent_graph.invoke(initial_state)

        # 6. Display results
        logger.info("Agent processing completed successfully")

        for m in result['messages']:
            m.pretty_print()

        # 7. Log final statistics
        logger.info("Agent execution summary", extra={
            "total_messages": len(result['messages']),
            "user_messages": len([m for m in result['messages'] if hasattr(m, 'type') and m.type == 'human']),
            "assistant_messages": len([m for m in result['messages'] if hasattr(m, 'type') and m.type == 'ai']),
            "tool_messages": len([m for m in result['messages'] if hasattr(m, 'type') and m.type == 'tool'])
        })

        logger.info("Application completed successfully")

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Check logs for detailed error information.")
        raise


if __name__ == "__main__":
    # Run single query demo
    main()