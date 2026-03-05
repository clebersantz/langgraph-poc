"""Shared tool execution loop for agent nodes."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

logger = logging.getLogger(__name__)

# Maximum number of tool-call → execute → re-invoke rounds per agent activation.
# This guards against infinite loops while still allowing multi-step tool use.
_MAX_TOOL_ROUNDS = 10


async def run_tool_loop(
    agent_llm: Any,
    tools: list[Any],
    messages: list[BaseMessage],
) -> BaseMessage:
    """Invoke *agent_llm* and execute any tool calls it returns, repeating until
    the LLM produces a final response with no further tool calls.

    Args:
        agent_llm: LLM with tools already bound via ``llm.bind_tools(tools)``.
        tools: The same tool list passed to ``bind_tools`` — used to look up
            callable objects by name so their results can be fed back.
        messages: The formatted prompt messages to start the conversation.

    Returns:
        The final :class:`~langchain_core.messages.BaseMessage` after all tool
        calls have been executed (or the first response if none were requested).
    """
    tool_map = {t.name: t for t in tools}
    response = await agent_llm.ainvoke(messages)

    for round_num in range(_MAX_TOOL_ROUNDS):
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            break

        logger.debug("Agent tool round %d: %d call(s)", round_num + 1, len(tool_calls))
        messages = list(messages) + [response]
        tool_msgs: list[BaseMessage] = []

        for tc in tool_calls:
            t = tool_map.get(tc["name"])
            if t is None:
                result: Any = {"error": f"Unknown tool: {tc['name']}"}
            else:
                try:
                    result = await t.ainvoke(tc["args"])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Tool '%s' raised %s: %s", tc["name"], type(exc).__name__, exc)
                    result = {"error": str(exc)}

            content = json.dumps(result) if not isinstance(result, str) else result
            tool_msgs.append(ToolMessage(content=content, tool_call_id=tc["id"]))

        messages = messages + tool_msgs
        response = await agent_llm.ainvoke(messages)

    return response
