"""Shared tool execution loop for agent nodes."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

logger = logging.getLogger(__name__)

# Maximum number of tool-call → execute → re-invoke rounds per agent activation.
# This guards against infinite loops while still allowing multi-step tool use.
_MAX_TOOL_ROUNDS = 10


def _workspace_is_empty(path: str) -> bool:
    """Return True if the workspace directory contains no files."""
    if not path or not os.path.isdir(path):
        return True
    for _, _, files in os.walk(path):
        if files:
            return False
    return True


def _write_workspace_file(workspace_path: str, fname: str, content: str) -> None:
    """Write *content* to *fname* (relative) inside *workspace_path*."""
    # Sanitize: strip leading slashes so join works correctly
    fname = fname.lstrip("/\\")
    full_path = Path(workspace_path) / fname
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    logger.info("Fallback extractor: created %s", full_path)


def extract_files_from_content(content: str, workspace_path: str) -> list[str]:
    """Extract file contents from an LLM text response and write them to *workspace_path*.

    This is a last-resort fallback used when ``run_tool_loop`` completed without
    the model making any ``create_file`` tool calls.  Two common response formats
    are recognised:

    1. **JSON code block** — the model emits a ``\\`\\`\\`json`` block whose top-level
       object has a ``"files"`` array of ``{"path": ..., "content": ...}`` dicts.

    2. **Markdown bold-filename header** — ``**filename.py**`` (or with backtick
       escaping) directly above a fenced code block.

    Args:
        content: The raw text content from the LLM response.
        workspace_path: Absolute path of the per-run workspace directory.

    Returns:
        List of relative file paths that were successfully written.
    """
    if not content or not workspace_path:
        return []

    created: list[str] = []
    seen: set[str] = set()

    # ── Format 1: ```json\n{"files": [...]}\n``` ──────────────────────────────
    # Use a greedy match from the first '{' to the last '}' inside the fence so
    # that nested JSON structures (arrays, nested objects) are captured correctly.
    for block in re.findall(r"```(?:json)?\s*\n(\{.*\})\s*\n```", content, re.DOTALL):
        try:
            data = json.loads(block)
            for f in data.get("files", []):
                fname = str(f.get("path", f.get("filename", ""))).strip()
                fcontent = str(f.get("content", ""))
                if fname and fname not in seen:
                    _write_workspace_file(workspace_path, fname, fcontent)
                    created.append(fname)
                    seen.add(fname)
        except (ValueError, KeyError, TypeError):
            continue

    if created:
        return created

    # ── Format 2: **filename.ext** (or **`filename.ext`**) before code block ──
    for m in re.finditer(
        r"\*\*`?([A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z0-9]+)`?\*\*[^\n]*\n"
        r"```[^\n]*\n(.*?)```",
        content,
        re.DOTALL,
    ):
        fname = m.group(1).strip()
        fcontent = m.group(2).rstrip("\n")
        if fname and fname not in seen:
            _write_workspace_file(workspace_path, fname, fcontent)
            created.append(fname)
            seen.add(fname)

    return created


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
