"""Code manipulation and execution tools for the multi-agent system."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Maximum output size to prevent flooding state
_MAX_OUTPUT_CHARS = 8_000


@tool
def run_command(
    command: str,
    working_dir: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Run a shell command in a subprocess.

    Args:
        command: Shell command to execute.
        working_dir: Working directory for the command (optional).
        timeout: Timeout in seconds (default: 120).

    Returns:
        Dictionary with stdout, stderr, and return_code.

    Security note:
        This tool uses shell=True to support compound commands (pipes, redirects,
        subshells) required by development agents. It must only be called in a
        sandboxed environment (e.g., a Docker container) and must never receive
        untrusted user input directly.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,  # noqa: S602
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
        )
        stdout = result.stdout[:_MAX_OUTPUT_CHARS]
        stderr = result.stderr[:_MAX_OUTPUT_CHARS]
        logger.info("Command '%s' exited with code %d", command, result.returncode)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        logger.warning("Command '%s' timed out after %ds", command, timeout)
        return {"error": f"Command timed out after {timeout}s", "return_code": -1, "success": False}
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to run command '%s': %s", command, e)
        return {"error": str(e), "return_code": -1, "success": False}


@tool
def create_file(path: str, content: str, overwrite: bool = False) -> dict[str, str]:
    """Create or overwrite a file with given content.

    Args:
        path: Absolute or relative file path.
        content: File content as string.
        overwrite: Whether to overwrite existing files (default: False).

    Returns:
        Dictionary with path and status.
    """
    try:
        file_path = Path(path)
        if file_path.exists() and not overwrite:
            return {
                "path": str(file_path),
                "status": "exists",
                "error": "File exists and overwrite=False",
            }
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        logger.info("Created file: %s", file_path)
        return {"path": str(file_path), "status": "created"}
    except OSError as e:
        logger.error("Failed to create file '%s': %s", path, e)
        return {"path": path, "error": str(e)}


@tool
def read_file(path: str, max_chars: int = 8000) -> dict[str, str]:
    """Read the content of a file.

    Args:
        path: Absolute or relative file path.
        max_chars: Maximum characters to return (default: 8000).

    Returns:
        Dictionary with path and content.
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return {"path": path, "error": f"File not found: {path}"}
        content = file_path.read_text(encoding="utf-8")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[...truncated, total {len(content)} chars]"
        return {"path": str(file_path), "content": content}
    except OSError as e:
        logger.error("Failed to read file '%s': %s", path, e)
        return {"path": path, "error": str(e)}


@tool
def list_directory(path: str, recursive: bool = False) -> dict[str, Any]:
    """List files and directories at a path.

    Args:
        path: Directory path to list.
        recursive: Whether to list recursively (default: False).

    Returns:
        Dictionary with list of entries.
    """
    try:
        dir_path = Path(path)
        if not dir_path.exists():
            return {"path": path, "error": f"Path not found: {path}"}
        if not dir_path.is_dir():
            return {"path": path, "error": f"Not a directory: {path}"}

        entries = []
        if recursive:
            for item in sorted(dir_path.rglob("*")):
                entries.append(
                    {
                        "name": str(item.relative_to(dir_path)),
                        "type": "file" if item.is_file() else "dir",
                        "size": item.stat().st_size if item.is_file() else None,
                    }
                )
        else:
            for item in sorted(dir_path.iterdir()):
                entries.append(
                    {
                        "name": item.name,
                        "type": "file" if item.is_file() else "dir",
                        "size": item.stat().st_size if item.is_file() else None,
                    }
                )
        return {"path": str(dir_path), "entries": entries}
    except OSError as e:
        logger.error("Failed to list directory '%s': %s", path, e)
        return {"path": path, "error": str(e)}


@tool
def search_code(pattern: str, search_path: str, file_pattern: str = "*.py") -> dict[str, Any]:
    """Search for a pattern in code files using grep.

    Args:
        pattern: Regular expression or string to search for.
        search_path: Directory to search in.
        file_pattern: Glob pattern for file types (default: '*.py').

    Returns:
        Dictionary with list of matches.
    """
    try:
        result = subprocess.run(
            ["grep", "-r", "-n", "--include", file_pattern, pattern, search_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        matches = []
        for line in result.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append({"file": parts[0], "line": parts[1], "content": parts[2]})
        return {"pattern": pattern, "matches": matches, "count": len(matches)}
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out", "pattern": pattern}
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to search code: %s", e)
        return {"error": str(e), "pattern": pattern}
