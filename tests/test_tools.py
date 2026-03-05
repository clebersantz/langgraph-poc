"""Tests for tool functions."""

from __future__ import annotations

from pathlib import Path

from src.tools.code_tools import create_file, list_directory, read_file, run_command, search_code


class TestRunCommand:
    """Tests for the run_command tool."""

    def test_successful_command(self):
        result = run_command.invoke({"command": "echo hello"})
        assert result["success"] is True
        assert "hello" in result["stdout"]
        assert result["return_code"] == 0

    def test_failed_command(self):
        result = run_command.invoke({"command": "exit 1"})
        assert result["success"] is False
        assert result["return_code"] == 1

    def test_command_with_working_dir(self, tmp_path):
        result = run_command.invoke({"command": "pwd", "working_dir": str(tmp_path)})
        assert result["success"] is True
        assert str(tmp_path) in result["stdout"]

    def test_command_timeout(self):
        result = run_command.invoke({"command": "sleep 10", "timeout": 1})
        assert result["success"] is False
        assert "timed out" in result["error"].lower()


class TestCreateFile:
    """Tests for the create_file tool."""

    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "test.txt")
        result = create_file.invoke({"path": path, "content": "hello world"})
        assert result["status"] == "created"
        assert Path(path).read_text() == "hello world"

    def test_creates_nested_directories(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "c" / "file.txt")
        result = create_file.invoke({"path": path, "content": "nested"})
        assert result["status"] == "created"
        assert Path(path).exists()

    def test_does_not_overwrite_by_default(self, tmp_path):
        path = str(tmp_path / "existing.txt")
        Path(path).write_text("original")
        result = create_file.invoke({"path": path, "content": "new content"})
        assert "error" in result
        assert Path(path).read_text() == "original"

    def test_overwrites_when_flag_set(self, tmp_path):
        path = str(tmp_path / "existing.txt")
        Path(path).write_text("original")
        result = create_file.invoke({"path": path, "content": "new content", "overwrite": True})
        assert result["status"] == "created"
        assert Path(path).read_text() == "new content"


class TestReadFile:
    """Tests for the read_file tool."""

    def test_reads_existing_file(self, tmp_path):
        path = tmp_path / "file.txt"
        path.write_text("file content")
        result = read_file.invoke({"path": str(path)})
        assert result["content"] == "file content"

    def test_returns_error_for_missing_file(self, tmp_path):
        result = read_file.invoke({"path": str(tmp_path / "missing.txt")})
        assert "error" in result

    def test_truncates_large_files(self, tmp_path):
        path = tmp_path / "large.txt"
        path.write_text("x" * 20000)
        result = read_file.invoke({"path": str(path), "max_chars": 100})
        assert result["content"].startswith("x" * 100)
        assert "truncated" in result["content"]
        assert "20000" in result["content"]


class TestListDirectory:
    """Tests for the list_directory tool."""

    def test_lists_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = list_directory.invoke({"path": str(tmp_path)})
        names = [e["name"] for e in result["entries"]]
        assert "a.txt" in names
        assert "b.txt" in names

    def test_returns_error_for_missing_path(self, tmp_path):
        result = list_directory.invoke({"path": str(tmp_path / "missing")})
        assert "error" in result

    def test_recursive_listing(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "deep.txt").write_text("deep")
        result = list_directory.invoke({"path": str(tmp_path), "recursive": True})
        names = [e["name"] for e in result["entries"]]
        assert any("deep.txt" in n for n in names)


class TestSearchCode:
    """Tests for the search_code tool."""

    def test_finds_pattern_in_files(self, tmp_path):
        py_file = tmp_path / "sample.py"
        py_file.write_text("def hello_world():\n    pass\n")
        result = search_code.invoke(
            {
                "pattern": "hello_world",
                "search_path": str(tmp_path),
                "file_pattern": "*.py",
            }
        )
        assert result["count"] >= 1
        assert any("hello_world" in m["content"] for m in result["matches"])

    def test_returns_empty_for_no_matches(self, tmp_path):
        py_file = tmp_path / "empty.py"
        py_file.write_text("x = 1\n")
        result = search_code.invoke(
            {
                "pattern": "nonexistent_pattern_xyz",
                "search_path": str(tmp_path),
            }
        )
        assert result["count"] == 0
