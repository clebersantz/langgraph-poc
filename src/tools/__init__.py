"""Tools package for the multi-agent system."""

from src.tools.code_tools import (
    create_file,
    list_directory,
    read_file,
    run_command,
    search_code,
)
from src.tools.git_tools import (
    clone_repository,
    commit_changes,
    create_branch,
    get_diff,
    merge_branch,
    pull_changes,
    push_changes,
)
from src.tools.github_tools import (
    add_comment_to_issue,
    add_comment_to_pr,
    close_issue,
    create_issue,
    create_pull_request,
    get_issue,
    get_pull_request,
    list_issues,
    list_pull_requests,
    merge_pull_request,
    update_issue,
    update_pull_request,
)

__all__ = [
    # Code tools
    "create_file",
    "list_directory",
    "read_file",
    "run_command",
    "search_code",
    # Git tools
    "clone_repository",
    "commit_changes",
    "create_branch",
    "get_diff",
    "merge_branch",
    "pull_changes",
    "push_changes",
    # GitHub tools
    "add_comment_to_issue",
    "add_comment_to_pr",
    "close_issue",
    "create_issue",
    "create_pull_request",
    "get_issue",
    "get_pull_request",
    "list_issues",
    "list_pull_requests",
    "merge_pull_request",
    "update_issue",
    "update_pull_request",
]
