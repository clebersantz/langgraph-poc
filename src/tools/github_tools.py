"""GitHub integration tools for the multi-agent system."""

from __future__ import annotations

import logging
from typing import Any

from github import Auth, Github, GithubException
from langchain_core.tools import tool

from src.config import get_settings

logger = logging.getLogger(__name__)


def _get_github_client() -> Github:
    """Return an authenticated GitHub client."""
    settings = get_settings()
    token = settings.github_token.get_secret_value()
    if not token:
        raise ValueError("GITHUB_TOKEN is not configured or is empty")
    return Github(auth=Auth.Token(token))


def _get_repo(client: Github | None = None):
    """Return the configured GitHub repository."""
    settings = get_settings()
    if client is None:
        client = _get_github_client()
    owner = settings.github_owner
    repo_name = settings.github_repo
    if not owner or not repo_name:
        raise ValueError("GITHUB_OWNER and GITHUB_REPO must be set")
    return client.get_repo(f"{owner}/{repo_name}")


@tool
def create_issue(title: str, body: str, labels: list[str] | None = None) -> dict[str, Any]:
    """Create a new GitHub issue.

    Args:
        title: Issue title.
        body: Issue body/description in markdown.
        labels: Optional list of label names to apply.

    Returns:
        Dictionary with issue number, title, and URL.
    """
    try:
        repo = _get_repo()
        kwargs: dict[str, Any] = {"title": title, "body": body}
        if labels:
            kwargs["labels"] = labels
        issue = repo.create_issue(**kwargs)
        logger.info("Created issue #%d: %s", issue.number, issue.title)
        return {
            "number": issue.number,
            "title": issue.title,
            "url": issue.html_url,
            "state": issue.state,
        }
    except GithubException as e:
        logger.error("Failed to create issue: %s", e)
        return {"error": str(e)}


@tool
def get_issue(issue_number: int) -> dict[str, Any]:
    """Get details of a GitHub issue.

    Args:
        issue_number: The issue number to retrieve.

    Returns:
        Dictionary with issue details.
    """
    try:
        repo = _get_repo()
        issue = repo.get_issue(issue_number)
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "state": issue.state,
            "url": issue.html_url,
            "labels": [label.name for label in issue.labels],
            "assignees": [a.login for a in issue.assignees],
            "comments": issue.comments,
            "created_at": str(issue.created_at),
            "updated_at": str(issue.updated_at),
        }
    except GithubException as e:
        logger.error("Failed to get issue #%d: %s", issue_number, e)
        return {"error": str(e)}


@tool
def list_issues(state: str = "open", labels: list[str] | None = None) -> list[dict[str, Any]]:
    """List GitHub issues.

    Args:
        state: Filter by state ('open', 'closed', 'all').
        labels: Optional list of label names to filter by.

    Returns:
        List of issue dictionaries.
    """
    try:
        repo = _get_repo()
        kwargs: dict[str, Any] = {"state": state}
        if labels:
            kwargs["labels"] = labels
        issues = repo.get_issues(**kwargs)
        return [
            {
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "url": issue.html_url,
                "labels": [label.name for label in issue.labels],
            }
            for issue in issues
        ]
    except GithubException as e:
        logger.error("Failed to list issues: %s", e)
        return [{"error": str(e)}]


@tool
def update_issue(
    issue_number: int,
    title: str | None = None,
    body: str | None = None,
    state: str | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Update a GitHub issue.

    Args:
        issue_number: The issue number to update.
        title: New title (optional).
        body: New body (optional).
        state: New state ('open' or 'closed', optional).
        labels: New labels list (optional).

    Returns:
        Updated issue details.
    """
    try:
        repo = _get_repo()
        issue = repo.get_issue(issue_number)
        kwargs: dict[str, Any] = {}
        if title is not None:
            kwargs["title"] = title
        if body is not None:
            kwargs["body"] = body
        if state is not None:
            kwargs["state"] = state
        if labels is not None:
            kwargs["labels"] = labels
        issue.edit(**kwargs)
        return {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "url": issue.html_url,
        }
    except GithubException as e:
        logger.error("Failed to update issue #%d: %s", issue_number, e)
        return {"error": str(e)}


@tool
def close_issue(issue_number: int, comment: str | None = None) -> dict[str, Any]:
    """Close a GitHub issue.

    Args:
        issue_number: The issue number to close.
        comment: Optional comment to add before closing.

    Returns:
        Closed issue details.
    """
    try:
        repo = _get_repo()
        issue = repo.get_issue(issue_number)
        if comment:
            issue.create_comment(comment)
        issue.edit(state="closed")
        return {"number": issue.number, "state": issue.state, "url": issue.html_url}
    except GithubException as e:
        logger.error("Failed to close issue #%d: %s", issue_number, e)
        return {"error": str(e)}


@tool
def add_comment_to_issue(issue_number: int, comment: str) -> dict[str, Any]:
    """Add a comment to a GitHub issue.

    Args:
        issue_number: The issue number to comment on.
        comment: Comment text in markdown.

    Returns:
        Comment details.
    """
    try:
        repo = _get_repo()
        issue = repo.get_issue(issue_number)
        created = issue.create_comment(comment)
        return {"id": created.id, "url": created.html_url, "body": created.body}
    except GithubException as e:
        logger.error("Failed to add comment to issue #%d: %s", issue_number, e)
        return {"error": str(e)}


@tool
def create_pull_request(
    title: str,
    body: str,
    head: str,
    base: str = "main",
    draft: bool = False,
) -> dict[str, Any]:
    """Create a GitHub pull request.

    Args:
        title: PR title.
        body: PR description in markdown.
        head: Head branch (source).
        base: Base branch (target, default: 'main').
        draft: Whether to create as draft.

    Returns:
        PR details dictionary.
    """
    try:
        repo = _get_repo()
        pr = repo.create_pull(title=title, body=body, head=head, base=base, draft=draft)
        logger.info("Created PR #%d: %s", pr.number, pr.title)
        return {
            "number": pr.number,
            "title": pr.title,
            "url": pr.html_url,
            "state": pr.state,
            "head": pr.head.ref,
            "base": pr.base.ref,
            "draft": pr.draft,
        }
    except GithubException as e:
        logger.error("Failed to create PR: %s", e)
        return {"error": str(e)}


@tool
def get_pull_request(pr_number: int) -> dict[str, Any]:
    """Get details of a GitHub pull request.

    Args:
        pr_number: The PR number to retrieve.

    Returns:
        PR details dictionary.
    """
    try:
        repo = _get_repo()
        pr = repo.get_pull(pr_number)
        return {
            "number": pr.number,
            "title": pr.title,
            "body": pr.body,
            "state": pr.state,
            "url": pr.html_url,
            "head": pr.head.ref,
            "base": pr.base.ref,
            "draft": pr.draft,
            "merged": pr.merged,
            "mergeable": pr.mergeable,
            "labels": [label.name for label in pr.labels],
            "reviews": pr.get_reviews().totalCount,
        }
    except GithubException as e:
        logger.error("Failed to get PR #%d: %s", pr_number, e)
        return {"error": str(e)}


@tool
def list_pull_requests(state: str = "open") -> list[dict[str, Any]]:
    """List GitHub pull requests.

    Args:
        state: Filter by state ('open', 'closed', 'all').

    Returns:
        List of PR dictionaries.
    """
    try:
        repo = _get_repo()
        prs = repo.get_pulls(state=state)
        return [
            {
                "number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "url": pr.html_url,
                "head": pr.head.ref,
                "base": pr.base.ref,
            }
            for pr in prs
        ]
    except GithubException as e:
        logger.error("Failed to list PRs: %s", e)
        return [{"error": str(e)}]


@tool
def update_pull_request(
    pr_number: int,
    title: str | None = None,
    body: str | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    """Update a GitHub pull request.

    Args:
        pr_number: The PR number to update.
        title: New title (optional).
        body: New body (optional).
        state: New state ('open' or 'closed', optional).

    Returns:
        Updated PR details.
    """
    try:
        repo = _get_repo()
        pr = repo.get_pull(pr_number)
        kwargs: dict[str, Any] = {}
        if title is not None:
            kwargs["title"] = title
        if body is not None:
            kwargs["body"] = body
        if state is not None:
            kwargs["state"] = state
        pr.edit(**kwargs)
        return {"number": pr.number, "title": pr.title, "state": pr.state, "url": pr.html_url}
    except GithubException as e:
        logger.error("Failed to update PR #%d: %s", pr_number, e)
        return {"error": str(e)}


@tool
def add_comment_to_pr(pr_number: int, comment: str) -> dict[str, Any]:
    """Add a review comment to a GitHub pull request.

    Args:
        pr_number: The PR number to comment on.
        comment: Comment text in markdown.

    Returns:
        Comment details.
    """
    try:
        repo = _get_repo()
        issue = repo.get_issue(pr_number)
        created = issue.create_comment(comment)
        return {"id": created.id, "url": created.html_url}
    except GithubException as e:
        logger.error("Failed to add comment to PR #%d: %s", pr_number, e)
        return {"error": str(e)}


@tool
def merge_pull_request(
    pr_number: int,
    commit_message: str | None = None,
    merge_method: str = "squash",
) -> dict[str, Any]:
    """Merge a GitHub pull request.

    Args:
        pr_number: The PR number to merge.
        commit_message: Optional custom commit message.
        merge_method: Merge strategy ('merge', 'squash', 'rebase').

    Returns:
        Merge result details.
    """
    try:
        repo = _get_repo()
        pr = repo.get_pull(pr_number)
        kwargs: dict[str, Any] = {"merge_method": merge_method}
        if commit_message:
            kwargs["commit_message"] = commit_message
        result = pr.merge(**kwargs)
        return {"merged": result.merged, "sha": result.sha, "message": result.message}
    except GithubException as e:
        logger.error("Failed to merge PR #%d: %s", pr_number, e)
        return {"error": str(e)}
