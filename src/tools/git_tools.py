"""Git operations tools for the multi-agent system."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import git
from langchain_core.tools import tool

from src.config import get_settings

logger = logging.getLogger(__name__)


def _get_workspace() -> Path:
    """Return the workspace directory path."""
    settings = get_settings()
    workspace = Path(settings.workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _configure_git(repo: git.Repo) -> None:
    """Configure git user identity for the repository."""
    settings = get_settings()
    with repo.config_writer() as config:
        config.set_value("user", "name", settings.git_user_name)
        config.set_value("user", "email", settings.git_user_email)


@tool
def clone_repository(url: str, directory: str | None = None) -> dict[str, str]:
    """Clone a git repository into the workspace.

    Args:
        url: Repository URL to clone (HTTPS or SSH).
        directory: Optional subdirectory name within workspace.

    Returns:
        Dictionary with 'path' of cloned repository or 'error'.
    """
    try:
        workspace = _get_workspace()
        settings = get_settings()
        token = settings.github_token.get_secret_value()

        # Inject token into HTTPS URL if available
        if token and url.startswith("https://github.com/"):
            url = url.replace("https://github.com/", f"https://{token}@github.com/")

        dest = workspace / (directory or Path(url.split("/")[-1].removesuffix(".git")))

        if dest.exists():
            logger.info("Repository already exists at %s, pulling latest", dest)
            repo = git.Repo(dest)
            repo.remotes.origin.pull()
            return {"path": str(dest), "status": "pulled"}

        repo = git.Repo.clone_from(url, dest)
        _configure_git(repo)
        logger.info("Cloned repository to %s", dest)
        return {"path": str(dest), "status": "cloned"}
    except git.GitCommandError as e:
        logger.error("Failed to clone repository: %s", e)
        return {"error": str(e)}


@tool
def create_branch(repo_path: str, branch_name: str, from_branch: str = "main") -> dict[str, str]:
    """Create a new branch in a local repository.

    Args:
        repo_path: Absolute path to the local repository.
        branch_name: Name of the new branch.
        from_branch: Base branch to create from (default: 'main').

    Returns:
        Dictionary with branch name or error.
    """
    try:
        repo = git.Repo(repo_path)
        # Fetch latest
        repo.remotes.origin.fetch()
        # Checkout base branch
        repo.git.checkout(from_branch)
        repo.remotes.origin.pull()
        # Create new branch
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()
        logger.info("Created branch '%s' from '%s'", branch_name, from_branch)
        return {"branch": branch_name, "status": "created"}
    except git.GitCommandError as e:
        logger.error("Failed to create branch '%s': %s", branch_name, e)
        return {"error": str(e)}


@tool
def commit_changes(repo_path: str, message: str, add_all: bool = True) -> dict[str, str]:
    """Stage and commit changes in a repository.

    Args:
        repo_path: Absolute path to the local repository.
        message: Commit message.
        add_all: Whether to stage all changes (default: True).

    Returns:
        Dictionary with commit SHA or error.
    """
    try:
        repo = git.Repo(repo_path)
        if add_all:
            repo.git.add(A=True)
        else:
            repo.git.add(update=True)

        if not repo.index.diff("HEAD") and not repo.untracked_files:
            return {"status": "nothing_to_commit", "sha": ""}

        commit = repo.index.commit(message)
        logger.info("Committed changes: %s (%s)", message, commit.hexsha[:8])
        return {"sha": commit.hexsha, "message": message, "status": "committed"}
    except git.GitCommandError as e:
        logger.error("Failed to commit changes: %s", e)
        return {"error": str(e)}


@tool
def push_changes(repo_path: str, branch: str | None = None, force: bool = False) -> dict[str, str]:
    """Push local commits to the remote repository.

    Args:
        repo_path: Absolute path to the local repository.
        branch: Branch to push (defaults to current branch).
        force: Whether to force push (default: False).

    Returns:
        Dictionary with push status or error.
    """
    try:
        repo = git.Repo(repo_path)
        settings = get_settings()
        token = settings.github_token.get_secret_value()

        if not branch:
            branch = repo.active_branch.name

        # Inject credentials into remote URL if needed
        origin = repo.remotes.origin
        parsed = urlparse(origin.url)
        if token and parsed.hostname == "github.com" and not parsed.username:
            new_url = origin.url.replace("https://github.com/", f"https://{token}@github.com/")
            origin.set_url(new_url)

        push_info = origin.push(refspec=branch, force=force)
        for info in push_info:
            if info.flags & git.remote.PushInfo.ERROR:
                return {"error": f"Push failed: {info.summary}"}

        logger.info("Pushed branch '%s' to origin", branch)
        return {"branch": branch, "status": "pushed"}
    except git.GitCommandError as e:
        logger.error("Failed to push changes: %s", e)
        return {"error": str(e)}


@tool
def pull_changes(repo_path: str, branch: str | None = None) -> dict[str, str]:
    """Pull latest changes from the remote repository.

    Args:
        repo_path: Absolute path to the local repository.
        branch: Branch to pull (defaults to current branch).

    Returns:
        Dictionary with pull status or error.
    """
    try:
        repo = git.Repo(repo_path)
        if branch:
            repo.git.checkout(branch)
        repo.remotes.origin.pull()
        logger.info("Pulled latest changes for branch '%s'", repo.active_branch.name)
        return {"branch": repo.active_branch.name, "status": "pulled"}
    except git.GitCommandError as e:
        logger.error("Failed to pull changes: %s", e)
        return {"error": str(e)}


@tool
def get_diff(repo_path: str, staged: bool = False) -> dict[str, str]:
    """Get the current diff of a repository.

    Args:
        repo_path: Absolute path to the local repository.
        staged: If True, show staged diff; otherwise show unstaged (default: False).

    Returns:
        Dictionary with diff text or error.
    """
    try:
        repo = git.Repo(repo_path)
        if staged:
            diff = repo.git.diff("--cached")
        else:
            diff = repo.git.diff()
        return {"diff": diff, "status": "ok"}
    except git.GitCommandError as e:
        logger.error("Failed to get diff: %s", e)
        return {"error": str(e)}


@tool
def merge_branch(repo_path: str, source_branch: str, target_branch: str = "main") -> dict[str, str]:
    """Merge a source branch into a target branch locally.

    Args:
        repo_path: Absolute path to the local repository.
        source_branch: Branch to merge from.
        target_branch: Branch to merge into (default: 'main').

    Returns:
        Dictionary with merge status or error.
    """
    try:
        repo = git.Repo(repo_path)
        repo.git.checkout(target_branch)
        repo.git.merge(source_branch)
        logger.info("Merged '%s' into '%s'", source_branch, target_branch)
        return {"status": "merged", "source": source_branch, "target": target_branch}
    except git.GitCommandError as e:
        logger.error("Failed to merge '%s' into '%s': %s", source_branch, target_branch, e)
        return {"error": str(e)}
