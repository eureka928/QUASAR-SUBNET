#!/usr/bin/env python3
"""
Git Push Helper - Commit and push to GitHub with PAT authentication

Usage:
    python scripts/git_push_helper.py \
        --repo-path ./quasar_work/flash-linear-attention \
        --github-username YOUR_USERNAME \
        --github-token YOUR_PAT \
        --commit-message "Your commit message"
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command and return output."""
    print(f"  > {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0 and check:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout.strip()


def configure_git(repo_path, username, email=None):
    """Configure git user for the repository."""
    if email is None:
        email = f"{username}@users.noreply.github.com"

    print("\n[1] Configuring git user...")
    run_cmd(f'git config user.name "{username}"', cwd=repo_path)
    run_cmd(f'git config user.email "{email}"', cwd=repo_path)
    print(f"  Configured: {username} <{email}>")


def set_remote_with_pat(repo_path, username, token, repo_name="flash-linear-attention"):
    """Set remote URL with PAT authentication."""
    print("\n[2] Setting remote URL with authentication...")

    # URL format: https://USERNAME:TOKEN@github.com/USERNAME/REPO.git
    remote_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"

    # Check if origin exists
    result = run_cmd("git remote -v", cwd=repo_path, check=False)

    if "origin" in (result or ""):
        run_cmd(f'git remote set-url origin "{remote_url}"', cwd=repo_path)
        print("  Updated origin remote")
    else:
        run_cmd(f'git remote add origin "{remote_url}"', cwd=repo_path)
        print("  Added origin remote")

    # Don't print the token
    print(f"  Remote: https://{username}:****@github.com/{username}/{repo_name}.git")


def stage_and_commit(repo_path, message, files=None):
    """Stage files and create commit."""
    print("\n[3] Staging and committing...")

    # Check status first
    status = run_cmd("git status --porcelain", cwd=repo_path)
    if not status:
        print("  No changes to commit")
        return False

    # Stage files
    if files:
        for f in files:
            run_cmd(f'git add "{f}"', cwd=repo_path)
    else:
        run_cmd("git add -A", cwd=repo_path)

    # Commit
    # Use a temp file for the commit message to handle special characters
    msg_file = Path(repo_path) / ".commit_msg_temp"
    with open(msg_file, 'w') as f:
        f.write(message)

    result = run_cmd(f'git commit -F "{msg_file}"', cwd=repo_path, check=False)
    msg_file.unlink()  # Delete temp file

    if result is None:
        print("  Commit may have failed or no changes")
        return False

    print("  Committed successfully")
    return True


def push_to_remote(repo_path, branch="main", force=False):
    """Push to remote repository."""
    print("\n[4] Pushing to remote...")

    force_flag = "--force" if force else ""
    result = run_cmd(f"git push {force_flag} -u origin {branch}", cwd=repo_path, check=False)

    if result is None:
        print("  Push may have failed - check if you have write access")
        return False

    print(f"  Pushed to origin/{branch}")
    return True


def squash_commits(repo_path, num_commits, message):
    """Squash last N commits into one."""
    print(f"\n[*] Squashing last {num_commits} commits...")

    run_cmd(f"git reset --soft HEAD~{num_commits}", cwd=repo_path)

    # Use temp file for commit message
    msg_file = Path(repo_path) / ".commit_msg_temp"
    with open(msg_file, 'w') as f:
        f.write(message)

    run_cmd(f'git commit -F "{msg_file}"', cwd=repo_path)
    msg_file.unlink()

    print(f"  Squashed {num_commits} commits into one")


def main():
    parser = argparse.ArgumentParser(description="Git Push Helper with PAT authentication")
    parser.add_argument("--repo-path", required=True, help="Path to git repository")
    parser.add_argument("--github-username", required=True, help="GitHub username")
    parser.add_argument("--github-token", required=True, help="GitHub Personal Access Token")
    parser.add_argument("--github-email", help="GitHub email (optional)")
    parser.add_argument("--repo-name", default="flash-linear-attention", help="Repository name")
    parser.add_argument("--branch", default="main", help="Branch to push to")
    parser.add_argument("--commit-message", "-m", help="Commit message")
    parser.add_argument("--files", nargs="*", help="Specific files to stage (default: all)")
    parser.add_argument("--force", action="store_true", help="Force push")
    parser.add_argument("--squash", type=int, help="Squash last N commits before pushing")
    parser.add_argument("--skip-commit", action="store_true", help="Skip commit, only push")

    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()

    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    if not (repo_path / ".git").exists():
        print(f"Error: Not a git repository: {repo_path}")
        sys.exit(1)

    print("=" * 60)
    print("  Git Push Helper")
    print("=" * 60)
    print(f"Repository: {repo_path}")
    print(f"Username: {args.github_username}")
    print(f"Repo: {args.repo_name}")
    print(f"Branch: {args.branch}")

    # Configure git
    configure_git(repo_path, args.github_username, args.github_email)

    # Set remote with PAT
    set_remote_with_pat(repo_path, args.github_username, args.github_token, args.repo_name)

    # Squash if requested
    if args.squash:
        if not args.commit_message:
            print("Error: --commit-message required when squashing")
            sys.exit(1)
        squash_commits(repo_path, args.squash, args.commit_message)

    # Stage and commit
    if not args.skip_commit and not args.squash:
        if not args.commit_message:
            print("Error: --commit-message required")
            sys.exit(1)
        stage_and_commit(repo_path, args.commit_message, args.files)

    # Push
    success = push_to_remote(repo_path, args.branch, args.force)

    if success:
        print("\n" + "=" * 60)
        print("  SUCCESS!")
        print("=" * 60)
        print(f"\nYour fork URL for submission:")
        print(f"  https://github.com/{args.github_username}/{args.repo_name}")

        # Get commit hash
        commit_hash = run_cmd("git rev-parse HEAD", cwd=repo_path)
        print(f"\nCommit hash: {commit_hash}")
    else:
        print("\n" + "=" * 60)
        print("  FAILED - Check errors above")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
