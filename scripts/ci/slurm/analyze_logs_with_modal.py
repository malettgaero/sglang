#!/usr/bin/env python3
"""Analyze srtslurm logs with Claude Code inside a Modal sandbox.

This script accepts either:
- a local log directory
- a `.tar.gz` bundle such as `multinode_server_logs.tar.gz`

It uploads the logs into an ephemeral Modal sandbox, installs and runs
Claude Code with an analysis prompt, and prints the resulting markdown.

Example:
  uv run --with modal python scripts/ci/slurm/analyze_logs_with_modal.py \
    --tarball /tmp/multinode_server_logs.tar.gz \
    --job-id 4645
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

try:
    import modal
except ImportError:  # pragma: no cover - runtime guard for local usage
    modal = None


logger = logging.getLogger("slurm_log_analysis")

SANDBOX_TIMEOUT = 600
DEFAULT_REPOS = [
    "https://github.com/sgl-project/sglang.git",
]
PROMPT_PATH = Path(__file__).with_name("log_analysis_prompt.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a srtslurm log bundle with Claude Code in Modal."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tarball",
        type=Path,
        help="Path to a local multinode_server_logs.tar.gz bundle.",
    )
    source.add_argument(
        "--log-dir",
        type=Path,
        help="Path to an unpacked log directory.",
    )
    parser.add_argument(
        "--job-id",
        default="unknown",
        help="Job identifier used in the report header and logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the markdown analysis.",
    )
    parser.add_argument(
        "--repo-url",
        action="append",
        dest="repo_urls",
        help=(
            "Optional extra repo URL to clone into the sandbox for context. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=SANDBOX_TIMEOUT,
        help="Sandbox lifetime in seconds.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def build_sandbox_image() -> "modal.Image":
    if modal is None:
        raise RuntimeError(
            "The 'modal' package is required. Run this script with "
            "`uv run --with modal python ...` or install modal locally."
        )

    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("bash", "curl", "git", "gh", "sudo")
        .run_commands(
            "useradd -m -s /bin/bash claude",
            "su - claude -c 'curl -fsSL https://claude.ai/install.sh | bash'",
        )
        .env(
            {
                "PATH": "/home/claude/.local/bin:$PATH",
                "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
                "ANTHROPIC_API_KEY": "",
            }
        )
    )


def prepare_log_dir(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.log_dir:
        if not args.log_dir.is_dir():
            raise FileNotFoundError(f"log directory not found: {args.log_dir}")
        return args.log_dir.resolve(), None

    assert args.tarball is not None
    if not args.tarball.is_file():
        raise FileNotFoundError(f"tarball not found: {args.tarball}")

    temp_dir = Path(tempfile.mkdtemp(prefix="sglang_logs_"))
    with tarfile.open(args.tarball, "r:gz") as archive:
        archive.extractall(temp_dir)
    return temp_dir.resolve(), temp_dir


def build_prompt(job_id: str, repo_urls: list[str]) -> str:
    skill_content = PROMPT_PATH.read_text()
    repo_lines = []
    for repo_url in repo_urls:
        repo_name = repo_url.rsplit("/", 1)[-1].removesuffix(".git")
        repo_lines.append(f"- **{repo_name} repo**: `/home/claude/repos/{repo_name}/`")
    repo_section = "\n".join(repo_lines) if repo_lines else "- No extra repos were requested."

    return f"""{skill_content}

---

## Your Environment

- **Logs**: `/home/claude/logs/`
- **GitHub CLI**: `gh` is installed and authenticated if `GH_TOKEN` is present
{repo_section}

## Task

Analyze the log bundle for job `{job_id}`.

1. Read the logs in the priority order described above.
2. Correlate timestamps to identify the most likely root cause.
3. Use `gh` to inspect related PRs or commits if it materially helps.
4. Write the final markdown report to `/home/claude/logs/ai_analysis.md`.

Keep the report concise but concrete. Include evidence, likely root cause, and notable noise.
"""


def upload_tree(sandbox: "modal.Sandbox", log_dir: Path) -> None:
    log_files = [path for path in log_dir.rglob("*") if path.is_file()]
    logger.info("Uploading %d log files into the sandbox", len(log_files))
    for index, log_file in enumerate(log_files, start=1):
        rel_path = log_file.relative_to(log_dir)
        remote_path = Path("/home/claude/logs") / rel_path
        sandbox.exec("mkdir", "-p", str(remote_path.parent)).wait()
        with sandbox.open(str(remote_path), "wb") as handle:
            handle.write(log_file.read_bytes())
        if index % 10 == 0 or index == len(log_files):
            logger.info("Uploaded %d/%d files", index, len(log_files))
    sandbox.exec("chown", "-R", "claude:claude", "/home/claude/logs").wait()


def clone_context_repos(sandbox: "modal.Sandbox", repo_urls: list[str]) -> None:
    if not repo_urls:
        return

    sandbox.exec(
        "su",
        "-",
        "claude",
        "-c",
        "mkdir -p /home/claude/repos",
    ).wait()

    for repo_url in repo_urls:
        repo_name = repo_url.rsplit("/", 1)[-1].removesuffix(".git")
        logger.info("Cloning %s into the sandbox", repo_name)
        sandbox.exec(
            "su",
            "-",
            "claude",
            "-c",
            f"git clone --depth 100 {repo_url} /home/claude/repos/{repo_name}",
        ).wait()


def run_claude_analysis(
    *,
    log_dir: Path,
    job_id: str,
    repo_urls: list[str],
    timeout_seconds: int,
) -> str:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    gh_token = os.environ.get("GH_TOKEN")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set to run Claude analysis.")

    prompt = build_prompt(job_id, repo_urls)
    app = modal.App.lookup("sglang-log-analyzer", create_if_missing=True)
    sandbox = modal.Sandbox.create(
        app=app,
        image=build_sandbox_image(),
        timeout=timeout_seconds,
    )
    logger.info("Created Modal sandbox %s", sandbox.object_id)

    try:
        sandbox.exec(
            "su",
            "-",
            "claude",
            "-c",
            "mkdir -p /home/claude/logs /home/claude/repos",
        ).wait()

        clone_context_repos(sandbox, repo_urls)
        upload_tree(sandbox, log_dir)

        with sandbox.open("/home/claude/prompt.txt", "w") as handle:
            handle.write(prompt)
        sandbox.exec("chown", "claude:claude", "/home/claude/prompt.txt").wait()

        env_content = (
            "export ANTHROPIC_BASE_URL='https://openrouter.ai/api'\n"
            f"export ANTHROPIC_AUTH_TOKEN='{openrouter_key}'\n"
            "export ANTHROPIC_API_KEY=''\n"
        )
        if gh_token:
            env_content += f"export GH_TOKEN='{gh_token}'\n"

        with sandbox.open("/home/claude/.claude_env", "w") as handle:
            handle.write(env_content)
        sandbox.exec("chmod", "600", "/home/claude/.claude_env").wait()
        sandbox.exec("chown", "claude:claude", "/home/claude/.claude_env").wait()

        runner_script = """#!/bin/bash
set -euo pipefail
source /home/claude/.claude_env
cd /home/claude/logs
claude -p "$(cat /home/claude/prompt.txt)" --dangerously-skip-permissions
"""
        with sandbox.open("/home/claude/run_claude.sh", "w") as handle:
            handle.write(runner_script)
        sandbox.exec("chmod", "+x", "/home/claude/run_claude.sh").wait()
        sandbox.exec("chown", "claude:claude", "/home/claude/run_claude.sh").wait()

        logger.info("Running Claude Code analysis")
        process = sandbox.exec(
            "su",
            "-",
            "claude",
            "-c",
            "/home/claude/run_claude.sh",
            pty=True,
        )
        process.wait()

        stderr = process.stderr.read()
        if stderr:
            logger.warning("Claude stderr: %s", stderr[:500])

        try:
            with sandbox.open("/home/claude/logs/ai_analysis.md", "r") as handle:
                return handle.read()
        except Exception:
            stdout = process.stdout.read()
            if stdout:
                return stdout
            raise RuntimeError("Claude analysis completed without producing output.")
    finally:
        sandbox.terminate()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    repo_urls = list(DEFAULT_REPOS)
    if args.repo_urls:
        repo_urls.extend(args.repo_urls)

    log_dir, cleanup_dir = prepare_log_dir(args)
    try:
        analysis = run_claude_analysis(
            log_dir=log_dir,
            job_id=args.job_id,
            repo_urls=repo_urls,
            timeout_seconds=args.timeout_seconds,
        )
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    print(analysis)
    if args.output:
        args.output.write_text(analysis)
        logger.info("Wrote analysis to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
