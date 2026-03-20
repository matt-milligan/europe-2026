#!/usr/bin/env python3
"""
GitHub Learning Tool — search GitHub for repos on topics you're interested in,
then ask Claude to explain how the code works so you can learn.

Usage:
    python github_learn.py
    python github_learn.py --topic "async python web scraping"

Requires:
    ANTHROPIC_API_KEY  — your Anthropic API key
    GITHUB_TOKEN       — (optional) GitHub personal access token for higher rate limits
"""

import os
import sys
import json
import textwrap
import argparse
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional

import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import box

console = Console()

# ── GitHub helpers ────────────────────────────────────────────────────────────

GITHUB_API = "https://api.github.com"

def github_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "github-learn-tool/1.0",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_get(path: str, params: Optional[dict] = None) -> dict:
    url = f"{GITHUB_API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=github_headers())
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        console.print(f"[red]GitHub API error {e.code}:[/red] {body[:200]}")
        return {}


def search_repos(query: str, sort: str = "stars", per_page: int = 10) -> list[dict]:
    data = github_get("/search/repositories", {
        "q": query,
        "sort": sort,
        "order": "desc",
        "per_page": per_page,
    })
    return data.get("items", [])


def get_repo_contents(owner: str, repo: str, path: str = "") -> list[dict]:
    data = github_get(f"/repos/{owner}/{repo}/contents/{path}")
    if isinstance(data, list):
        return data
    return []


def fetch_file_content(owner: str, repo: str, file_path: str) -> Optional[str]:
    """Fetch a file's raw content (text files only, up to ~100 KB)."""
    import base64
    data = github_get(f"/repos/{owner}/{repo}/contents/{file_path}")
    if not isinstance(data, dict):
        return None
    encoding = data.get("encoding")
    content = data.get("content", "")
    if encoding == "base64":
        try:
            decoded = base64.b64decode(content).decode("utf-8", errors="replace")
            return decoded[:100_000]  # cap at 100 KB
        except Exception:
            return None
    return content or None


def get_readme(owner: str, repo: str) -> Optional[str]:
    """Fetch the README for a repo."""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        content = fetch_file_content(owner, repo, name)
        if content:
            return content
    return None


def list_interesting_files(owner: str, repo: str, max_files: int = 20) -> list[str]:
    """Return a flat list of interesting source files (skipping binaries/build artifacts)."""
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                 "dist", "build", ".next", "vendor", "target"}
    interesting_ext = {
        ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h",
        ".rb", ".php", ".swift", ".kt", ".cs", ".sh", ".yaml", ".yml",
        ".toml", ".json", ".md",
    }

    found: list[str] = []

    def walk(path: str, depth: int = 0):
        if depth > 3 or len(found) >= max_files:
            return
        items = get_repo_contents(owner, repo, path)
        for item in items:
            if len(found) >= max_files:
                break
            name = item.get("name", "")
            item_type = item.get("type")
            item_path = item.get("path", "")
            if item_type == "dir":
                if name not in skip_dirs:
                    walk(item_path, depth + 1)
            elif item_type == "file":
                _, ext = os.path.splitext(name)
                if ext.lower() in interesting_ext or name in ("Makefile", "Dockerfile"):
                    found.append(item_path)

    walk("")
    return found


# ── Claude helpers ─────────────────────────────────────────────────────────────

def make_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY environment variable not set.")
        console.print("Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    return anthropic.Anthropic(api_key=key)


def ask_claude(client: anthropic.Anthropic, system: str, user: str) -> str:
    """Stream a response from Claude and return the full text."""
    collected: list[str] = []
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    collected.append(event.delta.text)
                    console.print(event.delta.text, end="", markup=False)
    console.print()
    return "".join(collected)


# ── UI helpers ─────────────────────────────────────────────────────────────────

def display_repos(repos: list[dict]) -> None:
    table = Table(
        title="Search Results",
        box=box.ROUNDED,
        show_lines=False,
        highlight=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Repository", style="bold cyan", no_wrap=True)
    table.add_column("Stars", justify="right", style="yellow", width=7)
    table.add_column("Language", width=12)
    table.add_column("Description")

    for i, repo in enumerate(repos, 1):
        table.add_row(
            str(i),
            repo["full_name"],
            str(repo.get("stargazers_count", 0)),
            repo.get("language") or "—",
            (repo.get("description") or "")[:80],
        )
    console.print(table)


def display_files(files: list[str]) -> None:
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("File", style="cyan")
    for i, f in enumerate(files, 1):
        table.add_row(str(i), f)
    console.print(table)


def show_file(owner: str, repo: str, file_path: str) -> Optional[str]:
    content = fetch_file_content(owner, repo, file_path)
    if not content:
        console.print("[red]Could not fetch file.[/red]")
        return None
    _, ext = os.path.splitext(file_path)
    lang_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
        ".sh": "bash", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".json": "json", ".md": "markdown",
        ".c": "c", ".cpp": "cpp", ".h": "c",
    }
    lang = lang_map.get(ext.lower(), "text")
    syntax = Syntax(content, lang, theme="monokai", line_numbers=True,
                    word_wrap=False)
    console.print(Panel(syntax, title=file_path, border_style="dim"))
    return content


# ── Explanation flows ──────────────────────────────────────────────────────────

SYSTEM_TEACHER = textwrap.dedent("""
    You are an expert software engineer and patient teacher. When explaining
    code you:
    - Start with the big picture (what this code does and why)
    - Explain key concepts and patterns used
    - Walk through important sections with clear, plain-English explanations
    - Highlight anything clever, unusual, or worth learning
    - Suggest what topics the learner should study next to go deeper

    Keep explanations friendly and educational, not overly terse or academic.
    Use markdown formatting (headings, code blocks, bullet points) so your
    explanations are easy to read.
""").strip()


def explain_file(client: anthropic.Anthropic, owner: str, repo: str,
                 file_path: str, extra_context: str = "") -> None:
    content = fetch_file_content(owner, repo, file_path)
    if not content:
        console.print("[red]Could not fetch file content.[/red]")
        return

    console.print(Panel(
        f"Explaining [bold cyan]{file_path}[/bold cyan] from [bold]{owner}/{repo}[/bold]",
        border_style="blue",
    ))

    prompt = f"Please explain this file from the `{owner}/{repo}` repository.\n\n"
    if extra_context:
        prompt += f"Context: {extra_context}\n\n"
    prompt += f"File: `{file_path}`\n\n```\n{content[:20_000]}\n```"

    ask_claude(client, SYSTEM_TEACHER, prompt)


def explain_repo_overview(client: anthropic.Anthropic, repo: dict) -> None:
    owner, name = repo["full_name"].split("/", 1)
    readme = get_readme(owner, name) or "(no README found)"

    console.print(Panel(
        f"Giving an overview of [bold cyan]{repo['full_name']}[/bold cyan]",
        border_style="blue",
    ))

    prompt = textwrap.dedent(f"""
        Repository: {repo['full_name']}
        Stars: {repo.get('stargazers_count', 0)}
        Language: {repo.get('language', 'unknown')}
        Description: {repo.get('description', 'none')}
        Topics: {', '.join(repo.get('topics', [])) or 'none'}

        README (first 8000 chars):
        ---
        {readme[:8000]}
        ---

        Please give me a thorough overview of this project: what it does, why it
        exists, the main technologies used, and what I can learn from studying it.
    """).strip()

    ask_claude(client, SYSTEM_TEACHER, prompt)


def freeform_question(client: anthropic.Anthropic, repo: dict,
                      selected_files: list[tuple[str, str]],
                      question: str) -> None:
    owner, name = repo["full_name"].split("/", 1)

    parts = [
        f"Repository: {repo['full_name']}",
        f"Language: {repo.get('language', 'unknown')}",
        f"Description: {repo.get('description', '')}",
        "",
        "Files loaded for context:",
    ]
    for fpath, fcontent in selected_files:
        parts.append(f"\n--- {fpath} ---\n```\n{fcontent[:8000]}\n```")

    parts.append(f"\nMy question: {question}")

    ask_claude(client, SYSTEM_TEACHER, "\n".join(parts))


# ── Main interactive loop ──────────────────────────────────────────────────────

def repo_session(client: anthropic.Anthropic, repo: dict) -> None:
    """Interactive learning session for a single repo."""
    owner, name = repo["full_name"].split("/", 1)
    console.print(Panel(
        f"[bold green]Now exploring:[/bold green] [bold cyan]{repo['full_name']}[/bold cyan]\n"
        f"{repo.get('description', '')}\n"
        f"⭐ {repo.get('stargazers_count', 0):,}  •  "
        f"Language: {repo.get('language', '?')}  •  "
        f"[dim]{repo.get('html_url', '')}[/dim]",
        border_style="green",
    ))

    loaded_files: list[tuple[str, str]] = []  # (path, content)

    while True:
        console.print("\n[bold]What would you like to do?[/bold]")
        console.print("  [cyan]1[/cyan]  Get an overview of this repo")
        console.print("  [cyan]2[/cyan]  Browse and explain a specific file")
        console.print("  [cyan]3[/cyan]  Load files into context & ask a question")
        console.print("  [cyan]4[/cyan]  Ask a freeform question")
        console.print("  [cyan]b[/cyan]  Back to search results")
        console.print("  [cyan]q[/cyan]  Quit")

        choice = Prompt.ask("\nChoice", default="1")

        if choice == "1":
            explain_repo_overview(client, repo)

        elif choice == "2":
            with console.status("Fetching file list…"):
                files = list_interesting_files(owner, name)
            if not files:
                console.print("[yellow]No source files found.[/yellow]")
                continue
            display_files(files)
            idx = IntPrompt.ask("File number to explain", default=1)
            if 1 <= idx <= len(files):
                show_file(owner, name, files[idx - 1])
                console.print()
                explain_file(client, owner, name, files[idx - 1])

        elif choice == "3":
            with console.status("Fetching file list…"):
                files = list_interesting_files(owner, name)
            if not files:
                console.print("[yellow]No source files found.[/yellow]")
                continue
            display_files(files)
            raw = Prompt.ask("Enter file numbers to load (comma-separated, e.g. 1,3,5)", default="1")
            indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            loaded_files = []
            for i in indices:
                if 1 <= i <= len(files):
                    fpath = files[i - 1]
                    with console.status(f"Fetching {fpath}…"):
                        fcontent = fetch_file_content(owner, name, fpath) or ""
                    loaded_files.append((fpath, fcontent))
                    console.print(f"  ✓ Loaded [cyan]{fpath}[/cyan] ({len(fcontent):,} chars)")
            if loaded_files:
                question = Prompt.ask("\nWhat would you like to know about these files?")
                console.print()
                freeform_question(client, repo, loaded_files, question)

        elif choice == "4":
            question = Prompt.ask("Your question")
            console.print()
            freeform_question(client, repo, loaded_files, question)

        elif choice.lower() == "b":
            break

        elif choice.lower() == "q":
            console.print("[dim]Goodbye![/dim]")
            sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search GitHub and learn from code with Claude's help.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python github_learn.py
              python github_learn.py --topic "async Python HTTP server"
              python github_learn.py --topic "rust embedded systems" --sort updated
        """),
    )
    parser.add_argument("--topic", "-t", help="Initial search topic")
    parser.add_argument(
        "--sort", choices=["stars", "updated", "forks", "help-wanted-issues"],
        default="stars", help="Sort order for results (default: stars)",
    )
    parser.add_argument(
        "--results", "-n", type=int, default=8,
        help="Number of results to show (default: 8)",
    )
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]GitHub Learning Tool[/bold cyan]\n"
        "[dim]Search GitHub for interesting repos, then ask Claude to explain the code.[/dim]",
        border_style="cyan",
    ))

    client = make_client()
    repos: list[dict] = []

    topic = args.topic
    if not topic:
        topic = Prompt.ask("\nWhat topic or technology are you interested in learning?")

    while True:
        if not repos or topic:
            with console.status(f"Searching GitHub for [bold]{topic}[/bold]…"):
                repos = search_repos(topic, sort=args.sort, per_page=args.results)
            if not repos:
                console.print("[yellow]No repositories found. Try a different search.[/yellow]")
                topic = Prompt.ask("Search topic")
                continue
            console.print()
            display_repos(repos)
            topic = ""  # clear so we don't re-search unless user asks

        console.print("\n[bold]Options:[/bold]")
        console.print("  [cyan]1–" + str(len(repos)) + "[/cyan]  Open a repository")
        console.print("  [cyan]s[/cyan]    New search")
        console.print("  [cyan]q[/cyan]    Quit")

        choice = Prompt.ask("\nChoice", default="1")

        if choice.lower() == "s":
            topic = Prompt.ask("New search topic")
            repos = []

        elif choice.lower() == "q":
            console.print("[dim]Goodbye![/dim]")
            break

        elif choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(repos):
                repo_session(client, repos[idx - 1])
            else:
                console.print(f"[yellow]Please enter a number between 1 and {len(repos)}.[/yellow]")
        else:
            console.print("[yellow]Unrecognised option.[/yellow]")


if __name__ == "__main__":
    main()
