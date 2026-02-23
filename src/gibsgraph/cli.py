"""GibsGraph CLI — `gibsgraph ask "your question"`"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

log = structlog.get_logger(__name__)
console = Console()

_VERSION = "0.1.1"


def main() -> None:
    """CLI entrypoint."""
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        console.print("[bold]GibsGraph[/] — natural language queries for Neo4j\n")
        console.print("[bold]Usage:[/]")
        console.print("  gibsgraph ask <question>    Ask a question about your graph")
        console.print("  gibsgraph ingest <file>     Ingest a text file into the graph")
        console.print("  gibsgraph --version         Show version")
        console.print("  gibsgraph --help            Show this help\n")
        console.print("[dim]Set NEO4J_PASSWORD and OPENAI_API_KEY (or ANTHROPIC_API_KEY) first.[/]")
        sys.exit(0)

    command = sys.argv[1]

    if command == "--version":
        console.print(f"gibsgraph {_VERSION}")
        sys.exit(0)

    if command == "ask":
        if len(sys.argv) < 3:
            console.print("[bold red]Error:[/] Please provide a question.")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        _cmd_ask(question)

    elif command == "ingest":
        if len(sys.argv) < 3:
            console.print("[bold red]Error:[/] Please provide a file path.")
            sys.exit(1)
        path = sys.argv[2]
        _cmd_ingest(path)

    else:
        console.print(f"[bold red]Unknown command:[/] {command}")
        console.print("Run [bold]gibsgraph --help[/] for usage.")
        sys.exit(1)


def _cmd_ask(question: str) -> None:
    from gibsgraph import Graph

    console.print(f"\n[bold cyan]GibsGraph[/] — asking: [italic]{question}[/]\n")

    try:
        with Graph() as g:
            result = g.ask(question)

        console.print(Panel(Text(result.answer), title="Answer", border_style="green"))

        if result.cypher:
            console.print(f"\n[dim]Cypher used:[/]\n{result.cypher}")
        if result.bloom_url:
            console.print(f"\n[dim]Bloom visualization:[/] {result.bloom_url}")
        if result.visualization:
            console.print(f"\n[dim]Mermaid:[/]\n{result.visualization}")
        if result.errors:
            console.print(f"\n[yellow]Warnings:[/] {result.errors}")

    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        log.exception("cli.ask_failed")
        sys.exit(1)


def _cmd_ingest(path: str) -> None:
    from gibsgraph import Graph

    console.print(f"\n[bold cyan]GibsGraph[/] — ingesting: [italic]{path}[/]\n")

    try:
        text = Path(path).read_text(encoding="utf-8")
        with Graph(read_only=False) as g:
            result = g.ingest(text, source=path)
        console.print(f"[green]{result}[/]")
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        log.exception("cli.ingest_failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
