"""GibsGraph CLI — `gibsgraph ask "your question"`"""

from __future__ import annotations

import sys

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

log = structlog.get_logger(__name__)
console = Console()


def main() -> None:
    """CLI entrypoint."""
    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/] gibsgraph ask <question>")
        console.print("       gibsgraph ingest <file.txt>")
        sys.exit(1)

    command = sys.argv[1]

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
        sys.exit(1)


def _cmd_ask(question: str) -> None:
    from gibsgraph import Graph

    console.print(f"\n[bold cyan]GibsGraph[/] — asking: [italic]{question}[/]\n")

    try:
        g = Graph()
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
        g = Graph(read_only=False)
        result = g.ingest(open(path).read(), source=path)  # noqa: WPS515
        console.print(f"[green]✓[/] {result}")
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
