from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.traceback import install as install_rich_traceback

console = Console()


def configure_cli_output() -> None:
    """Install consistent warning and uncaught-error rendering for script entry points."""
    install_rich_traceback(console=console, show_locals=False, word_wrap=True)
    warnings.showwarning = _show_warning


def _show_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any = None,
    line: str | None = None,
) -> None:
    del file, line
    print_warnings(
        "Warning",
        [(category.__name__, str(message), f"{Path(filename).name}:{lineno}")],
    )


def _display_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def print_summary(title: str, rows: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> None:
    """Print a compact, consistent completion summary for a CLI command."""
    items = rows.items() if isinstance(rows, Mapping) else rows
    table = Table(
        title=title,
        title_style="bold cyan",
        box=box.ROUNDED,
        show_header=False,
        pad_edge=True,
    )
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for label, value in items:
        table.add_row(str(label), _display_value(value))
    console.print(table)


def print_breakdown(
    title: str,
    headers: Iterable[str],
    rows: Iterable[Iterable[Any]],
    *,
    right_align: Iterable[int] = (),
) -> None:
    """Print a normalized table for per-category or per-split results."""
    aligned = set(right_align)
    table = Table(title=title, title_style="bold cyan", box=box.SIMPLE_HEAVY)
    for index, header in enumerate(headers):
        table.add_column(str(header), justify="right" if index in aligned else "left")
    for row in rows:
        table.add_row(*(_display_value(value) for value in row))
    console.print(table)


def print_warnings(
    title: str,
    warning_rows: Iterable[tuple[str, str, str]],
) -> None:
    rows = list(warning_rows)
    if not rows:
        return
    table = Table(title=title, title_style="bold yellow", box=box.ROUNDED)
    table.add_column("Category", style="yellow", no_wrap=True)
    table.add_column("Message", overflow="fold")
    table.add_column("Source", style="dim", no_wrap=True)
    for category, message, source in rows:
        table.add_row(category, message, source)
    console.print(table)


@contextmanager
def captured_warnings(title: str):
    """Collect expected warnings and render them once without raw Python locations."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    print_warnings(
        title,
        (
            (
                warning.category.__name__,
                str(warning.message),
                f"{Path(warning.filename).name}:{warning.lineno}",
            )
            for warning in caught
        ),
    )


@contextmanager
def phase_progress(description: str, total: int, *, enabled: bool = True):
    """Create a repository-owned Rich progress bar for coarse workflow phases."""
    if not enabled:
        yield None, None
        return
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(description, total=total)
        yield progress, task_id
