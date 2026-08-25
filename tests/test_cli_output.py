from __future__ import annotations

import warnings

from cypher_3dsg_agent_distillation import cli_output


def test_captured_warnings_groups_warning_details(monkeypatch):
    rendered = []
    monkeypatch.setattr(
        cli_output,
        "print_warnings",
        lambda title, rows: rendered.append((title, list(rows))),
    )

    with cli_output.captured_warnings("Grouped warnings"):
        warnings.warn("first warning", UserWarning, stacklevel=1)
        warnings.warn("second warning", RuntimeWarning, stacklevel=1)

    assert rendered[0][0] == "Grouped warnings"
    assert [(category, message) for category, message, _source in rendered[0][1]] == [
        ("UserWarning", "first warning"),
        ("RuntimeWarning", "second warning"),
    ]


def test_fallback_warning_renderer_uses_normalized_warning_table(monkeypatch):
    rendered = []
    monkeypatch.setattr(
        cli_output,
        "print_warnings",
        lambda title, rows: rendered.append((title, list(rows))),
    )

    cli_output._show_warning(
        "something happened",
        UserWarning,
        "/workspace/example.py",
        12,
    )

    assert rendered == [
        ("Warning", [("UserWarning", "something happened", "example.py:12")])
    ]
