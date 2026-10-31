"""Command line entry point.

The CLI is deliberately thin: every command either reports something about a
store or, in the case of ``serve-mcp``, hands control to a transport loop. All
the interesting logic lives in the library so both remain testable.
"""

from __future__ import annotations

import asyncio

import typer

app = typer.Typer(
    name="sable-vector",
    no_args_is_help=True,
    add_completion=False,
    help="Embedded hybrid vector search for local AI agent runtimes.",
)


@app.callback()
def main() -> None:
    """Embedded hybrid vector search for local AI agent runtimes."""


@app.command("serve-mcp")
def serve_mcp() -> None:
    """Start the Model Context Protocol server on stdio.

    This blocks: it is the transport loop, and it returns only when the client
    disconnects. MCP is asynchronous while Typer is synchronous, so the loop is
    driven from a single ``asyncio.run`` rather than mixed into the command.
    """
    try:
        from sable_vector.adapter.mcp_server import run_server
    except ImportError:
        typer.echo(
            "serve-mcp needs the 'mcp' extra:  pip install \"sable-vector[mcp]\"",
            err=True,
        )
        raise typer.Exit(code=1) from None

    asyncio.run(run_server())


if __name__ == "__main__":  # python -m sable_vector.cli
    app()
