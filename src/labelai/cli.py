import asyncio
import typing as T
from pathlib import Path

import cyclopts
import uvloop
from rich.console import Console

import labelai.configs as configs
import labelai.settings as settings

app = cyclopts.App(
    name="Label-AI",
    help="Label-AI - Run dataset labelling pipeline from YAML configuration files.",
)

console = Console()


@app.command
def job(
    name: str,
    *,
    extras: T.Annotated[
        T.Sequence[str] | None,
        cyclopts.Parameter(help="Additional config overrides in key=value format"),
    ] = None,
    config: T.Annotated[
        Path | None,
        cyclopts.Parameter(help="Override config file path"),
    ],
) -> None:
    console.print(f"[blue]Running job:[/blue] {name}")
    console.print(f"[dim]Config:[/dim] {config}")

    merged_config = configs.load_and_merge_configs([config], extras)
    object_ = configs.to_object(merged_config)

    print(object_)

    # Create and run job
    job_settings = settings.JobSettings.model_validate(object_)

    console.print(
        f"[green]✓[/green] Job started: {job_settings.job.__class__.__name__}"
    )
    run_job(job_settings.job.run)
    console.print("[green]✓[/green] Job completed successfully")


def run_job(func: T.Callable) -> None:
    if asyncio.iscoroutinefunction(func):
        uvloop.run(func())
    else:
        func()


def execute(argv: T.Sequence[str] | None = None) -> int:
    try:
        app(argv)
        return 0
    except cyclopts.ValidationError as e:
        # Handle cyclopts ValidationError which may not have a proper __str__ method
        try:
            error_msg = str(e)
        except (NotImplementedError, Exception):
            error_msg = (
                getattr(e, "msg", None)
                or getattr(e, "exception_message", None)
                or "Unknown validation error"
            )
        console.print(f"[red]Error:[/red] {error_msg}")
        return 1
