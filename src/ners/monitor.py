#!.venv/bin/python3
from ners.processing.monitoring.pipeline_monitor import PipelineMonitor


def status(*, detailed: bool = False) -> None:
    PipelineMonitor().print_status(detailed=detailed)


def clean_step(step: str, *, keep_last: int = 1) -> None:
    PipelineMonitor().clean_step_checkpoints(step, keep_last)


def reset_step(step: str) -> None:
    PipelineMonitor().reset_step(step)
