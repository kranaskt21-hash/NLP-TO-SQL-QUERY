"""Typed, deterministic summaries shared by both pipelines."""

from pydantic import BaseModel


class GroundedSummary(BaseModel):
    returned_rows: int
    result_truncated: bool
    preview_rows: int
    preview_truncated: bool
    columns: list[str]
    first_row: list | None


def summary_payload(execution, limit=20):
    return GroundedSummary(
        returned_rows=len(execution.rows),
        result_truncated=execution.truncated,
        preview_rows=min(len(execution.rows), limit),
        preview_truncated=len(execution.rows) > limit,
        columns=execution.columns,
        first_row=list(execution.rows[0]) if execution.rows else None,
    )
