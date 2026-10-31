"""Text table for one benchmark run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.metrics import Metrics


@dataclass(frozen=True)
class Record:
    case: str
    controller: str
    metrics: Metrics


def format_report(preamble: str, records: list[Record]) -> str:
    return preamble.rstrip() + "\n\n" + format_table(records) + "\n"


def format_table(records: list[Record]) -> str:
    header = (
        "case",
        "controller",
        "settling_s",
        "peak_force_N",
        "effort",
        "saturation_%",
        "max_cart_m",
        "max_link_deg",
    )
    rows = [header]

    for record in records:
        metrics = record.metrics
        settling = "not settled" if metrics.settling_time is None else f"{metrics.settling_time:.3f}"
        links = " ".join(f"{np.degrees(value):.2f}" for value in metrics.max_link)
        rows.append(
            (
                record.case,
                record.controller,
                settling,
                f"{metrics.peak_applied_force:.3f}",
                f"{metrics.control_effort:.3f}",
                f"{100.0 * metrics.saturation:.2f}",
                f"{metrics.max_cart:.4f}",
                links,
            )
        )

    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    lines = []

    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row)))

        if index == 0:
            lines.append("  ".join("-" * width for width in widths))

    return "\n".join(lines)
