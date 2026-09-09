"""System tools: time and date."""

from __future__ import annotations

import datetime

from tools.registry import tool


@tool(
    name="get_time",
    description="Get the current local time.",
    params={"type": "object", "properties": {}},
)
def get_time() -> str:
    return datetime.datetime.now().strftime("%I:%M %p").lstrip("0")


@tool(
    name="get_date",
    description="Get today's date.",
    params={"type": "object", "properties": {}},
)
def get_date() -> str:
    return datetime.date.today().strftime("%A, %B %d, %Y")
