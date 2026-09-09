"""File tools: search, open, and (confirmed) delete.

Searching is capped (time + result count) so it stays light on a weak PC.
Destructive operations are registered with safety="confirm" so the router
never runs them without explicit user confirmation.
"""

from __future__ import annotations

import os
import time
import shutil

from tools.registry import tool

_MAX_WALK_SECONDS = 8.0
_MAX_WALK_RESULTS = 15


@tool(
    name="file_search",
    description="Search a folder (default: the user's Documents/Desktop) for files whose name contains a query.",
    params={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "filename fragment to find"},
            "folder": {"type": "string", "description": "folder to search (optional)"},
        },
        "required": ["query"],
    },
)
def file_search(query: str, folder: str | None = None) -> str:
    roots = _default_roots() if not folder else [folder]
    query = query.lower()
    matches: list[str] = []
    start = time.time()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip heavy/irrelevant dirs.
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith((".", "$")) and d not in ("node_modules", "venv", ".venv")
            ]
            for name in filenames:
                if query in name.lower():
                    matches.append(os.path.join(dirpath, name))
                    if len(matches) >= _MAX_WALK_RESULTS:
                        break
            if len(matches) >= _MAX_WALK_RESULTS or time.time() - start > _MAX_WALK_SECONDS:
                break
        if len(matches) >= _MAX_WALK_RESULTS:
            break
    if not matches:
        return "no files found"
    return "; ".join(matches[:_MAX_WALK_RESULTS])


def _default_roots() -> list[str]:
    home = os.path.expanduser("~")
    return [
        os.path.join(home, "Documents"),
        os.path.join(home, "Desktop"),
        os.path.join(home, "Downloads"),
    ]


@tool(
    name="file_open",
    description="Open a file with its default application.",
    params={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "absolute file path"}},
        "required": ["path"],
    },
)
def file_open(path: str) -> str:
    if not os.path.isfile(path):
        raise ValueError(f"file not found: {path}")
    os.startfile(path)  # type: ignore[attr-defined]
    return f"opened {os.path.basename(path)}"


@tool(
    name="delete_path",
    description="Permanently delete a file or folder. Requires confirmation before it runs.",
    params={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "absolute path to delete"}},
        "required": ["path"],
    },
    safety="confirm",
)
def delete_path(path: str) -> str:
    if not os.path.exists(path):
        raise ValueError(f"path not found: {path}")
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
    return f"deleted {path}"
