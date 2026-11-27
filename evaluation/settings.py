"""Deployment answers: an environment variable, else the user's answer in the local preferences file.

`docs/local-preferences.md` is the tracked template. `python -m evaluation.setup --init` copies it to
`docs/local-preferences.local.md`, which git ignores and which holds this machine's answers. Code reads the local
file when it exists, else the template, whose answers are all unanswered.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "docs" / "local-preferences.md"
LOCAL = REPO_ROOT / "docs" / "local-preferences.local.md"


def preferences_file() -> Path:
    return LOCAL if LOCAL.is_file() else TEMPLATE


def answer(section: int, item: str | None = None, text: str | None = None) -> str | None:
    """The answer to section `section` (`## <n>.`), or to its list item starting with `item`; None when unanswered.
    A value written in backticks is returned without them."""
    text = preferences_file().read_text() if text is None else text
    match = re.search(rf"^## {section}\. .*?(?=^## \d+\. |\Z)", text, re.M | re.S)
    if match is None:
        return None
    body = match.group(0)
    if item is not None:
        found = re.search(rf"^- {re.escape(item)}.*?(?=^- |^#|\Z)", body, re.M | re.S)
        body = found.group(0) if found else ""
    found = re.search(r"Answer:\s*(.+?)\s*$", body, re.M)
    if found is None or found.group(1).startswith("unanswered"):
        return None
    value = found.group(1)
    return value[1:-1] if len(value) > 1 and value[0] == value[-1] == "`" else value


def setting(variable: str, section: int, item: str | None = None) -> str | None:
    return os.environ.get(variable) or answer(section, item)


def path_setting(variable: str, section: int) -> Path | None:
    """A path answer; a relative one is relative to this repository's root, as the preferences file records it."""
    value = setting(variable, section)
    if not value:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() or os.environ.get(variable) else REPO_ROOT / path).resolve()


def unanswered(text: str | None = None) -> list[str]:
    """Each unanswered item, as `<section heading>: <item>`; a list item's answer may sit on a continuation line."""
    text = preferences_file().read_text() if text is None else text
    found, section, heading, item = [], "", "", ""
    for line in text.splitlines():
        if line.startswith("## "):
            section = heading = line[3:].strip()
            item = ""
            continue
        if line.startswith("### "):
            heading, item = f"{section} / {line[4:].strip()}", ""
            continue
        if line.startswith("- "):
            item = re.split(r"[:(.]|, for example", line[2:], maxsplit=1)[0].strip()
        elif not line.startswith("  "):
            item = ""
        if re.search(r"(?<!`)(Answer|^- [^:]+):\s*unanswered", line) and heading[:1].isdigit():
            found.append(f"{heading}: {item}" if item else heading)
    return found
