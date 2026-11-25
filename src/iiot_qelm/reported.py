"""Loader for the transcribed published values (references/reported/*.json). Used by the audit and the report builder ONLY for
comparison; no training or inference code imports it."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from .errors import MissingInputError


def references_dir() -> Path:
    cands = []
    if os.environ.get("IIOT_QELM_REFERENCES"):
        cands.append(Path(os.environ["IIOT_QELM_REFERENCES"]))
    cands += [Path(__file__).resolve().parents[2] / "references" / "reported", Path.cwd() / "references" / "reported"]
    for c in cands:
        if (c / "table1.json").is_file():
            return c
    raise MissingInputError("references/reported/*.json", "run from the repository root or set IIOT_QELM_REFERENCES=<dir> (regenerate with `python tools/build_reported.py`)")


def load(name: str) -> dict:
    return json.loads((references_dir() / f"{name}.json").read_text(encoding="utf-8"))


def rows(name: str) -> List[dict]:
    t = load(name)
    cols = t["columns"]
    return [dict(zip(cols, r)) for r in t["rows"]]
