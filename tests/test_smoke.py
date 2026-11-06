"""Smoke test that imports every kgrel module and runs the pilot end-to-end
on synthetic fixtures (no API key needed).

Wraps scripts/99_pilot_smoke.py so `pytest -q` catches regressions in CI.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_imports() -> None:
    import kgrel  # noqa: F401
    from kgrel import canonical, dataset, extract, gate, match, metrics  # noqa: F401
    from kgrel import paraphrase, parse, prompt, sample, schema, signals  # noqa: F401


def test_pilot_smoke_runs() -> None:
    """Runs scripts/99_pilot_smoke.py as a subprocess; last line must be [smoke] OK."""
    script = REPO / "scripts" / "99_pilot_smoke.py"
    assert script.exists(), f"missing {script}"
    env = {
        "PYTHONPATH": str(REPO / "src"),
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    out = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, f"smoke exited {result.returncode}:\n{out}"
    last = [l for l in out.splitlines() if l.strip()][-1]
    assert last.strip() == "[smoke] OK", f"unexpected last line: {last}"
