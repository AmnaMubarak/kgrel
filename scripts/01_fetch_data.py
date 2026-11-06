"""Download the Text2KGBench benchmark into data/raw/text2kgbench/.

Preferred source: the official GitHub repo (cenguix/Text2KGBench), shallow-cloned.
Falls back to a direct zip download if `git` is not on PATH.

The repo ships the Wikidata-TekGen sentences inside `tekgen.zip` (LFS), so we
also unpack that zip into the right place so the config globs match:

    data/raw/text2kgbench/data/wikidata_tekgen/ontologies/*_ontology.json
    data/raw/text2kgbench/data/wikidata_tekgen/test/*_test.jsonl
    data/raw/text2kgbench/data/wikidata_tekgen/ground_truth/*_ground_truth.jsonl

Usage:
    python scripts/01_fetch_data.py

Note: if you only want to reproduce our N=300 experiments, you do NOT need
to run this script. The frozen sample is already checked in at
`data/sample/sample.jsonl`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
RAW_DIR = REPO / "data" / "raw" / "text2kgbench"

REPO_URL = "https://github.com/cenguix/Text2KGBench.git"
ZIP_URL = "https://codeload.github.com/cenguix/Text2KGBench/zip/refs/heads/main"


def _git_clone() -> bool:
    if not shutil.which("git"):
        return False
    print(f"[fetch] git clone --depth=1 {REPO_URL}")
    try:
        subprocess.check_call(
            ["git", "clone", "--depth=1", REPO_URL, str(RAW_DIR)]
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[fetch] git clone failed: {e}")
        return False


def _zip_download() -> bool:
    print(f"[fetch] downloading {ZIP_URL}")
    tmp_zip = RAW_DIR.parent / "_text2kgbench_main.zip"
    urlretrieve(ZIP_URL, tmp_zip)
    print("[fetch] extracting")
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(RAW_DIR.parent)
    extracted = RAW_DIR.parent / "Text2KGBench-main"
    if extracted.exists():
        extracted.rename(RAW_DIR)
    tmp_zip.unlink(missing_ok=True)
    return True


def _unpack_tekgen_zip() -> None:
    """Expand the bundled tekgen.zip so wikidata_tekgen/ gets its files."""
    tekgen_zip = RAW_DIR / "tekgen.zip"
    probe = RAW_DIR / "data" / "wikidata_tekgen" / "ontologies"
    probe_files = list(probe.glob("*_ontology.json")) if probe.exists() else []
    if probe_files:
        print(f"[fetch] wikidata_tekgen already populated "
              f"({len(probe_files)} ontology files)")
        return
    if not tekgen_zip.exists():
        print("[fetch] WARNING: no tekgen.zip found; wikidata_tekgen/ will be empty.")
        print("[fetch]          Download manually from the Text2KGBench LFS release.")
        return
    size_mb = tekgen_zip.stat().st_size / (1024 * 1024)
    print(f"[fetch] extracting tekgen.zip ({size_mb:.0f} MB)...")
    with zipfile.ZipFile(tekgen_zip, "r") as zf:
        zf.extractall(RAW_DIR / "data")
    print("[fetch] tekgen.zip extracted")


def main() -> int:
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not (RAW_DIR.exists() and any(RAW_DIR.iterdir())):
        if not _git_clone():
            _zip_download()
    else:
        print(f"[fetch] repo already present: {RAW_DIR}")

    _unpack_tekgen_zip()
    print(f"[fetch] done -> {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
