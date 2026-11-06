"""Environment check — run this first.

Verifies:
  1. GROQ_API_KEY is set
  2. The `groq` package is installed
  3. Each configured model responds to one tiny test prompt
  4. The response parses as triples JSON

Prints a one-line pass/fail per model. Exit code 0 only if all models pass.

Usage:
    python thesis/implementation/scripts/00_check_env.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve()
IMPL = HERE.parents[1]
sys.path.insert(0, str(IMPL / "src"))

# Load .env so GROQ_API_KEY is available without manual export.
try:
    from dotenv import load_dotenv
    load_dotenv(IMPL / ".env")
except ImportError:
    pass  # dotenv is optional; env vars can still be set manually

from kgrel.extract import ModelSpec, make_client   # noqa: E402
from kgrel.parse import parse_triples              # noqa: E402

TEST_SENTENCE = (
    "Peter and the Piskies: Cornish Folk and Fairy Tales is a 1958 anthology "
    "illustrated by Raymond Briggs."
)
TEST_MESSAGES = [
    {"role": "system",
     "content": ("Extract knowledge-graph triples from the passage. "
                 "Return ONLY a JSON object with key `triples`, "
                 "a list of {subject, relation, object}.")},
    {"role": "user",
     "content": f"Passage:\n{TEST_SENTENCE}\n\nReturn JSON."},
]


def check_env() -> bool:
    ok = True
    if not os.environ.get("GROQ_API_KEY"):
        print("[env] FAIL  GROQ_API_KEY not set")
        print("      Get one at https://console.groq.com/keys (free, no card)")
        print("      Then: $env:GROQ_API_KEY = 'gsk_...'")
        ok = False
    else:
        key = os.environ["GROQ_API_KEY"]
        masked = key[:6] + "..." + key[-4:] if len(key) > 12 else "set"
        print(f"[env] OK    GROQ_API_KEY = {masked}")

    try:
        import groq  # noqa: F401
        print(f"[env] OK    groq package installed (v{groq.__version__})")
    except ImportError:
        print("[env] FAIL  groq package not installed.  Run:  pip install groq")
        ok = False
    return ok


def check_model(spec: ModelSpec) -> bool:
    try:
        client = make_client(spec)
    except Exception as e:
        print(f"[{spec.name:>18}]  FAIL  client init: {e}")
        return False

    try:
        raw = client.call(TEST_MESSAGES)
    except Exception as e:
        print(f"[{spec.name:>18}]  FAIL  API call: {e}")
        return False

    triples, parse_ok = parse_triples(raw)
    if not parse_ok:
        print(f"[{spec.name:>18}]  WARN  response did not parse as JSON")
        print(f"                      raw (first 180 chars): {raw[:180]!r}")
        return False

    print(f"[{spec.name:>18}]  OK    {len(triples)} triples parsed; "
          f"first = {triples[0].as_tuple() if triples else '()'}")
    return True


def main() -> int:
    print("=== environment ===")
    env_ok = check_env()
    if not env_ok:
        print("\n[env] fix the above before running any model calls.")
        return 2

    cfg = yaml.safe_load((IMPL / "configs" / "config.yaml").read_text(encoding="utf-8"))
    print("\n=== model smoke tests ===")
    all_ok = True
    for mc in cfg["models"]:
        spec = ModelSpec(
            name=mc["name"], backend=mc["backend"],
            api_model_id=mc.get("api_model_id"),
            temperature=mc.get("temperature", 0.0),
            top_p=mc.get("top_p", 1.0),
            max_tokens=mc.get("max_tokens", 512),
        )
        all_ok &= check_model(spec)

    print()
    if all_ok:
        print("[env] ALL PASSED — you are ready to run scripts/10_pilot.py")
        return 0
    else:
        print("[env] one or more models failed; see messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
