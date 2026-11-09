# kgrel — Reliability-Aware Dynamic Ontology Evolution for LLM-Based Knowledge-Graph Construction

> *Can we predict, quantify, explain, and control the reliability of LLM-generated knowledge **before** it enters a KG, including when the ontology itself must evolve?*

**kgrel** is the research code behind an M.S. thesis on reliability for open-world knowledge-graph construction with large language models. It is a lightweight, reproducible pipeline that:

1. loads the [Text2KGBench](https://github.com/cenguix/Text2KGBench) Wikidata-TekGen benchmark,
2. prompts an LLM to extract (subject, relation, object) triples from a sentence,
3. computes three per-triple **risk signals** — *source support*, *paraphrase instability*, *ontology novelty* —
4. runs a transparent **gate** that outputs one of three decisions: `insert` / `abstain` / `propose_schema_update`,
5. scores the result against gold under both strict and canonicalized matching, broken down by an explicit five-type **error taxonomy** (T1 unsupported hallucination, T2 relational distortion, T3 entity error, T4 missed fact, T5 format/serialization error).

The frozen stratified sample (**N = 300 sentences across 10 Wikidata-TekGen ontologies**, 20 % open-world hold-out, 50-sentence paraphrase sub-study) is checked in, so the headline experiments are **bit-exact reproducible without any benchmark download**.

All three evaluated models — `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-20b` — run on [Groq's free tier](https://console.groq.com/keys): **one API key, no credit card**. (The model line-up evolved during the project — earlier versions used Llama-3.3-70B / Llama-3.1-8B / Gemma2-9B, which were decommissioned from Groq's free tier in 2026.)

---

## Quick start

```bash
# 1) Clone and install
git clone https://github.com/AmnaMubarak/kgrel.git
cd kgrel
pip install -e .                 # or: pip install -r requirements.txt

# 2) Add your free Groq key
cp .env.example .env
#   -> edit .env and paste the key from https://console.groq.com/keys

# 3) Smoke test (no API call)
python scripts/99_pilot_smoke.py                   # expect: [smoke] OK
pytest -q                                          # same, via pytest

# 4) Verify the three models are reachable
python scripts/00_check_env.py

# 5) 40-sentence pilot on the two small models (quick sanity run)
python scripts/10_pilot.py

# 6) Full N=300 run on all three models
python scripts/03_extract.py --run-id core --models all
python scripts/04_score.py   --run-id core

# 7) (optional) After editing gate thresholds, the ontology layer, or a
#     signal implementation, rescore CACHED predictions without burning
#     any API quota:
python scripts/13_rescore.py --run-id pilot
```

The frozen sample is already at `data/sample/sample.jsonl`. You only need the raw benchmark if you want to **re-generate** the sample or run on sentences outside the frozen set:

```bash
python scripts/01_fetch_data.py    # ~1.4 GB download, extracts tekgen.zip
python scripts/02_sample.py        # regenerates data/sample/sample.jsonl
```

---

## Repository layout

```
kgrel/
├── src/kgrel/            Python package (import-light, modules do lazy imports)
│   ├── schema.py            Triple / Ontology / GoldItem dataclasses
│   ├── dataset.py           Text2KGBench Wikidata-TekGen loader (tolerant)
│   ├── sample.py            Stratified N=300 sampler + open-world hold-out
│   ├── prompt.py            Prompt templates + JSON schema
│   ├── parse.py             Robust JSON parser (tolerates fenced code, trailing text)
│   ├── extract.py           Backends: Groq, OpenAI-compatible, Mock (dry-run)
│   ├── canonical.py         Entity / relation normalization + rapidfuzz matching
│   ├── match.py             Strict + canonicalized gold matching
│   ├── paraphrase.py        Paraphrase generator + meaning-preservation filter
│   ├── signals.py           source_support, paraphrase_instability, ontology_novelty
│   ├── gate.py              insert / abstain / propose_schema_update
│   └── metrics.py           P/R/F1 + bootstrap CIs, Cohen κ, AUROC / P@k
├── scripts/              Numbered pipeline (run in order)
│   ├── 00_check_env.py      Pings each model, validates GROQ_API_KEY
│   ├── 01_fetch_data.py     Clones Text2KGBench + unpacks tekgen.zip
│   ├── 02_sample.py         Builds the frozen N=300 sample from the benchmark
│   ├── 03_extract.py        Runs the three LLMs on the sample
│   ├── 04_score.py          Per-model P/R/F1, error-taxonomy breakdown
│   ├── 10_pilot.py          40-sentence × 2-model pilot (parser recovery, coverage)
│   ├── 11_overlap_csv.py    Writes per-annotator CSVs for inter-annotator agreement
│   ├── 12_kappa.py          Computes Cohen κ (gate ≥ 0.60 to freeze taxonomy)
│   └── 99_pilot_smoke.py    End-to-end test on synthetic fixtures (no API key)
├── configs/config.yaml    Frozen research config (seed, model list, thresholds)
├── data/sample/           Checked-in frozen sample.jsonl + paraphrase_subset.jsonl
├── tests/test_smoke.py    pytest wrapper around the smoke script
└── pyproject.toml         MIT license, Python >= 3.8, package metadata
```

---

## The three risk signals (in one sentence each)

| Signal | What it measures | Range |
|---|---|---|
| `source_support` | How much of the triple's content is lexically grounded in the source sentence (token-overlap, canonicalized) | `[0, 1]` |
| `paraphrase_instability` | Fraction of paraphrased re-prompts that fail to return the same triple | `[0, 1]` |
| `ontology_novelty` | `in_schema` / `synonym` (fuzzy ≥ 92) / `new_type` / `junk` — classification of the relation vs. the ontology's declared relations | categorical |

The gate (`src/kgrel/gate.py`) is a short, auditable rule block — not a learned model — so every decision is explainable:

```
abstain                  if format_ok is False or novelty == "junk"
abstain                  if source_support < support_min
                         or paraphrase_stability < stability_min
insert                   if novelty in {"in_schema", "synonym"}
propose_schema_update    if novelty == "new_type"
                         and source_support >= propose_schema_min_support
```

Thresholds live in `configs/config.yaml` and are set on the Phase-2 pilot, not peeked at from the N=300 test set.

---

## Error taxonomy

Every scored triple is tagged with exactly one of:

- **T1 — unsupported_hallucination** — content not entailed by the source sentence
- **T2 — relational_distortion_or_ontology_violation** — predicate doesn't match the sentence's relation, or violates the ontology's relation list
- **T3 — entity_error** — wrong or malformed subject/object span
- **T4 — missed_fact** — gold triple the model did not produce
- **T5 — format_or_serialization_error** — un-parseable JSON, wrong keys, etc.

Content labels T1–T3 are annotated by two humans on a 30-triples-per-model overlap sample; Cohen κ ≥ 0.60 is the gate to freeze the taxonomy before the full N=300 run.

---

## Reproducing the headline numbers

```bash
pip install -e .
cp .env.example .env            # add GROQ_API_KEY
python scripts/00_check_env.py
python scripts/03_extract.py --run-id paper --models all --sample data/sample/sample.jsonl
python scripts/04_score.py   --run-id paper
```

Results land in `data/results/paper/`:

- `metrics_per_model.csv` — micro / macro P, R, F1 (strict + canonicalized) with 95 % bootstrap CIs
- `taxonomy_per_model.csv` — T1–T5 rates per model
- `gate_outcomes.csv` — `insert / abstain / propose_schema_update` counts
- `openworld.csv` — accuracy on the 20 % held-out-relation subset
- `paraphrase.csv` — instability-vs-correctness correlations on the 50-sentence sub-study

---

## Citation

If you use kgrel, please cite both the toolkit and the thesis:

```bibtex
@misc{mubarak2026kgrel,
  author = {Mubarak, Amna},
  title  = {kgrel: Reliability-Aware Dynamic Ontology Evolution for LLM-Based Knowledge-Graph Construction},
  year   = {2026},
  howpublished = {\url{https://github.com/AmnaMubarak/kgrel}},
  note   = {Version 0.1.0}
}

@mastersthesis{mubarak2027thesis,
  author = {Mubarak, Amna},
  title  = {Reliability-Aware Dynamic Ontology Evolution for Open-World LLM Knowledge Graph Construction},
  school = {FAST-NUCES Karachi},
  year   = {2027},
  type   = {M.S. Data Science Thesis}
}
```

A machine-readable `CITATION.cff` is included.

---

## License

MIT — see [`LICENSE`](LICENSE).

The included Text2KGBench frozen sample is derived from [cenguix/Text2KGBench](https://github.com/cenguix/Text2KGBench), which is released under its own license; please respect it if you re-distribute.

---

## Acknowledgments

Thesis supervised by **Dr. Zulfiqar Memon** (FAST-NUCES Karachi). Builds on top of [Text2KGBench](https://arxiv.org/abs/2308.02357) (Mihindukulasooriya et al., ISWC 2023) and the risk-signal / error-taxonomy lines of work in [GenRES](https://arxiv.org/abs/2402.10744), [KGHaluBench](https://arxiv.org/abs/2502.13593), [GraphJudge](https://arxiv.org/abs/2411.17388), and [GraphRefine](https://arxiv.org/abs/2509.16531).
