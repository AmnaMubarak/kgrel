# Pilot annotation guidelines (v1 — November 2026)

**Purpose:** two annotators independently label 60 triples (30 from each model). Cohen's κ on the overlap must reach **≥ 0.60** before we scale to the full N=300 run. Target reading time: **15 minutes**. Target annotation time: **~90 minutes per annotator**.

**Context:** the task is LLM-based knowledge-graph construction. For each triple `(subject, relation, object)` the model produced from a source sentence, you decide **one** of five labels: `correct`, `T1`, `T2`, `T3`, or `T5`.

> **Do NOT read each other's file** until you are both done. Looking is the one thing that destroys the κ measurement.

---

## 1. Workflow

1. Open your assigned CSV in Excel / Google Sheets / LibreOffice Calc:
   - Annotator A → `data/annotation/pilot_annotator_A_amna.csv`
   - Annotator B → `data/annotation/pilot_annotator_B_labmate.csv`
2. Both files have the **same 60 triples in the same order**. Different `human_label` columns (empty), same everything else.
3. For every row, fill in **`human_label`** (one of the 5 allowed values below). Leave the other three annotation columns blank unless noted.
4. Save as **CSV UTF-8** (not `.xlsx`). The κ script reads CSV only.
5. Message each other when done. Then run `python scripts/12_kappa.py --run-id pilot`.

---

## 2. What each column in the CSV means

| Column | Meaning |
|---|---|
| `model` | Which LLM produced this triple (`qwen3-27b` or `gpt-oss-20b`). |
| `sent_id`, `ontology` | Benchmark-internal identifiers; informational only. |
| `openworld_held_out` | `True` if the gold relation was hidden from the model's prompt for this sentence. Harder case. |
| `sentence` | The source text the model was asked to extract from. **Your gold reference.** |
| `subject`, `relation`, `object` | The triple the model emitted — **the thing you are judging**. |
| `matched` | Did the triple canonical-match an official gold triple? `False` doesn't mean wrong (see §4). |
| `support`, `novelty`, `format_ok`, `gate_action` | Automatic signals, for your reference only. Don't let them bias your judgment. |

---

## 3. The five allowed `human_label` values

Pick **exactly one**. Decision ladder — walk down and stop at the first match:

### 🚫 `T5` — Format / serialization error
Use this when the triple is **structurally broken**: missing subject/object/relation, non-string value, completely empty field, obvious JSON corruption. These are form, not content, failures.

> *Pilot shipped with `format_ok=True` for every row (the parser already filtered broken JSON into `n_format_fail`), so T5 should be rare in your overlap. Use it only if you see something like `subject=""` or `relation=null` slipping through.*

### ✅ `correct`
Use this when the triple is a **faithful** representation of a fact in the source sentence, judged as if explaining it to a reasonable person. Must satisfy **all three**:

- **Subject** names the right entity (minor variants of spelling / casing are fine).
- **Relation** means the same thing as what the sentence asserts (natural synonyms OK, e.g. `author` ≈ `wrote`).
- **Object** names the right value.

> It does **not** matter whether `matched=True` in the CSV. The strict canonical matcher misses many correct paraphrases.

**Example (label this as `correct`):**
- Sentence: `"Václav Klaus was a Czech economist and politician who served as the second President of the Czech Republic."`
- Triple: `(Václav Klaus, head of state, Czech Republic)`
- → **correct** (the sentence says he was President of that country — "head of state" is a fair relation).

### ❌ `T1` — Unsupported hallucination
The triple's content is **not entailed** by the source sentence. A reasonable reader could not conclude this triple from the text alone — the model invented it (or imported outside knowledge not grounded in this sentence).

**Example:**
- Sentence: `"Elon Musk founded SpaceX in 2002."`
- Triple: `(Elon Musk, author of, Dune)`
- → **T1** (nothing in the sentence supports it).

**Tricky cases:**
- World-knowledge inference *beyond* the sentence = **T1** (we only score what the text says).
- Information from parentheticals that the LLM incorrectly expanded = **T1**.

### 🔀 `T2` — Relational distortion / ontology violation
The **subject** and **object** are correctly from the sentence, but the **relation** is wrong — either it doesn't match the sentence's assertion, or it violates how the ontology uses that relation.

**Example:**
- Sentence: `"Doom 3: Resurrection of Evil was developed by Nerve Software and published by Activision."`
- Triple: `(Doom 3: Resurrection of Evil, developer, Activision)`
- → **T2** (Activision is the **publisher**, not the developer; relation is swapped).

**Tricky cases:**
- If the relation name in the triple is a reasonable natural-language synonym of the intended one (e.g. `made` for `created`), label as **`correct`**, not T2. Only flag T2 when the semantic meaning diverges.
- If the triple uses a relation NOT in the ontology at all but fits the sentence well → still label `correct` (we let the `ontology_novelty` signal flag those separately). Only use T2 if the relation is semantically wrong.

### 👤 `T3` — Entity error
The **relation** is reasonable, but a subject or object is **wrong, misspelled, mis-scoped, or misreferenced**.

**Example:**
- Sentence: `"Taylor Park Dam is located on the Taylor River, a tributary of the Gunnison River."`
- Triple: `(Taylor River, tributary, Gunnison River)`
- → **correct** (this one is fine actually — "Taylor River is a tributary of the Gunnison").

- Sentence: `"Barack Obama met Angela Merkel in Berlin in 2015."`
- Triple: `(Barack Obama, met, Angela)`
- → **T3** (object `Angela` is truncated / mis-scoped — her name is `Angela Merkel`).

**Tricky cases:**
- Pronouns substituted for referents the sentence makes clear → T3.
- Spelling variants that still unambiguously refer to the same entity → `correct` (not T3).
- Wrong entity entirely (confused one person with another) → T3.

---

## 4. The two "assist" columns (optional)

Fill these in only when relevant — they are *not* part of the κ calculation. They help us later populate the `propose_schema_update` list.

- **`new_type_candidate` = `1`** when the model used a relation that isn't in the ontology but looks like a *reasonable candidate for a new ontology relation*. Example: a relation like `"competition class"` that isn't in the schema but makes sense for the sport ontology.
- **`synonym_of`** = *name of the ontology relation it should be merged with*. Example: if the model emits `wrote` and the ontology has `author`, fill `synonym_of = "author"`.

Leave both blank for anything you label `correct` or `T5`.

---

## 5. Decision-ladder cheat-sheet (pin this)

For each row, ask in order:

```
1. Is the triple structurally broken?                → T5
2. Does the sentence NOT support this fact at all?   → T1
3. Is the subject or object wrong / mis-scoped?      → T3
4. Is the relation wrong (not a reasonable synonym)? → T2
5. Otherwise                                         → correct
```

Stop at the first `yes`. Don't double-label.

---

## 6. Hard cases — how to resolve, honestly

| Situation | Label |
|---|---|
| The relation in the triple is a *reasonable natural-language synonym* of what the sentence asserts | `correct` |
| The triple is technically true in the real world but **not** in this sentence | `T1` |
| Two triples from the same sentence both look correct, but only one matches gold | both `correct` |
| Open-world sentence where the model emits a *new* relation that fits the sentence | `correct` (and set `new_type_candidate=1` if the relation is a plausible ontology addition) |
| You genuinely cannot decide between `correct` and T2 | lean `T2` and note why in `human_comment` |
| You genuinely cannot decide between T1 and T3 | lean `T1` (hallucination is the more conservative label) |
| Triple contains unicode / encoding mojibake but is otherwise correct | `correct`, note in `human_comment` |

**Golden rule:** *we score what the sentence says, not what the world knows.*

---

## 7. What happens next

1. You both finish labeling → ping each other.
2. Save both files as `pilot_annotator_A_amna.csv` and `pilot_annotator_B_labmate.csv` in `data/annotation/`.
3. Run `python scripts/12_kappa.py --run-id pilot`.
4. If **κ ≥ 0.60** overall → taxonomy is frozen, we proceed to the full N=300 run.
5. If **κ < 0.60** → we review the disagreements together, refine this guideline (v2), re-annotate just the disagreement cases, re-check κ.

κ of **0.60-0.80** is "substantial agreement" (Landis & Koch 1977); 0.80+ is "almost perfect". We're targeting at least substantial. This is normal for a first-pass pilot.

---

**Questions during labeling** → leave a quick note in the `human_comment` column. Don't ping the other annotator — anything that looks ambiguous to you probably looks ambiguous to them, and we WANT that signal.
