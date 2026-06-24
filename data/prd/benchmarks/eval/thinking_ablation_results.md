# Thinking-ablation bench — manual grade (#16, R02b)

**Axis under test:** `THINKING_POLICY` (the real lever — see discovery #40; `THINKING_LEVEL` is a
no-op on Qwen3). `on` = thinking left ON (policy `off`, no soft-switch); `gate` = thinking gated
OFF every turn (policy `gate_all`, appends `/no_think`).

**Model:** Qwen3-0.6B-Q8_0 (llama-server :8090, n_ctx=4096). **Corpus:** `data/default-dataroom.zip`
(Meridian Robotics, 10 docs). **Budget:** 1 turn. **Reps:** 3 per (query × config).
**Harness:** `scratchpad/think_bench.sh` → `runs/think_bench/results.jsonl`.

Grading is MANUAL (doctrine §0/§4 — the answer string is the SoT, read against the frozen gold).

## Gold answers (from queryset.jsonl)

| qid | query | gold |
|---|---|---|
| q01_factoid  | What is the battery life of the Atlas-7?            | 8 hours |
| q11_literal  | What error code did incident INC-2041 report?      | ERR_TORQUE_OVERFLOW |
| q20_dispersed| What is the total capital Meridian has raised?     | 75.5 million US dollars |
| q26_multihop | Who led the team that shipped Atlas-7 v3.0?        | Priya Nair |

## Grade (✓ = answer matches gold; manually read each ANSWER.md)

| query | thinking-ON | gated-OFF |
|---|---|---|
| q01 factoid   | r1 ✓ · r2 ✗ "not available" · r3 ✓ = **2/3** | r1 ✗ · r2 ✗ · r3 ✗ (all "not provided") = **0/3** |
| q11 literal   | r1 ✓ · r2 ✓ · r3 ✗ (done=false, empty) = **2/3** | r1 ✗ · r2 ✗ ("403 Unauthorized" halluc) · r3 ✗ = **0/3** |
| q20 dispersed | r1 ✓ · r2 ✗ ("$425M" halluc) · r3 ✓ = **2/3** | r1 ✗ ("$210M") · r2 ✗ · r3 ✗ = **0/3** |
| q26 multihop  | r1 ✗ ("Locomotion") · r2 ✓ · r3 ✗ = **1/3** | r1 ✗ ("Airborne") · r2 ✓ · r3 ✗ ("US Air Force") = **1/3** |
| **TOTAL**     | **7/12 = 58%** | **1/12 = 8%** |

## Speed (llm_ms, from timing.json — the SoT)

| config | output tokens (range) | llm_ms (median) | elapsed (median) |
|---|---|---|---|
| thinking-ON | 366–834 | ~20,000 | ~28 s |
| gated-OFF   | 21–40 (q26r2 outlier 345) | ~4,000  | ~5 s |

Gating is ~5× faster — and the token-count gap (ON 360–834 vs GATE 21–40) is physical proof the
`/no_think` switch actually fired (not a silent no-op).

## Gate decision (the #16 deliverable)

The #16 gate: **a faster config ships only if its grade ≥ the thinking-ON baseline.**
Gated grade **8% < ON baseline 58% → GATE FAILS.** Default `THINKING_POLICY` stays `off`
(thinking ON). The #17 default-flip is correctly **BLOCKED** (fail-closed, doctrine §1.9).

## Root cause (first principles — why gating collapses grounding)

On a 0.6B model the chain-of-thought IS the tool-use driver. The thinking tokens are where the
model decides to call `search_dataroom`, reads the retrieved chunk, and grounds the answer. Gate
that off and it skips the search and answers from priors → it either refuses ("not provided") or
hallucinates a confident wrong value ("$210M", "403 Unauthorized", "United States Air Force"). The
mechanism works exactly as designed — this is a model-capacity result, not a harness bug.

**Corollary:** swap in a larger model that can drive tool use without visible CoT and re-run this
bench; if gated grade then meets the baseline, the #17 flip auto-qualifies through the same gate.

**Secondary observation:** GATE q26 r2 spent 17s / 345 tokens and answered correctly — Qwen3 does
not obey `/no_think` 100% of the time; occasionally it thinks anyway. Aggregate signal is still
unambiguous.
