# Zalo AI Challenge 2026 — DataMate: Competition Context

## Document provenance and verification status

- **User-requested rollback, November 8, 2026.** After reporting slow new inference, the user requested the previous system and explicitly asked to update the 200-question Colab workflow after rollback. Restored all inference Python source byte-for-byte from the archived pre-v7 pipeline-6 release (SHA256 38407bd0bbdf97489f3de59d11c88c1f604a061f81b30914e8bba43ddbfbed64). Both 100-question source/notebook and 200-question sealed kit/notebook now target pipeline 6/backend 3 under fresh rollback run names. Diagnostic 100-question version is 8. Checkpoints, benchmark question/reference bytes, scoring, logs and frozen Gemma SFT artifacts are preserved; v7 code/notebooks/kit are archived. The team runner retains pipeline identity/resume bookkeeping but accepts pipeline 6; it does not change model decisions. All **225 software tests pass**. No new full-checkpoint GPU speed/PA measurement was performed. See [rollback scope, backups and reload steps](rollback-pipeline6-2026-11-08.md). This changes the active software release, not competition rules.

- **Team200 kit synchronized with interface v7, November 8, 2026.** At the user's request, the latest completed turn in the Codex chat “Phân tích kết quả inference lần 2” was read and compared with current workspace files. That turn deliberately updated only the 100-question workflow, leaving the previous 200-question kit on pipeline 6. The dedicated 200-question source ZIP now includes every current inference source file, including the new typed-parameter module and all v7 interface/prompt/review fixes; its notebook pins the new source hash, requires pipeline >=7 and defaults to a fresh `team200-interface-v7` run name. Numeric/boolean load settings and self_review mode match the referenced notebook. Both frozen benchmark ZIPs, all 200 question/reference bytes, model checkpoints and output contracts remain unchanged. The old kit was preserved under `outputs/team200-colab/history/f709f78e1bf0/`. All 55 targeted tests passed; current source/file equality, independent pipeline-7 import/preflight, notebook compilation and nested archive hashes were verified. No checkpoint GPU inference or new model score/timing was performed. See [current kit and replacement instructions](team200-colab.md).

- **Agent interface v7 repair, November 8, 2026.** The user authorized software/interface fixes after the pipeline-6 analysis, reserving model semantic errors for later fine-tuning and planning new Qwen/Gemma runs on the same 100 questions. Implemented query-reference finalization with checked projection/type, stable native aliases, bounded one-column legacy projection repair, typed parameters, canonical entity-key checks before ledger zeros, verified redundant period assertions, ISO timestamp checks, exact price-entity normalization, actionable source feedback, evidence-backed unanswerable without redundant review, and explicit exhausted-review handling. Profile feedback is paginated without cutting rulebook/README or query result rows. All **241 software tests pass**. CPU replay covers all 31 recorded projection-error questions and recovers all 15 formerly lost correct candidates in fixed trajectories; both offline tokenizers fit the 100 initial/replayed conversations at 16K with 768 reserved output tokens. These are software/context checks, not new model PA/GPU timings. Source ZIP/notebook updated to pipeline 7/diagnostic 9, backend protocol 3, unchanged random-100 seed/order. No question/business-ID answer rules, training data regeneration, model fine-tuning or independent competition-rule verification. Frozen Gemma SFT and team200 releases remain unchanged. See [implementation and limits](interface-v7-2026-11-08.md).

- **Dedicated Colab runner for the frozen team benchmark.** The user requested a runnable Colab notebook for the same 200 questions, official-shaped submission and six-field metadata, automatic runtime release, and private review/logs like earlier inference runs. Added `DataMate_Colab_Team200.ipynb` plus a sealed source/input kit. The agent reads the label-free participant dataset; coordinator labels are opened only after the generation loop for review. Outputs split into a two-file group submission ZIP and a private diagnostic ZIP. Resume validates source/model/config/data/runtime, skips completed logs and accounts for separate load sessions; unknown timing after hard runtime loss remains null. Reports use all 200 questions as the PA denominator. Two verified ZIPs and a fresh invocation receipt precede Drive flush/unassign. All 37 targeted tests passed using synthetic agents/API stubs; package/source import and notebook-cell compilation were verified. **No full-checkpoint GPU/Colab inference, new model PA/timing or compute-unit measurement was performed.** Existing runtime agent, old 100-question notebook and frozen fine-tuning releases remain unchanged. This implements the user's benchmark workflow, not new competition rules. See [notebook, output contract and validation limits](team200-colab.md).

- **Team benchmark protocol v2, November 7, 2026.** At the user's request, participant instructions now support arbitrary agent/model/framework designs and require only official-shaped `submission.jsonl` plus six metadata fields: participant, models, GPU, initialization seconds, batch inference seconds and optional compute units. Question logs are optional for this internal benchmark; the competition Docker logging requirement in the supplied summary is unchanged. Examples cover all nine output types, with answer values for the seven value types and no answer required for abstain/unanswerable. The 200 question bytes, order, seed **4000930841**, business data and reference bytes are unchanged from v1. All 18 software tests passed; standalone release verification accepts six-field metadata without logs, computes batch mean only for a complete submission and leaves per-question percentiles unknown. Both v2 ZIPs passed CRC/per-file SHA256 verification. No model inference or new model score/timing was performed. An indexed official portal Description result read November 7 corroborates the nine type names and id/type/answer example; a direct portal open still returned no readable text, so full Evaluation/Rule/Submission Guidance remain independently unverified. See [current v2 package and exact limits](team-benchmark-train200-2026-11-07.md).

- **Team benchmark preparation, November 7, 2026.** The user requested a shareable 200-question random TRAIN benchmark. The original registered TRAIN fingerprint was verified, then a uniform sample without replacement was frozen with random seed **4000930841** before joining labels. The sample covers 28 businesses; no type/business balancing, forced cases, old-question exclusion or holdout filtering was used. Participants receive the complete relevant business data, TRAIN rulebook/calendar and label-free questions; a separate coordinator ZIP contains the original 200 references/aliases and a frozen local evaluator for centralized scoring and per-question comparisons. Both ZIPs passed CRC/per-file SHA256 checks; 17 software tests passed and synthetic scoring checks yielded expected boundary values. **No new model inference, model PA, GPU timing or compute-unit measurement was performed.** This is an internal TRAIN benchmark, not an independent holdout or official-rule verification. See [release, protocol and limitations](team-benchmark-train200-2026-11-07.md).

- **Qwen 9B pipeline-6 full run received November 7, 2026.** The user supplied `outputs/diagnosis-2026-11-07/qwen35-9b-random100-shared-quality-Nov7th-01/`. All 100 IDs/order, source fingerprint/runtime hashes, 311 raw hashes and 207 ZIP members were verified. Re-scoring yields 43 correct, 20 wrong, 37 abstentions, PA 0.23 versus prior Qwen's 0.17; correct answers fall from 49 while calls rise 194→311 and mean latency 9.64→10.68 seconds. Same checkpoint manifest/GPU/BF16; Torch and Transformers versions changed, so this is not a review-only ablation. EOS stopping is correct in all calls. Column projection fails in 31 questions, losing 15 correct candidates; after 84 calculated candidates the model only emits final actions, with zero post-candidate plan revisions. CPU probes isolate text/numeric prescription filtering and names bypassing customer-key validation; manually corrected plans/SQL match all 20 wrong references. Mechanical projection replay also releases six wrong candidates and is explicitly counterfactual, not a repaired-model PA. Remaining weaknesses include semantic scope/time/grain, API repair, anonymous-customer mapping, debt/lot operations and review at the last budgeted step. Analysis/docs only; runtime, source ZIP and frozen Gemma SFT releases remain unchanged. See [full report and evidence](inference-review-qwen9b-pipeline6-2026-11-07.md). These remain repeatedly inspected TRAIN development results, not independent holdout or official-rule verification.

- **Shared quality repair, November 7, 2026.** The user authorized implementation after the full 100-question Qwen analysis, for subsequent Gemma 12B and Qwen 9B inference. Pipeline 6 tightens plan fields/periods, handles evidence-backed serial dates/POS roles, resolves catalog dimensions, adds shared aggregate/debt/stock operations and fixes SQL CTE/view lineage. Colab and both presets use the same model for candidate review before final; that costs an extra call on the direct path and has no measured PA/speed benefit yet. Source/import preflight occurs before weight copying/loading. All 213 software tests pass. Final CPU replay matches 32/32 manually corrected cases and preserves 47/47 previously correct native plans; six additional previously abstained cases match with hand-supplied plans, while one missing anonymous-customer default remains guarded. All 100 initial and 47 review prompts fit both pinned tokenizers at 16K plus 768 reserved output tokens. These are tool/context checks, not new model scores. CPU/tokenizer evidence and limitations are documented in [the implementation report](shared-quality-fix-2026-11-07.md). No new full-checkpoint GPU inference, fine-tuning or independent competition-rule verification was performed; Gemma seed/source releases remain frozen.

- **Full Qwen 9B protocol-fixed run received November 6, 2026.** The user supplied `outputs/diagnosis-2026-11-06/qwen35-9b-random100-protocol-fix-Nov6th-02/`. All 100 questions/order match the prior random TRAIN sample; source hashes match pipeline 5/backend 3, and all 193 raw hashes/207 ZIP members were verified. Re-scoring gives 49 correct, 32 wrong, 19 abstentions, PA 0.17; mean 9.64 seconds versus Gemma fast-v5 23.19 seconds/PA 0.30. Checkpoint/prompt differences prevent a model-only comparison. 192 generations end at im_end; there is one answer-path parse failure from the token cap and two malformed schema-repair proposals. 31 of 79 auto-finals are wrong. CPU probes diagnose ignored plan fields, wrong time anchors/grains/conditions, missing entity/category mapping, debt-flow semantics, serial dates and SQL lineage false positives. Every wrong/abstained case has a diagnostic note; manually corrected CPU outputs are not a new model PA. Runtime/notebook/ZIP/SFT releases were not changed. See [full analysis and proposed priorities](inference-review-qwen9b-100-2026-11-06.md). This remains development on repeatedly inspected TRAIN, not independent holdout or official-rule verification.

- **Qwen 9B notebook output received later on November 6, 2026.** The current notebook records one interactive abstention in 5.905 seconds/two calls. Load output says pipeline 4 and generation_tokens None; its system prompt exactly matches the pre-repair source. Both raw actions fail old trailing-marker parsing, and the new diagnostic cell rejects the old in-memory agent before running the seven-question batch. The delivered local ZIP still contains pipeline 5/backend 3 and matches current runtime source; the notebook cannot distinguish a stale Drive ZIP from cached Python imports. Current parser CPU replay accepts both raw actions, but period kind day is unsupported, a separate interface error hidden by parsing. No repaired-runtime 9B benchmark or general Qwen quality comparison is available. See [notebook evidence and interpretation](inference-review-qwen9b-notebook-2026-11-06.md). Only analysis/docs were added; runtime, inference ZIP and Gemma SFT artifacts were not changed.

- **Qwen protocol repair, November 6, 2026.** The user authorized the proposed fixes for both 9B and 14B, and allowed unclear 14B documentation to be skipped. Implemented validated checkpoint/tokenizer/chat-turn EOS union, explicit generation overrides, consecutive terminal-marker parsing, concrete plan examples and field feedback, bounded consecutive parse retries and effective-token logging. Both pinned Qwen tokenizers resolve stop IDs to [248044, 248046]; Gemma retains [1, 106] and pad 0. A tiny random Qwen3.5 with forced logits verifies actual Transformers/backend turn-end stopping on CPU; this is not full-checkpoint inference. All 187 software tests pass. Archived raw replay parses 21/35 and rejects all 13 invented conversations; it is not a new answer score. Updated inference ZIP/notebook run the same first seven under a new run name. No repaired-run GPU speed/PA is available; Gemma SFT archives remain frozen. See [implementation, verification and Colab instructions](qwen-protocol-fix-2026-11-06.md). These engineering observations do not independently verify competition rules.

- **Partial Qwen upscaled-14B inference received November 6, 2026.** The user described seven questions; the supplied `outputs/diagnosis-2026-11-05/qwen35-upscaled14b-v2-random100-Nov5th/` actually contains ten completed questions, all abstentions, from the unchanged random 100 plan. These take 490.56 seconds and 35 model calls; the paired first seven take 362.29 seconds versus Gemma fast-v5's 225.23 seconds. All 35 raw outputs fail action parsing before answer-tool execution. Checkpoint/runtime generation EOS only includes endoftext 248044, omitting tokenizer turn-end im_end 248046; 22 outputs have both terminal markers, while the parser removes only one. Thirteen outputs continue with model-generated user turns. Repeated calls account for 87.47% of generation-call time. CPU trailer-only normalization parses 21 outputs, but none pass plan validation; an earlier-stop diagnostic parses 29 with three structurally valid plans, not a measured repaired-run PA. Input/source/raw hashes match; GPU BF16/cuda placement is unchanged and full SLM schema preparation is absent. Runtime and source ZIP were not modified during this diagnosis. See [latency, compatibility faults and proposed fixes](inference-review-qwen14b-2026-11-06.md). These development observations do not independently verify competition rules.

- **Optional inference checkpoint, November 6, 2026.** At the user's request to try `Pinkstackorg/Qwen-3.5-upscaled-14B-V2-noft`, added a separate registry key `qwen35-upscaled14b-v2`, keeping the official Qwen 9B key unchanged. Pinned revision `f4fe3d2ad8f2e14799624cdbafa5d46219bc1fb2` is dated March 24, 2026; its weight LFS hash matches upload commit `43418eac1081e1d44dbf1b94fce556b4976fc0a3` that day. Public repository metadata reports Apache 2.0. Actual safetensors header count is 14,599,485,040 parameters including 456,010,480 vision parameters, below the team's 15B constraint. Offline meta-model/tokenizer checks succeeded; no full-weight loading or generation benchmark was performed. Agent policy and fast-v5 budgets remain unchanged, and the Gemma-only SFT release is preserved. See [checkpoint evidence and Colab use](qwen35-upscaled14b-v2-colab.md). This does not independently verify official competition rules or organizer acceptance.

- **Full fast-v5 inference received November 6, 2026.** The user supplied `outputs/diagnosis-2026-11-05/gemma4-12b-random100-fast-v5-Nov5th/`. All 100 IDs and their order match baseline v3, with matching approved TRAIN fingerprint and recorded inference source hashes. Local re-scoring gives 46 correct, 16 wrong and 38 abstentions, PA 0.30 versus v3's 0.36. Mean time increases from 18.26 to 23.19 seconds, excluding model load, despite calls falling from 351 to 256; output tokens rise from 18,791 to 32,769. Both runtime records show the same checkpoint manifest, GPU, BF16 and greedy settings. CPU probes isolate denominator scope, entity matching, rank identity/contract, inventory source and SQL CTE lineage problems; these are diagnostic hand-edited variants, not new model inference or an improved PA. Runtime and SFT packages were not changed in this analysis. See [full comparison, evidence and proposed fixes](inference-review-fast-v5-2026-11-06.md). These are development observations, not independently verified competition requirements.

- **Gemma training seed preparation, November 5, 2026.** The user approved preparing data from the 72-question pipeline-v2 run and 100-question baseline-v3 run, requested Colab Pro, then restricted this preparation to Gemma. The 172 logs contain 171 unique training-source questions. Source-first task annotations replayed through fast v5 yield 136 accepted training questions / 144 assistant decisions, 17 queued training questions and 18 selected questions kept out of training under the existing business split. No new model inference or GPU fine-tuning was performed; these are curated tool-verified examples, not newly generated LLM trajectories. Both legacy bundles lack historical initial prompts. See [Gemma seed provenance, Colab workflow and limits](gemma-seed72-100-colab-pro.md). This implementation update does not independently verify competition rules.

- **Performance implementation, November 5, 2026.** After the partial fourth run, the user authorized the simplification recommendations. The new default uses deterministic preparation, independent mandatory question plans and automatic guarded execution/projection. Same-model schema repair is bounded and on demand, with partial acceptance and failure caching; code-only mode disables it. Verification has 142 passing software tests, 22/22 manual CPU regression calculations and replay/tokenizer checks, with no new GPU inference or model PA measurement. The notebook first runs the same seven questions before resuming the unchanged random sample of 100. See [fast v5 implementation and limits](fast-inference-v5-2026-11-05.md). This does not independently verify competition rules.

- **Partial fourth inference result received November 5, 2026.** Saved Colab output contains seven completed questions, all abstentions, and a KeyboardInterrupt during question eight. The seven completed records average 179.14 seconds and use 95 model calls. Preparation accounts for 1,045.55 of 1,253.95 recorded seconds (83.38%) and 53 calls. The same seven IDs in round three took 118.46 seconds with six correct and one wrong. A complete interactive trace shows whole-policy/schema rejection on citations and empty normalized facts despite an answerable source. These are partial results, not a full-100 benchmark. See [round-four diagnosis and proposed simplification](inference-review-7-v4-2026-11-05.md).

- **Implementation update, November 5, 2026.** The user authorized the recommendations following the third inference review. Pipeline v3 adds bounded inference-time business/rulebook interpretation using the same loaded model, explicit per-question plans, composable calculations, SQL checks and preparation/debug logging. Verification and remaining limitations are recorded in [pipeline v3](pipeline-v3-2026-11-05.md). The new pipeline has not yet produced a model inference benchmark, and this update does not independently verify competition rules.

- **Third inference result received November 4, 2026.** The user supplied `outputs/diagnosis-2026-11-03/gemma4-12b-stress100-baseline-v3/`. All 100 selected IDs match the prior random draw. Re-scoring with local training labels gives 49 correct, 13 wrong and 38 abstentions, with PA 0.36. CPU probes reproduce errors in time periods, cancellation scope, SQL projection and schema binding. The second and third runs share only one question, so their PA values are measurements on different samples. See [third-run review](inference-review-100-v3-2026-11-04.md).

- **Third-run sample, November 3, 2026:** the user requested 100 random questions without selection criteria. Prepared a uniform sample without replacement from all 3,000 training questions, seed 20261103, with no answer-type/language/business quotas, known-case insertion, old-question exclusion or holdout filtering. Files and the notebook plan match; no LLM inference has been run. The draw includes 18 questions from formerly reserved holdout businesses, so this run does not preserve those questions as unseen development evidence. Existing training partitions were not changed. See [third-run instructions](inference-round3-random100.md).

- **Logging clarification, November 3, 2026:** the user clarified that debugging requires the LLM's actual output, generated SQL, tool results and errors at each step. Existing JSON traces already contain these; readable Markdown exports were added for the received run and future batch runs, with LLM SQL distinguished from metric-generated SQL. See [step log guide](debug-step-logs.md). This change does not enable thinking or alter business rules.

- **Received inference result, November 3, 2026:** the user supplied the second 72-question development run for pipeline v2. Re-scoring confirms 56 correct, 1 wrong and 15 abstentions (PA 0.76389) on the same diagnostic sample. Source hashes, data fingerprint and logs match; CPU probes identify brittle question/period contracts and confirm that thinking output is explicitly disabled. See [second-run review and limitations](inference-review-72-v2-2026-11-03.md). This is a measured development result, not an estimate of private-test performance or official-rule verification.

- **Implementation update, November 3, 2026:** the user supplied a completed 72-question development diagnostic and authorized fixes to the protocol, data preparation and verification tools. See [analysis of the received run](inference-review-72-2026-11-03.md) and [pipeline v2 implementation/verification status](pipeline-v2-2026-11-03.md). These observations use the actual training rulebook and do not independently verify official competition-wide requirements.

- **Implementation update, November 1, 2026:** the user has now requested and authorized a modular-monolith implementation with selectable Qwen/Gemma models and a local test UI. See [architecture](architecture.md), [model comparison and pinned checkpoint evidence](model-comparison.md), and [actual verification status](verification-2026-11-01.md). The original sections below preserve the October 30 competition summary; implementation choices and observed data are recorded separately and do not establish new official rules.

- Saved on **October 30, 2026**, using the user's supplied competition summary.
- This file preserves the requirements, examples, and unknowns in that summary. It does not select an architecture, model, framework, development plan, or implementation strategy.
- The user attributes the summary to the official portal and reviewed screenshots/documentation. Those underlying screenshots and documents have not been supplied in this workspace.
- Official portal: <https://challenge.zalo.ai/portal/datamate/>.
- An attempt to read the portal on October 30, 2026 returned no readable page text through the browsing tool. This does **not** establish that the portal is unavailable; it means the rules below have **not been independently verified** in this session.
- Treat the details below as **user-provided competition context** pending comparison with the official materials. Actual split rulebooks and official submission/Docker instructions must be checked when available.
- Dates in this document describe the supplied context as of October 30, 2026; they are not a live status report.

## 1. General information

**Competition:** Zalo AI Challenge 2026.

**Track:** DataMate — Talking To Your Data Through Agentic AI.

**Final solution submission deadline:** November 16, 2026, 23:59 (GMT+7).

The participant is working as part of a team, is familiar with Python and programming fundamentals, and has no previous experience building AI Agents.

## 2. Problem definition

Build an AI Agent capable of answering natural-language Vietnamese questions about structured business data from Vietnamese small businesses.

The agent must understand questions and business terminology, retrieve relevant information, perform calculations, and return answers in the required structured format.

Challenges include:

- Vietnamese informal expressions, abbreviations, and colloquial questions.
- Unfamiliar database schemas and business-specific terminology.
- Reasoning over multiple tables and files.
- Applying business rules correctly.
- Numerical calculations and aggregations.
- Temporal expressions relative to a specified date.
- Recognizing questions that cannot be answered from available data.
- Exact structured outputs.

### 2.1 Input

```json
{
  "id": "q_06de33fa21cc",
  "business_id": "biz_118",
  "question": "sp bán dc nhiều nhất tuần rồi là gì",
  "as_of_date": "2026-03-04"
}
```

- `id`: Unique question identifier.
- `business_id`: Identifier of the business whose data should be used.
- `question`: Vietnamese natural-language question.
- `as_of_date`: Reference date for interpreting time expressions.

Each business supplies structured files and a `README.md` describing the business and potentially its data. Shared resources include a business rulebook and a solar–lunar calendar.

### 2.2 Expected output

```json
{
  "id": "q_06de33fa21cc",
  "type": "string",
  "answer": "Xi măng Hà Tiên PCB40 bao 50kg"
}
```

Supported answer types:

- `number`
- `percent`
- `string`
- `date`
- `boolean`
- `list`
- `table`
- `unanswerable`
- `abstain`

`unanswerable` means the necessary information is not available in the data. `abstain` means the agent chooses not to provide an answer. They have different scoring implications.

## 3. Evaluation

### 3.1 Per-question scoring

| Outcome | Score |
|---|---:|
| Correct | +1 |
| Wrong | -1 |
| Abstain | 0 |

An answer is correct only when its **type matches the reference type** and its **value satisfies the matching rule**. There is no partial credit.

An answer with type `abstain`, or a missing answer, counts as abstention.

### 3.2 Penalized Accuracy

```text
PA = (1 / N) × sum(s_i)
s_i = +1 for correct, 0 for abstain, -1 for wrong
```

Every question has equal weight. An `unanswerable` response to an answerable question is **wrong (-1)**.

### 3.3 Answer matching rules

**Number**

- Numerical values such as VND amounts, quantities, and counts.
- Exact matching.
- Reference values are integers or rounded to two decimal places.

**Percent**

- Percentage values expressed in percent units.
- Exact matching after rounding to two decimal places.
- The sign must be correct.

**String**

The prediction and each accepted reference form are normalized as follows:

1. Unicode NFC.
2. Lowercase.
3. Replace characters that are not letters, digits, or whitespace with spaces.
4. Trim leading/trailing whitespace and collapse runs of whitespace to one space.

Vietnamese diacritics are **not removed**. References may accept multiple forms, including alternative spellings and item codes.

**Date**

- ISO `yyyy-mm-dd` for dates.
- ISO `yyyy-mm` for month-specific questions.
- Exact matching.

**Boolean**

- JSON `true` or `false`.
- Exact matching.

**List**

- Ordered list of names or numbers.
- Order comes from the question or the rulebook default.
- Always compared position by position.
- Elements use the corresponding string or number matching rule.

**Table**

```json
{
  "columns": ["khach_hang", "so_tien_no"],
  "rows": [
    ["Chị Lan", 3200000],
    ["Anh Tú", 1500000]
  ]
}
```

- Columns are matched by normalized names.
- If column names do not match, columns are compared by position.
- Rows are compared as a set.
- Cell values use their respective string or number matching rules.

**Unanswerable**

Correct if and only if the reference is unanswerable.

**Abstain**

Always receives 0, regardless of the reference.

### 3.4 Ranking questions

- Use the ranking criterion in the question, or the rulebook when unspecified.
- Default ordering is descending by the criterion.
- Ties are ordered by ascending code.
- Questions whose reference ranking has a tie at the k-th position are excluded from test sets.
- If fewer than k candidates exist, the reference contains all candidates.

### 3.5 Final ranking tie-breakers

1. Total inference time in the organizers' re-run.
2. Earlier submission.

## 4. Dataset specification

### 4.1 Business types

Examples include grocery stores, coffee shops, bubble-tea shops, restaurants, pharmacies, building-material stores, motorbike repair shops, mother-and-baby stores, and other small businesses.

**The private test includes business types absent from training and public test data.**

### 4.2 Dataset characteristics

Each business contains approximately 12–30 months of activity and from a few hundred to about 300,000 rows. Seasonal effects include Tết, Mid-Autumn Festival, public holidays, paydays, and weekends.

Data may be supplied in one or more `.csv`, `.xlsx`, or `.sqlite` files. Each business also has a `README.md` whose quality varies: a complete data dictionary, informal notes, or almost no useful documentation.

### 4.3 Package structure

```text
<split>/
├── data/
│   ├── <business_id>/
│   │   ├── *.csv / *.xlsx / *.sqlite
│   │   └── README.md
│   └── ...
├── rulebook.md
├── calendar.csv
├── questions.jsonl
└── answers.jsonl
```

`answers.jsonl` is provided for training only.

### 4.4 Splits

| Split | Businesses | Questions | Reference answers |
|---|---:|---:|---|
| Training | 30 | ~3,000 | Available |
| Public Test | 10–20 | ~1,000 | Hidden |
| Private Test | 10–20 | ~500 | Hidden |

### 4.5 Training answers

Answers are joined to questions by `id`.

```json
{
  "id": "q_06de33fa21cc",
  "type": "string",
  "answer": "Xi măng Hà Tiên PCB40 bao 50kg",
  "alias": ["SP0001"]
}
```

`alias` provides accepted alternative name forms. Submissions do not need this field. Unanswerable reference questions have `answer: null`.

## 5. Business rules

Each split has its own `rulebook.md`, which defines default interpretations when a question does not specify them.

**Rulebooks can differ between training, public, and private splits. Use the rulebook for the current split.**

The following are examples from the supplied summary, **not immutable rules for every split**:

### Revenue

Revenue is the sum of line amounts after line-level discounts:

- Do not deduct order-level vouchers.
- Exclude VAT and delivery fees.
- Exclude cancelled orders.
- Exclude orders with any returned item.

### Best-selling products

“Bán chạy” defaults to ranking by **quantity**, not revenue.

### Calendar and time

- Weeks run Monday through Sunday.
- “Tháng này” means the month containing `as_of_date`, up to and including that date.
- The Tết window runs from lunar 23 tháng Chạp to mùng 10.

### Cost

Default cost uses weighted-average purchase cost by lot.

## 6. Technical requirements and competition rules

### 6.1 Model eligibility

Model weights must have been publicly released **before October 1, 2026**. The model license must permit the intended use. Fine-tuning eligible models is allowed.

### 6.2 Parameter limit

The total parameter count across **all models used at inference** must not exceed **15 billion**.

This includes language models, embedding models, rerankers, classifiers, and any other inference models, regardless of whether they are loaded simultaneously.

All mixture-of-experts parameters count. Quantization is allowed but does not reduce the parameter count used for this limit. Declare all models in the solution README.

### 6.3 Offline inference

Inference must run entirely offline inside the submitted Docker image. Commercial LLM APIs, external services, resource downloads during inference, and any network access are prohibited. Organizers run images with networking disabled.

### 6.4 No human in the loop

All answers must be generated automatically, with no manual intervention. Hard-coded answers and logic keyed on question identifiers or business identifiers are prohibited.

### 6.5 Code execution

Agents may generate and execute code in the container; this is expected but not mandatory. The mounted dataset is read-only. Write temporary files elsewhere.

### 6.6 Training data

Allowed:

- Provided training data.
- Self-generated synthetic data.
- Publicly available datasets.
- Publicly available eligible models.

Prohibited:

- Training on public test data.
- Training on private test data.
- Collecting answers to test questions by any means.

### 6.7 Reference hardware

- 2 × NVIDIA RTX 5090.
- 32 GB VRAM per GPU; 64 GB total VRAM.
- 512 GB system RAM.

According to the supplied summary, Blackwell GPUs require CUDA 12.8 or newer and compatible PyTorch/vLLM builds. Verify software compatibility against the official Docker guideline before implementation. The official guideline specifies the base image; its identity has not been provided here.

### 6.8 Runtime

The entire private test must finish within **6 hours of wall-clock time, including model loading**.

If a re-run exceeds the limit or terminates early, answers already written to `/result/submission.jsonl` are scored. Remaining unanswered questions count as abstain.

### 6.9 Reproducibility

The final score is determined by the organizers' Docker re-run.

```text
Reproducible when abs(PA(uploaded) - PA(re-run)) <= 0.03
```

For differences exceeding 0.03, organizers examine logs, may re-run again, and may request more information. Unexplained discrepancies may lead to disqualification.

## 7. Submission format

### 7.1 JSONL results

File: `submission.jsonl`. Each line must contain one valid JSON object.

```jsonl
{"id":"q_01234","type":"string","answer":"Sting dâu 330ml"}
{"id":"q_01235","type":"number","answer":48250000}
{"id":"q_01236","type":"percent","answer":-12.45}
{"id":"q_01237","type":"date","answer":"2026-08-15"}
{"id":"q_01238","type":"boolean","answer":true}
{"id":"q_01239","type":"list","answer":["Cà phê sữa","Bạc xỉu","Trà đào"]}
{"id":"q_01240","type":"table","answer":{"columns":["khach_hang","so_tien_no"],"rows":[["Chị Lan",3200000],["Anh Tú",1500000]]}}
{"id":"q_01241","type":"unanswerable","reason":"Dữ liệu không có giá vốn nên không tính được lợi nhuận"}
{"id":"q_01242","type":"abstain"}
```

`reason` is optional. Missing questions count as abstain.

### 7.2 Docker solution

The image must contain training source code, inference source code, required data and model weights, dependencies, `predict.sh`, and a README.

The entry point must process the dataset and write predictions.

| Purpose | Required path |
|---|---|
| Input | `/data` |
| Output | `/result/submission.jsonl` |
| Logs | `/result/logs/` |

Per-question logs must include at least question ID, final answer, and elapsed time. Extra logs may include generated code, execution results, errors, and other information explaining the answer. Organizers use logs to investigate reproducibility.

## 8. Competition rounds and dates

All times below use GMT+7.

### 8.1 Round 1 — Public Test

**October 26 – November 16, 2026.**

Teams develop solutions and run inference on public test data, then upload `submission.jsonl` to the portal.

- At most 5 submissions per day; the quota resets daily.
- Scores appear on the public leaderboard.
- Public scores are for development reference only.

### 8.2 Final solution submission

**November 16, 2026, 23:59.**

Submit the complete solution as a Docker package through the portal's Solution Submission form. Required materials include a public Google Drive download link, a checksum, and complete solution materials.

Each team may submit a final solution package up to 2 times. **Only the last submission is evaluated.** The solution is frozen before the private test phase.

### 8.3 Round 2 — Private Test

**November 20 – November 23, 2026.**

- Private data is released at **09:00 on November 20**.
- Teams run their already-submitted frozen solutions.
- Upload `submission.jsonl` by **November 23, 2026, 23:59**.
- The private leaderboard is displayed chronologically by submission time during this phase.
- Uploaded results are provisional.
- The official final ranking is based on organizers' re-runs of submitted Docker images.
- Organizers inspect source code and check prohibited behavior.

### 8.4 Results

- **December 2, 2026:** Top 2 teams from each track announced.
- **December 12, 2026:** Final rankings announced.

## 9. Available materials and unknowns

The user states that the following were reviewed when preparing the supplied summary:

- Challenge Description/Overview.
- Evaluation rules and scoring details, including screenshots.
- Data documentation: splits, structure, business rules, submission format.
- Competition Rules.
- Submission Guidance.

At the time this file was created, the actual downloaded training/public datasets and the separate Docker guideline had **not** been supplied in the conversation or workspace.

Consequently, the following are **not established**:

- Actual schemas, table and column names, keys, and data relationships.
- Actual data distributions, data quality, and README contents.
- Actual contents of split-specific `rulebook.md` and `calendar.csv`.
- The official Docker base image and full execution interface beyond the paths and `predict.sh` described above.
- Model selection or eligibility of any particular model.
- Architecture, framework, training strategy, deployment implementation, or project schedule.
- Accuracy, inference time, GPU memory consumption, or any benchmark performance.

Do not assume these details are known. Verify against the actual files and official materials when they become available.

## 10. Source and purpose

**Source of this saved file:** the user's consolidated competition context, supplied October 30, 2026.

**Official portal cited by the user:** <https://challenge.zalo.ai/portal/datamate/>.

Purpose: retain enough factual context for future work on the competition agent, without prematurely selecting a technical solution. Keep competition requirements, observations from real data, and the team's future design decisions distinct.
