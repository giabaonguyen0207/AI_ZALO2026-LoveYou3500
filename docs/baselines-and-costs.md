# DataMate baseline proposals and GPU fine-tuning budgets

Research date: **October 30, 2026**. Currency: **USD**.

Status: research and proposals only. No model has been downloaded, selected, trained, or benchmarked on DataMate. No GPU rental has been purchased. Competition requirements are taken from `competition-context.md`; the original organizer documents still need to be checked.

## Recommendation

Start with **Qwen3.5-9B without fine-tuning**, using a tool loop that executes SQL/Python and formats the result deterministically. Use Qwen3.5-4B as the cheaper engineering baseline. Compare Gemma 4 12B and Ministral 3 14B Reasoning as larger alternatives before deciding which checkpoint merits fine-tuning.

These are current, competitive open-weight candidates within the size limit, not established DataMate winners. General reasoning, coding, and tool-use benchmarks do not establish Vietnamese business-data accuracy. A larger parameter count is not a guarantee of a better score.

## Eligibility filter and parameter accounting

The user explicitly requires **publicly available weights before October 1, 2026**. An announcement, paper, repository creation date, or October release does not independently satisfy that requirement.

- Each baseline below is a separate deployment, using one language-model checkpoint. Do not add their parameter counts together unless combining them into one submitted system.
- Initial baselines use SQL metadata, lexical search/BM25, and programmatic checks, without a separate embedding model, reranker, classifier, or learned verifier.
- Count all deployed parameters, including embeddings and applicable vision/prediction modules. Marketing names and active MoE parameter counts are insufficient.
- Merge a selected LoRA adapter into its base model where supported; otherwise include the adapter parameters in the final inventory. Do not assume multiple specialized adapters are free under the cap.
- Before training, resolve and record the full checkpoint revision, pre-cutoff weight publication evidence, license, file hashes, and exact unique parameter count. The short revisions below are leads from repository histories, not a completed artifact manifest.

| Tier | Candidate checkpoint | Parameter estimate for a standalone baseline | Evidence of release before cutoff | Suggested role |
|---|---|---:|---|---|
| Small | `Qwen/Qwen3.5-4B` | ~4.66B for full repository checkpoint | Weights upload in March 2026; history revision `851bf6e` is dated March 2 | Fast development and simple SQL agent |
| Medium | `Qwen/Qwen3.5-9B` | ~9.65B for full repository checkpoint | Weights upload in March 2026; history revision `c202236` is dated March 2 | First practical main baseline |
| Medium alternative | `ibm-granite/granite-4.2-8b` | ~9B, rounded repository display | Released August 25, 2026; pre-cutoff history includes September 4 revision `f8de16c` | Enterprise/tool-use comparison; Vietnamese capability needs measurement |
| Large | `google/gemma-4-12B-it` | 11.95B in official specification | Released June 3, 2026; history includes July 20 revision `707f0a3` | Alternative model family for coding and multi-step tools |
| Near cap | `mistralai/Ministral-3-14B-Reasoning-2512` | ~13.9B: 13.5B language + 0.4B vision | Released December 2, 2025; history includes July 15 revision `51f9210` | Larger reasoning baseline; check latency and Vietnamese quality |

All five model repositories list Apache 2.0. Final artifact/license checks still apply to the exact revision and any extra resources.

Sources for the table:

- Qwen [4B card](https://huggingface.co/Qwen/Qwen3.5-4B), [4B parameter metadata](https://huggingface.co/api/models/Qwen/Qwen3.5-4B), and [4B history](https://huggingface.co/Qwen/Qwen3.5-4B/commits/main).
- Qwen [9B card](https://huggingface.co/Qwen/Qwen3.5-9B), [9B parameter metadata](https://huggingface.co/api/models/Qwen/Qwen3.5-9B), and [9B history](https://huggingface.co/Qwen/Qwen3.5-9B/commits/main). Metadata reports 4,659,865,088 and 9,653,104,368 parameters respectively; these are conservative full-checkpoint inventories rather than just the named language backbones.
- IBM [release announcement](https://research.ibm.com/blog/introducing-granite-4-2), [model card](https://huggingface.co/ibm-granite/granite-4.2-8b), and [history](https://huggingface.co/ibm-granite/granite-4.2-8b/commits/main).
- Google [June 3 release guide](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/), [official model specification](https://ai.google.dev/gemma/docs/core/model_card_4), and [history](https://huggingface.co/google/gemma-4-12B-it/commits/main).
- Mistral [release announcement](https://mistral.ai/news/mistral-3/), [model card](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512), and [history](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512/commits/main).

### Recent challenger requiring a historical checkpoint: K2 Horizon 7B

The [IFM launch announcement](https://ifm.ai/blog/k2/) is dated **September 3, 2026** and reports strong coding and agent results for the original K2 Horizon 7B. It is a useful additional challenger to Qwen3.5-9B, with no established Vietnamese DataMate result here.

However, the [repository history](https://huggingface.co/IFM/K2-Horizon-7B/commits/main) shows that a new `anti_doom_loop_verified` checkpoint was published and promoted to `main` on **October 1, 2026** (`9b31012` / `85d46bb`). **That replacement fails the user's strict pre-October cutoff.** Do not copy the current model card's October checkpoint recipe or apply its later benchmark improvements to the September weights.

The history contains a **September 28** revision, `f846b1e`, which is a candidate starting point for locating the pre-cutoff artifact. Direct historical tree/card retrieval did not succeed through the browsing tool in this session. Confirm that the historical revision resolves to the intended complete weights before treating this as an approved eligible option. The current repository display rounds the complete checkpoint to ~9B; recount the historical checkpoint separately.

The [current model card](https://huggingface.co/IFM/K2-Horizon-7B) lists Apache 2.0, downstream SFT support, custom runtime code, and long reasoning budgets. These indicate extra integration and latency work; they are not a verified training recipe or speed measurement for the September checkpoint.

### Models excluded by the total-parameter rule

Do not select a 20B+ checkpoint or a large MoE solely because its active count fits under 15B. For example, Google's [Gemma 4 specification](https://ai.google.dev/gemma/docs/core/model_card_4) lists 25.2B total parameters for the 26B-A4B model despite ~3.8B active parameters. It fails the cap.

## Agent baseline designs

Keep a common evaluation harness and ingestion layer so model comparisons are interpretable.

### Common execution path

1. Read the current split's rulebook, calendar, business README, schemas, value samples, and basic profiles.
2. Preserve CSV/XLSX/SQLite semantics while making relevant tables available to a SQL engine. Keep mounted data read-only and temporary artifacts outside it.
3. Give the model tools for schema inspection, value search, read-only SQL, and Python calculations when SQL is insufficient.
4. Have code calculate numbers, apply specified rounding and ordering, and serialize the required answer JSON.
5. Log the tool evidence, answer, errors, and time. Keep missing-data decisions distinct from abstention.

Use lexical retrieval initially; add a learned retriever only if validation shows enough benefit to justify its parameters and complexity.

### Small: constrained SQL agent

- Qwen3.5-4B, with question, relevant schema, rulebook excerpts, and `as_of_date` in context.
- Generate SQL, execute it, and permit a small bounded repair budget.
- Start with a maximum of two repair attempts, then evaluate how the budget affects accuracy and runtime.
- Appropriate for fast iteration and diagnosing ingestion/output failures. Complex joins, ambiguous business language, and uncommon rule interpretations remain hypotheses to test.

### Medium: tool-using SQL/Python agent

- Qwen3.5-9B as the first candidate; Granite 4.2-8B as an alternate.
- Inspect unfamiliar schemas, search sample values, issue SQL/Python calls, and revise based on execution results.
- Begin with a bounded 3–6 tool-action budget. Tune it on held-out training businesses.
- Use the same checkpoint to inspect its proposed answer when useful; a second separately trained verifier model is not required.
- K2 Horizon's verified September checkpoint can join this comparison once the artifact check passes.

### Large: plan, execute, verify using one checkpoint

- Gemma 4 12B or Ministral 3 14B Reasoning, evaluated separately.
- Produce an explicit query plan covering date range, joins, business-rule exclusions, aggregation, and answer type.
- Execute the plan; use a subsequent call to the same checkpoint to inspect the plan and tool evidence.
- Optionally evaluate a second independent query formulation for difficult questions. Agreement is evidence, not proof: two wrong queries can agree.
- Keep output-token and tool budgets bounded and measure them against the six-hour limit. Additional model calls increase runtime even when they add no new model weights.

These architecture differences are experimental options. Also compare all models under the same tool budget before attributing gains to model size.

## Fine-tuning method and training-data assumptions

Estimate a **supervised fine-tuning (SFT)** pilot using **QLoRA**: keep base weights quantized to 4-bit and train small low-rank adapter matrices, starting around rank 16–32. This is not full-parameter training, reinforcement learning, or pretraining.

Pilot assumptions, not observations from the actual dataset:

- 2,500 training examples, after reserving other training businesses for validation.
- Average 4,096 processed tokens per example, including prompt/context, tool interaction, and response.
- 2 epochs.
- **20,480,000 processed training tokens** in total.
- Maximum training sequence length initially around 4K; memory use depends on batch size and implementation.
- One rented GPU, BF16 computation where supported, gradient checkpointing, and a compatible trainer.
- Aggregate training throughput below means tokens across the whole batch per second, not interactive generation tokens per second.

The supplied competition summary describes final answers, not verified tool trajectories. Preparing trustworthy SQL/Python/tool examples is additional work. Do not assume 2,500 suitable traces are already available.

[Unsloth's Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) supports text fine-tuning for 4B/9B and reports BF16 LoRA examples at 10GB/22GB. [Gemma 4](https://unsloth.ai/docs/models/gemma-4/train) and [Ministral 3](https://unsloth.ai/docs/models/ministral-3) also have fine-tuning guides. Those claims do not guarantee a specific 4K batch configuration fits. A100 choices below provide headroom; they are not claimed minimum VRAM requirements. Granite and historical K2 compatibility need a short training smoke check before a full run.

## GPU rental prices and cost model

[Runpod's listed Pod prices](https://www.runpod.io/pricing), checked October 30, 2026 (page marked updated October 8):

| GPU | VRAM | Listed hourly rate used |
|---|---:|---:|
| RTX 4090 | 24GB | $0.89 |
| RTX 5090 | 32GB | $1.19 |
| A100 PCIe | 80GB | $1.79 |
| H100 PCIe, optional comparison only | 80GB | $2.89 |

These are reference listed prices, not a reserved-capacity quote. Availability and cloud tier can change the checkout price. Storage, taxes, transfer, and idle time are additional.

```text
processed_tokens = examples × average_tokens × epochs
billable_hours ≈ processed_tokens / (training_tokens_per_second × 3,600)
                 + setup/checkpoint/export overhead
GPU_cost ≈ billable_hours × number_of_GPUs × hourly_rate
```

### One pilot run

Throughput and overhead are **planning assumptions**, not published model-specific speed measurements and not measured in this project. Costs below are rounded outward from the calculation. They exclude generation-based validation runs.

| Model | GPU | Assumed training tokens/s | Overhead | Estimated billed hours | Estimated GPU cost |
|---|---|---:|---:|---:|---:|
| Qwen3.5-4B | 1 × RTX 4090 | 400–1,200 | 2h | 6.7–16.2h | **$6–15** |
| Qwen3.5-9B | 1 × RTX 5090 | 200–600 | 2h | 11.5–30.4h | **$13–37** |
| Granite 4.2-8B | 1 × RTX 5090 | 200–600 | 2h | 11.5–30.4h | **$13–37** |
| K2 Horizon 7B, historical artifact pending | 1 × A100 80GB | 150–500 | 3h | 14.4–40.9h | **$25–74** |
| Gemma 4 12B | 1 × A100 80GB | 150–450 | 2h | 14.6–39.9h | **$26–72** |
| Ministral 3 14B Reasoning | 1 × A100 80GB | 120–350 | 2h | 18.3–49.4h | **$32–89** |

H100 is not automatically cheaper overall despite potentially higher throughput: compare measured tokens per dollar. A single-GPU baseline avoids distributed-training work initially.

### Practical development allowance

Allow for three training attempts, generation-based validation, exports, storage, and some failed or idle time:

| Candidate | Suggested planning allowance |
|---|---:|
| Qwen3.5-4B | **$50–100** |
| Qwen3.5-9B or Granite 4.2-8B, each | **$100–200** |
| Historical K2 Horizon 7B, if verified | **$200–350** |
| Gemma 4 12B | **$200–350** |
| Ministral 3 14B Reasoning | **$250–450** |

These allowances are estimates, not promised totals. A failed setup or long agent rollout can exceed them. Data authoring, synthetic-data generation, human labor, full public/private inference, and a final two-5090 compatibility run are not included. Teacher-model generation, if used, needs its own eligibility/rules review and budget.

For a dataset ten times larger, the training portion of cost grows roughly tenfold at unchanged token length, epochs, and throughput; setup overhead does not. Longer sequences can lower throughput, so scaling by token count alone may understate the increase. Reinforcement learning adds repeated rollouts and requires a separate estimate.

## Evaluation and investment order

1. Build an unmodified-model baseline and the ingestion/execution/scoring harness. Fine-tuning spend is $0 at this stage; inference rental still costs money.
2. Hold out entire businesses from the provided training set to test unfamiliar schemas and terminology. Keep their labels out of training-data generation. Never train on public/private test data.
3. Compare Qwen3.5-4B, Qwen3.5-9B, Gemma 4 12B, and Ministral 3 14B on the same held-out questions and tool budget. Add the newer challengers if useful.
4. Examine errors before training: schema linking, date resolution, rule application, SQL execution, missing-data recognition, exact formatting, and abstention.
5. Fine-tune the most promising model on verified executable trajectories and compare against its unchanged base under the same conditions.
6. Benchmark the selected exported checkpoint inside the required offline container on the organizer GPU class. Training fit is not evidence of final inference throughput.

With +1 correct, -1 wrong, and 0 abstain, a perfectly calibrated probability p of a fully correct typed answer gives expected score `2p - 1`; answering is favorable only above 0.5. This is a mathematical decision threshold, not a recommendation to trust model self-reported confidence. Calibrate a practical abstention policy on held-out training businesses.

For approximately 500 private questions, six hours corresponds to 43.2 seconds per question under serial execution before reserving load/setup time. Batching and two GPUs can change throughput; measure total wall time and score together.

An initial **$150–300 GPU budget focused on Qwen3.5-9B** is a reasonable planning proposal. Compare larger models before committing further spend. No particular Penalized Accuracy or six-hour completion claim can be made before the dataset and benchmarks exist.
