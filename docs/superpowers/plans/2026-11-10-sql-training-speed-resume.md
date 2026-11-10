# SQL Training Speed and Resume Implementation Plan

> For agentic workers: execute inline with superpowers:executing-plans; no per-task agent fan-out. Preserve all existing user work.

**Goal:** eliminate the observed DeltaNet fallback and make notebook training recoverable from complete Drive checkpoints.

**Architecture:** pinned GPU-only kernel installation/probe; shared checkpoint integrity/resume helpers consumed by Trainer and notebook; training-only v1.3 release derived from immutable v1.2 data.

**Tech stack:** Python, PyTorch, Transformers 5.22.x, PEFT, FLA, causal-conv1d, Colab A100.

**Spec:** ../specs/2026-11-10-sql-training-speed-resume-design.md

## Global constraints

Full weights run only on remote GPU. TRAIN only; labels/context/objective and base revision unchanged. No overwrite of existing runs/packages/frozen archive. Resume requires matching model, dataset, resolved training identity. No measured speed claims without GPU results. Save every 5 steps; evaluate every 50.

## Review focus

Corrupt newest checkpoint falls back; incompatible run never resumes; a kernel import that silently falls back is rejected; notebook rerun after success does not train again; interrupted writes never receive a complete marker.

## Tasks

- [x] 1. Add checkpoint integrity/selection and callback tests (including interrupted save, changed config, real tiny Trainer resume); implement checkpoint helper and periodic saves.
- [x] 2. Add kernel-dispatch and speed-profile tests; implement a GPU forward/backward probe and pinned dependency profile preserving Torch; integrate preflight and environment reporting. GPU probe still requires execution on Colab.
- [x] 3. Update notebook orchestration for automatic resume, completed-run reuse, local model staging and persistent timing/logs. Test executed generated cells with controlled subprocesses.
- [ ] 4. Run focused and full CPU checks, review changes, create v1.3 source/data/notebook release with identical example bytes; preserve baseline evidence and document recovery instructions and measurement limits. Run remote kernel/pilot verification if available and authorized at the UI boundary.

## Execution ledger

- Baseline inspected: updated v1.2 notebook has a completed A100 pilot and fallback warnings. Current config cell was changed by user to full; do not trigger it.
- Ruling: retain the shared workspace because the user's updated uncommitted notebook and all source artifacts are here; do not switch branches or commit unrelated files.
- Ruling: use FLA 0.6.0's own Triton causal conv when the native package is absent, with explicit layout adapter and GPU numerical probe. This avoids a CUDA extension build on Python 3.13/Torch 2.14; native conv is retained if available. Exact wheel metadata/SHA verified from PyPI downloads.
- Checkpoint CPU proof: both a tiny generic Trainer and a tiny GPT2+PEFT LoRA run were interrupted after checkpoint 1, then resumed to step 3; final tensors matched uninterrupted training exactly, including dropout/RNG effects.
- Resume is host-locked to avoid launching two child trainers for one output on the same runtime. Do not run two Colab servers against the same RUN_TAG.
- Review fixes verified: malformed JSON checkpoint fallback; atomic root metadata recovery; relative gradient checks reject zero backward; final-checkpoint export without an extra optimizer update. Focused training/notebook checks: 31 passed. Full CPU suite: 272 passed, 1 transient Windows ZIP replace permission failure in Team200; that test passed on isolated retry without code changes. GPU validation remains unperformed.
- User requested an early stop to conserve Codex credit. Finish local v1.3 packaging/integrity verification only; do not launch GPU training or more long benchmarks.
