# ADR 003 — OOD gate: augmented reference set instead of a higher threshold

**Status:** accepted (release 1.1.0) · **Date:** 2026-09-26 · **Relates to:** [ADR 002](002-ood-threshold-adjustment.md) (τ 0.554 → 0.68), which it supersedes

## Context

Clear coffee-leaf photos were rejected as "not a coffee leaf" (example: KNN distance 0.605 > τ 0.554). Two fixes were made in parallel:

- **ADR 002 (on `main`):** keep the reference set, raise τ from 0.554 to 0.68. Its effects were estimated, not measured.
- **v1.1 (on `eleni-changes`):** find out *why* such photos score high, fix that, and re-fit τ by the unchanged rule.

The cause, measured on coffee leaves from two other datasets (BRACOL, Brazil; RoCoLe, Ecuador; never used for training or fitting — see the model card): the reference set held only clean, full-size training photos, in which every leaf on paper lies horizontally. Vertical leaves and small web/chat-app copies were therefore far from all references even though the classifier handles them as well as clean photos. v1.1 adds each training photo turned 90° and as a small re-compressed copy (7,056 references) and re-fits τ = 0.494.

## Both options measured on the same data

Release model, full decision (quality gate → OOD gate → confidence); `coffeeguard ood external` plus the Phase 6 OOD sets.

| | ADR 002: old references, τ 0.68 | **v1.1: augmented references, τ 0.494** |
|---|---:|---:|
| AUROC: coffee leaves from other datasets vs. other plants' leaves | 0.712 | **0.981** |
| Bean leaves (OOD test) not turned away | **96%** | 4% |
| Bean leaves given a confident disease answer | **16%** | 2% |
| Banana / tomato leaves (OOD cal) not turned away | 35% | 0% |
| Non-leaf images not turned away (test / cal) | 3% / 2% | 0% / 0% |
| BRACOL vertical + 300 px JPEG passed | 88% | **98%** |
| BRACOL 150 px JPEG passed | 67% | **99%** |
| BRACOL as photographed / leaf vertical / 300 px JPEG passed | 100% / 99% / 99% | 85% / 98% / 84% |
| RoCoLe (on the plant) passed | 100% | 93% |
| Own test photos answered / accuracy / uncertain / rejected | 91.3% / 99.1% / 8.2% / 0.5% | 90.2% / 99.4% / 5.8% / 4.0% |

A threshold moves along one ranking; it cannot improve the ranking. With the old references, coffee leaves from other datasets and other plants' leaves overlap (AUROC 0.712), so a τ loose enough for the former lets almost all bean leaves through — and one in six then gets a confident Cercospora/Rust/Phoma answer.

## Decision

Use the augmented reference set with τ from the unchanged fitting rule (v1.1, `artifacts/models/coffeeguard-effv2b0-v1` version 1.1.0). ADR 002's analysis of the problem stands; its fix (τ 0.68) is superseded. ADR 002 and the notebooks added with it are kept as the record of that option.

## Consequences

- The gate keeps its purpose (other plants and non-leaf images turned away) while accepting vertical and small coffee-leaf photos.
- Cost: about 4% of our own clean test photos and 15% of clean horizontal BRACOL photos are turned away, where τ 0.68 turned away almost none. A later refinement could fit τ on a validation set that includes turned and small copies.
- Neither option fixes the classifier: on unseen BRACOL photos it is right 63% of the time (Cercospora 1/61). That needs more varied training data and a field test set.
- Reproduce: `coffeeguard ood fit -b <bundle>` (augmented by default; `--no-augment-bank` gives the old reference set), `coffeeguard ood external-collect`, `coffeeguard ood external -b <bundle>`; for ADR 002's option, set `ood.tau` to 0.68 in a copy of the v1.0 bundle.
