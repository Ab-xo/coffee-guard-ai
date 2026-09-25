# Contributing to CoffeeGuard AI

## Setup

```bash
uv sync --all-extras        # Python 3.12 + all dependencies into .venv
uv run pre-commit install   # optional: run lint/format hooks on every commit
```

On a drive without hard-link support (e.g. exFAT), set `UV_LINK_MODE=copy`.

## Workflow

- One short-lived branch per plan phase (`phase/<n>-<topic>`), merged into `main` via pull request once CI is green; each merged phase is tagged (`v0.1-data`, …, `v1.0`).
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- Every step's results and decisions are recorded in [`docs/PROGRESS.md`](docs/PROGRESS.md).

## Code conventions

- All logic lives in `src/coffeeguard/` and is run through the `coffeeguard` CLI; notebooks only present artifacts.
- Configuration is YAML validated by pydantic models in `config.py` — add a field there rather than reading ad-hoc keys.
- `coffeeguard.inference` must depend only on numpy, pillow and onnxruntime (it ships in the API image).
- Style is enforced by ruff (`line-length = 100`): `uv run ruff check . && uv run ruff format .`
- Type hints on public functions; docstrings explain *why* when it isn't obvious.

## Tests

```bash
uv run pytest -m "not slow"   # fast unit tests
uv run pytest                 # everything, incl. the CPU train/export smoke test
```

- Tests use the synthetic dataset in `tests/fixtures/synthetic.py`; they never need the real data.
- Data and inference code need tests for failure paths (corrupt files, wrong labels, leakage), not only the happy path.

## Rules for results

- The test split is used only to report final candidates, never to choose anything.
- Any number quoted in docs must trace back to a run directory (config + git SHA + data fingerprint).
- No notebook outputs and no files > 5 MB committed outside model bundles.
