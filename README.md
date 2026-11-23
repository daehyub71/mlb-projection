# MLB Player Projections 2027 — just for fun

**English** | [한국어](README_KO.md)

> Work in progress. These are fan-made projections for fun. They are not betting advice and come with no guarantee.

Projects the 2027 season for every MLB player who appeared in the last five seasons (2022–2026):

- **Batters**: OPS, plate appearances (PA), home runs (HR)
- **Pitchers**: ERA, innings pitched (IP), K−BB%

Every projection is shown next to a league-average baseline. The models are also backtested against the 2024, 2025 and 2026 seasons, and those errors are published as they are.

## Status

| Milestone | Scope | Status |
|---|---|---|
| M0 | Setup, data contracts, test fixtures, web skeleton | in progress |
| M1–M4 | Collection, cleaning, models, backtests, 2027 projections | planned |
| M5–M7 | Design, web app, deployment | planned |

Plans and decisions are written up (in Korean) in [`docs/`](docs/): `SPEC.md`, `PLAN.md` and `TASKS.md`.

## Data sources

- [MLB Stats API](https://statsapi.mlb.com): season stats and player info
- [Baseball Savant](https://baseballsavant.mlb.com): Statcast leaderboards, Outs Above Average and Fielding Run Value

MLB data © MLB Advanced Media, L.P. Only individual, non-commercial, non-bulk use is permitted. This repository does not include the raw downloads (`data/raw/` is git-ignored). The small test fixtures (12 players or fewer per file) are listed in [`tests/fixtures/SOURCE.md`](tests/fixtures/SOURCE.md).

## Tech stack

- **Pipeline**: Python 3.13, standard-library HTTP, pandas; tests with pytest, ruff and mypy
- **Web**: Next.js 16 (static export), React 19, TypeScript, Tailwind CSS 4, the [Astryx](https://github.com/meta/astryx) design system; tests with Vitest

## Development

```bash
python3.13 -m venv venv
venv/bin/pip install -r requirements-dev.txt && venv/bin/pip install -e . --no-deps
venv/bin/ruff check . && venv/bin/mypy && venv/bin/pytest

cd web && npm ci && npm run lint && npm test && npm run build
```

## Structure

```
mlbproj/      pipeline package (contracts, collect, validate, clean, models, export)
contracts/    JSON schema shared by the pipeline and the web app
tests/        pytest suite and fixtures
web/          Next.js static site
docs/         SPEC, PLAN, TASKS (Korean)
```
