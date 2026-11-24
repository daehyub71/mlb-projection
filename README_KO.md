# 재미로 보는 2027 MLB 선수 성적 예측

[English](README.md) | **한국어**

> 만드는 중입니다. 재미로 보는 팬 예측이고, 실제 성적을 보장하지 않습니다. 베팅 조언이 아닙니다.

최근 5시즌(2022~2026)에 MLB 기록이 있는 선수 전원의 2027 성적을 예측합니다.

- **타자**: OPS, 타석(PA), 홈런(HR)
- **투수**: ERA, 이닝(IP), K−BB%

예측값은 언제나 리그 평균 기준선과 나란히 보여 줍니다. 같은 모델로 2024·2025·2026 시즌을 맞혀 보는 과거 검증도 하고, 그 오차를 그대로 공개합니다.

## 진행 상황

| 마일스톤 | 범위 | 상태 |
|---|---|---|
| M0~M4 | 기반 설정, 수집, 정제, 모델, 과거 검증, 2027 예측 | 완료 |
| M5 | 디자인(테마·시안) | 진행 중 |
| M6~M7 | 웹, 배포 | 예정 |

## 과거 검증 결과

모델 파라미터와 확장 모델 채택은 2024·2025 시즌만으로 정했습니다. 그 설정을 고정한 채 2026 시즌을 한 번만 시험했습니다([`data/out/final_test_2026.json`](data/out/final_test_2026.json)). 평균 절대 오차이며 낮을수록 좋습니다.

| 지표 | 이 모델 | 리그 평균 | 직전 시즌 |
|---|---|---|---|
| OPS (200타석 이상) | **.062** | .069 | .098 |
| K−BB% (50이닝 이상) | **3.62** | 4.40 | 4.43 |
| ERA (50이닝 이상) | **0.84** | 0.93 | 1.25 |
| PA | **72.4** | — | 74.9 |
| IP | **17.1** | — | 17.6 |
| HR | **2.61** | — | 2.91 |

2026 시험은 결함을 고친 뒤 한 번 더 돌렸습니다. 표본이 적은 타자의 Statcast 특성을 리그 평균 쪽으로 당기지 않아서, 1타석 선수가 OPS .963으로 예측되던 문제입니다. 이 결함은 2026 결과가 아니라 2027 예측표에서 드러났습니다. 첫 결과(OPS .061, K−BB% 3.63, ERA 0.85, HR 2.60)는 [`data/out/final_test_2026_run1.json`](data/out/final_test_2026_run1.json)에 그대로 두었고, 표는 두 번째 결과입니다.

출전량 예측은 보수적입니다. 부상·결장 위험을 평균에 반영하므로 주전 선수는 평소 출전량보다 낮게 예측되는 편입니다.

요구사항·계획·진도는 [`docs/`](docs/)의 `SPEC.md`·`PLAN.md`·`TASKS.md`에 있습니다.

## 데이터 출처

- [MLB Stats API](https://statsapi.mlb.com): 시즌 기록, 선수 정보
- [Baseball Savant](https://baseballsavant.mlb.com): Statcast 리더보드, Outs Above Average, Fielding Run Value

MLB 자료의 저작권은 MLB Advanced Media, L.P.에 있으며, 개인적·비상업적·소량 이용만 허용됩니다. 이 저장소에는 원본 데이터를 올리지 않습니다(`data/raw/`는 git 제외). 테스트용 소규모 픽스처(파일당 12명 이하)의 출처는 [`tests/fixtures/SOURCE.md`](tests/fixtures/SOURCE.md)에 적었습니다.

## 기술 스택

- **파이프라인**: Python 3.13, 표준 라이브러리 HTTP, pandas · pytest, ruff, mypy
- **웹**: Next.js 16(정적 export), React 19, TypeScript, Tailwind CSS 4, [Astryx](https://github.com/meta/astryx) 디자인 시스템 · Vitest

## 개발

```bash
python3.13 -m venv venv
venv/bin/pip install -r requirements-dev.txt && venv/bin/pip install -e . --no-deps
venv/bin/ruff check . && venv/bin/mypy && venv/bin/pytest

# 파이프라인 (원본은 각자 내려받고 커밋하지 않습니다)
venv/bin/python -m mlbproj collect && venv/bin/python -m mlbproj validate && venv/bin/python -m mlbproj build
venv/bin/python -m mlbproj backtest     # 2024·2025 과거 검증
venv/bin/python -m mlbproj export       # 2027 예측 → data/out CSV + 웹 JSON

cd web && npm ci && npm run lint && npm test && npm run build
```

## 구조

```
mlbproj/      파이프라인 패키지 (계약, 수집, 검증, 정제, 모델, 내보내기)
contracts/    파이프라인과 웹이 함께 쓰는 JSON 스키마
tests/        pytest 테스트와 픽스처
web/          Next.js 정적 사이트
docs/         SPEC, PLAN, TASKS
```
