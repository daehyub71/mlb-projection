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
| M0 | 기반 설정, 데이터 계약, 테스트 픽스처, 웹 골격 | 진행 중 |
| M1~M4 | 수집, 정제, 모델, 과거 검증, 2027 예측 | 예정 |
| M5~M7 | 디자인, 웹, 배포 | 예정 |

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
