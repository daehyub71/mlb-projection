# CLAUDE.md — mlb-projection

재미로 보는 2027 MLB 선수 성적 예측. 최근 5시즌(2022~2026) 기록으로 타자 OPS·PA·HR, 투수 ERA·IP·K−BB%를 예측하고, 2024~2026 과거 검증 결과와 함께 정적 웹으로 공개한다.

작업 전에 `docs/SPEC.md` → `docs/PLAN.md` → `docs/TASKS.md` 순으로 읽는다(DESIGN은 M5에서 생긴다). 워크스페이스 규칙(`../CLAUDE.md`) 위에 아래 규칙을 더한다.

## 스택

| 층 | 내용 |
|---|---|
| 파이프라인 | Python 3.13(**Homebrew** — python.org판은 urllib HTTPS 인증서 실패, SPEC E17), 표준 라이브러리 `urllib`, pandas 3 · pytest·ruff·mypy(strict) |
| 웹 | `web/` — Next.js 16 정적 export + React 19 + TypeScript 5.9 + Tailwind 4 + **Astryx 0.7.17**(정확히 고정) · Vitest |
| 계약 | `mlbproj/contracts.py`(열·경로·URL·상수), `contracts/web-data.schema.json`(웹 JSON) |

## 실행

```bash
# 처음 한 번
/opt/homebrew/bin/python3.13 -m venv venv
venv/bin/pip install -r requirements-dev.txt && venv/bin/pip install -e . --no-deps
cd web && npm ci

# 파이프라인 (명령은 마일스톤마다 붙는다, PLAN §2-4)
venv/bin/python -m mlbproj --help
```

## 검증 (변경 후 모두 통과해야 한다)

```bash
# 파이프라인 — 저장소 루트에서
venv/bin/ruff check . && venv/bin/ruff format --check .
venv/bin/mypy
venv/bin/pytest            # 외부 요청 테스트는 제외된다. 따로 돌릴 때: venv/bin/pytest -m live

# 웹 — 같은 호출 안에서 저장소 루트 기준으로 cd 한 뒤 (병렬 호출은 cwd를 공유한다)
cd "$(git rev-parse --show-toplevel)/web" && npm run lint && npm test && npm run build
npx astryx doctor --built out
```

## 규칙

- **공개 저장소**: `data/raw/`·`data/interim/`은 커밋하지 않는다(SPEC D6·N4, `tests/test_repo_hygiene.py`가 검사). 실제 데이터 픽스처는 파일당 12명 이하로 두고 `tests/fixtures/SOURCE.md`에 출처·이유를 적는다(PLAN P4).
- **계약은 메인 세션만 고친다**: `contracts.py`·`web-data.schema.json`을 바꾸면 `tests/fixtures/synthetic/web/` 샘플과 테스트를 같이 고친다. 스키마 검사기(`mlbproj/schema.py`)는 지원하지 않는 키워드에서 오류를 낸다 — 새 키워드를 쓰면 검사기부터 늘린다.
- **2026 최종 검증은 한 번**: `config/params_v1.json`을 커밋한 뒤 메인이 실행한다. 에이전트에게 맡기지 않는다(PLAN §2-3·§4).
- **Workflow**(PLAN §4): 모듈마다 Red(테스트 작성) → Green(구현) → Refactor(리뷰) + 독립 검증 1명, 워크플로당 10명 이하. 에이전트는 픽스처로만 작업하고 실수집·배포는 메인이 한다. 레인마다 쓸 수 있는 파일을 프롬프트에 적는다.
- **외부 요청**: 요청 사이 1초 이상, User-Agent는 `contracts.USER_AGENT`(개인정보 없음). KBO 공식 기록실은 자동 수집하지 않는다(robots 금지, SPEC E12).
- **웹 화면**: `web/.claude/CLAUDE.md`(Astryx가 쓴 안내 — 템플릿에서 시작, `<div>` 대신 컴포넌트, 토큰 기반 스타일, 테마는 테마 편집기로)를 따른다. 기반 테마 neutral은 M0 빌드 확인용 임시값이고, 사이트 전용 테마는 DESIGN에서 정한다. 화면 구현은 시안 합의 뒤(M6).
- **이미지**: Codex(`codex exec`)로 만든다. 실존 선수·MLB/구단 상표·글자를 넣지 않고, 넣기 전에 직접 열어 확인한다(SPEC D19·N9). 실측 숫자 차트는 생성 이미지로 그리지 않는다.
- **표기**: 모든 공개 화면에 "재미로 보는 예측" 고지와 출처(MLB Stats API, Baseball Savant)를 단다(SPEC D16).
