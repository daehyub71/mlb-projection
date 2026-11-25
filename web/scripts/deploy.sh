#!/usr/bin/env bash
# 배포 (SPEC D32) — 로컬에서 만든 정적 결과물(web/out)을 Vercel에 직접 올린다.
# 웹 JSON(public/data)은 어느 저장소에도 넣지 않는다. 데이터 갱신은 수동이다(D25).
#
# 사용 (저장소 루트에서):
#   venv/bin/python -m mlbproj export          # 2027 예측 → web/public/data
#   VERCEL_TOKEN=... web/scripts/deploy.sh     # 검사 → 빌드 → 업로드 (토큰 값은 출력하지 않는다)
set -euo pipefail

PROJECT="mlb-projection"
cd "$(dirname "$0")/.."

if [ -z "${VERCEL_TOKEN:-}" ]; then
  echo "VERCEL_TOKEN이 필요합니다 (값은 출력하지 않는다)" >&2
  exit 1
fi
if [ ! -f public/data/batters.json ]; then
  echo "public/data가 없습니다 — 저장소 루트에서 먼저 venv/bin/python -m mlbproj export" >&2
  exit 1
fi

npm run lint
npm test
npm run build
npx astryx doctor --built "$PWD"

# 보안 헤더·주소 규칙 (web/vercel.json)을 올리는 폴더 맨 위에 둔다
cp vercel.json out/vercel.json

cd out
vercel link --yes --project "$PROJECT" --token "$VERCEL_TOKEN" >/dev/null
vercel deploy --prod --yes --token "$VERCEL_TOKEN"
