# 테스트 픽스처 출처 (PLAN P4)

`tests/fixtures/raw/`는 실제 응답에서 **선수 14명만 잘라 낸** 것이다. 파일 하나에 들어가는 선수는 12명 이하이고, `tests/test_repo_hygiene.py`가 이 상한을 검사한다. 테스트 전용이며 화면이나 산출물에 쓰지 않는다.

MLB 자료 고지(SPEC E13): "Only individual, non-commercial, non-bulk use of the Materials is permitted". 원본 전체(`data/raw/`)는 저장소에 올리지 않는다(SPEC D6). API 응답의 `copyright` 문구는 그대로 남겼다.

## 받은 곳과 시각

| 파일 | 요청 | 받은 시각 |
|---|---|---|
| `raw/statsapi/{season}_{bat\|pit}.json` (2019~2026) | `contracts.statsapi_stats_url(season, role)` | 2026-11-23 21:56~21:58 KST |
| `raw/statsapi/people_000.json`(타자 8명)·`people_001.json`(투수 6명) | `contracts.statsapi_people_url(ids)` 1회 | 2026-11-24 08:09 KST |
| `raw/savant/{season}_{batter\|pitcher}.csv` (2019~2026) | `contracts.savant_leaderboard_url(season, role)` | 2026-11-24 08:06~08:08 KST |
| `raw/savant/{season}_{oaa\|frv}.csv` (2022~2026) | `contracts.savant_oaa_url`·`savant_frv_url` | 2026-11-24 08:06~08:08 KST |

## 잘라 낸 방법

- **API JSON**: `stats[0].splits`에서 아래 선수의 행만 남겼다. `totalSplits`는 남긴 행 수로 바꿨다. 나머지 필드는 원형 그대로다.
- **Savant CSV**: 첫 줄(BOM 포함 머리글)과 아래 선수의 줄을 **원문 그대로** 남겼다. 값의 따옴표 표기(`".274"` 등)도 원본과 같다.
- **선수 정보**: 14명을 한 번에 받아 타자·투수 두 파일로 나눴다(파일당 12명 이하).
- 잘라 낸 뒤에도 F2 ③(타자 Savant = API PA>0)과 ④(Savant 투수 ⊆ API 투수)가 모든 시즌에서 성립하는 것을 확인했다.

## 선수와 고른 이유 (PLAN §5 사례)

| ID | 선수 | 역할 | 사례 |
|---|---|---|---|
| 646240 | Rafael Devers | 타자 | 8시즌 모두 출전, 2025 시즌 중 이적(합산 1행, `numTeams`=2) |
| 592663 | J.T. Realmuto | 타자 | 8시즌 모두 출전한 포수 — FRV 포수 세부(D21) |
| 701538 | Jackson Merrill | 타자 | 2024 데뷔 |
| 805808 | Kevin McGonigle | 타자 | 2026 데뷔 |
| 650391 | Eloy Jiménez | 타자 | 2025 결장 후 2026 복귀, 2024 이적 |
| 408234 | Miguel Cabrera | 타자 | 2023 이후 미출전(은퇴), `active`=false(D23) |
| 660271 | Shohei Ohtani | 타자·투수 | 투타 겸업(E7) — 타격 포지션 DH, 투구 포지션 `TWP`, 2024 투구 없음 |
| 571771 | Enrique Hernández | 타자 | 등판한 야수(2024~2026 투구 행 포지션 3B). 2025는 Savant 투수 목록에 있고 2024·2026은 없다(E2), 2023 이적 |
| 605400 | Aaron Nola | 투수 | 8시즌 모두 선발, 2019·2021 타석(E10), 2020 시즌 |
| 628452 | Raisel Iglesias | 투수 | 마무리(세이브), 2022 이적 |
| 694973 | Paul Skenes | 투수 | 2024 데뷔 |
| 543037 | Gerrit Cole | 투수 | 2025 결장 후 2026 복귀 |
| 425794 | Adam Wainwright | 투수 | 2023 이후 미출전(은퇴), 2019·2021 타석 |
| 519381 | Pat Venditte | 투수 | 양손 투수 — Savant 2019·2020에 같은 값 2행(E19), 마지막 출전 2020, 2019 타격 행 0 PA |

## 들어 있지 않은 사례

- 타격 포지션 `X`(2019~2026 247행, 대부분 이적 합산 행) — 역할 판정(M2)에서 필요하면 추가한다.
- 리그 합계 검사(F2 ⑤)는 잘라 낸 파일로 검사할 수 없다. 범위 값을 인자로 받아 `synthetic/`에서 검사한다.
- 실제로 생기지 않는 경계 사례(K=0, PA=0인 타자 행 등)는 `synthetic/`에 따로 만든다.
