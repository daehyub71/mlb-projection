"""웹 JSON·검토용 CSV 내보내기 (SPEC F11·F13~F15·D14·D16·D21·D23·N1·N6). 레인 B가 구현한다.

웹과의 계약은 `contracts/web-data.schema.json` 하나다. 파일: batters.json(BatTable), pitchers.json(PitTable),
validation.json(ValidationReport), players/{player_id}.json(PlayerDetail). 여기서는 모델 계산을 하지 않는다 —
입력 표의 값을 모양에 맞추고, 계약에 적힌 간단한 식(비율·FIP·평균)과 반올림만 한다.

공통 규칙
- 목표 시즌 Y = meta["target_season"](2027). Y−1 = 마지막 입력 시즌.
- 반올림: 키 이름이 `contracts.WEB_ROUND`에 있으면 그 자릿수로 `round()`. 정수 열은 int 그대로.
  결측(NaN·None)은 null. NaN·inf가 JSON에 들어가면 안 된다.
- 행 순서: player_id 오름차순. 검증 행은 (target_season, role, model, metric, group) 순.
- 직렬화(`dump_json`): `json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\\n"`.
  같은 입력이면 같은 바이트(N1).

예측표 행 (BatRow·PitRow, 그 역할의 Y 대상 전원 = inp.projections[role]의 선수)
- player(PlayerSummary): name·active·last_mlb_season은 players 표. age = `age.season_age(birth_date, Y)`,
  생년월일이 없으면 그 역할 정제 표의 input_last 행 `age` + (Y − input_last), 그것도 없으면 null.
  team_name = 그 역할 정제 표의 input_last 행 team_name.
- window: 대상 표(cohorts, Y)의 input_first·input_last·n_seasons.
- 타자: sample_pa = 대상 표 sample. projection = 예측 표의 pa·hr·obp·slg·ops.
  prev = Y−1 타격 행이 있으면 {season, pa, hr, ops}, 없으면 null.
- 투수: sample_bf = 대상 표 sample, sample_ip = sample_outs / 3. projection = ip·era·k_bb_pct.
  prev = Y−1 투구 행이 있으면 {season, ip = outs/3, era = 27·er/outs, k_bb_pct = 100·(so − bb)/bf}
  (분모 0이면 그 값 null), 없으면 null.
- league_baseline: 타자 {"ops": 리그 Y−1 ops}, 투수 {"era": 리그 Y−1 era, "k_bb_pct": 리그 Y−1 k_bb_pct}
  (tables.league).
- 표 객체: {meta, role, model = inp.model[role], league_baseline, rows}.

선수 상세 (PlayerDetail, 두 역할 Y 대상의 합집합마다 한 파일)
- player: players 표의 player_id·name·birth_date·primary_position·bats·throws·active·last_mlb_season.
- bat / pit: 그 역할 Y 대상이 아니면 null(F12). 대상이면 BatDetail/PitDetail —
  window·sample·projection·prev·league_baseline은 예측표 행과 같은 값.
  history: 그 역할 정제 표에서 input_first ~ input_last 안의 행(최대 5개, season 오름차순).
    타자 BatSeason: k_pct = 100·so/pa, bb_pct = 100·bb/pa (pa 0이면 null), 나머지는 표 값.
    투수 PitSeason: ip = outs/3, era = 27·er/outs, fip = (13·hr + 3·(bb + hbp) − 2·so) / (outs/3) + 그 시즌
    리그 fip_const, k_pct = 100·so/bf, bb_pct = 100·bb/bf, k_bb_pct = k_pct − bb_pct (분모 0이면 null).
  group_error: group = n_seasons ≤ 2이면 "seasons_1_2", 아니면 "seasons_3_5".
    mae[지표] = evaluations에서 (role, model = inp.model[role], metric, group) 행의 mae를 목표 시즌들에 걸쳐
    단순 평균한 값(결측은 빼고 평균, 행이 없으면 null). 지표 = `BAT_METRICS`/`PIT_METRICS`.
- fielding: 그 선수의 수비 표 행(`FIELDING_YEARS` 안, 최근 5개, season 오름차순). outs_by_pos =
  {`POSITION_ABBR[p]`: outs_p} 중 outs_p > 0인 포지션만. 수비 행이 없으면 [].

검증 (ValidationReport): {meta, final_test = inp.final_test, rows = evaluations 전부(EvalRow)}.

쓰기 (`write_web`)
- 파일마다 스키마 정의(`WEB_FILES`, 선수 상세는 `WEB_PLAYER_DEF`)로 `schema.validate` 검사. 위반이 있으면
  아무것도 쓰지 않고 ValueError(파일 이름과 처음 몇 개 위반).
- `WEB_FIRST_SCREEN_FILES`가 각각 `WEB_FIRST_SCREEN_MAX_BYTES`를 넘으면 쓰기 전에 ValueError(N6).
- out_dir/players/ 안에 이번 묶음에 없는 .json 파일은 지운다(은퇴 등으로 빠진 선수). 다른 파일은 건드리지 않는다.

검토용 CSV (`write_projection_csv`): 역할마다 예측 표 열(`BAT_PROJECTIONS`·`PIT_PROJECTIONS` 순) 뒤에
name 열을 붙여 `PROJECTION_CSV[role]`에 쓴다. player_id 순, 줄 끝 "\\n", 실수는 pandas 기본 표기.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mlbproj.contracts import PROJECTION_CSV, Role
from mlbproj.tables import Tables


@dataclass(frozen=True)
class ExportInput:
    """내보내기 입력 한 묶음.

    Attributes:
        tables: 정제 표 묶음 (players·seasons·fielding·league·cohorts. cohorts에 목표 Y 행이 있어야 한다).
        projections: 역할 → 목표 Y의 최종 모델 예측 표 (`BAT_PROJECTIONS`·`PIT_PROJECTIONS`, 그 모델 행만).
        model: 역할 → 최종 모델 이름 ("extended" 또는 "weighted").
        evaluations: 과거 검증 행 (`EVALUATIONS`, 2024·2025와 최종 검증 뒤 2026).
        final_test: ValidationReport.final_test 객체. 최종 검증 전이면 None.
        meta: Meta 객체 전체 (schema_version 포함, 스키마 그대로).
    """

    tables: Tables
    projections: dict[Role, pd.DataFrame]
    model: dict[Role, str]
    evaluations: pd.DataFrame
    final_test: dict[str, Any] | None
    meta: dict[str, Any]


def build_web(inp: ExportInput) -> dict[str, Any]:
    """웹 JSON 묶음. 상대 경로("batters.json", "players/660271.json" …) → JSON 객체.

    Raises:
        ValueError: 예측 표의 선수가 목표 Y 대상 표에 없거나, players 표에 없을 때.
    """
    raise NotImplementedError


def dump_json(obj: Any) -> str:
    """결정적 직렬화 (모듈 docstring "직렬화")."""
    raise NotImplementedError


def write_web(files: Mapping[str, Any], out_dir: Path, schema: dict[str, Any]) -> dict[str, int]:
    """스키마 검사·크기 검사 뒤 파일을 쓴다. 반환: 상대 경로 → 바이트 수 (경로 순).

    Raises:
        ValueError: 스키마 위반, 첫 화면 크기 초과, 모르는 파일 이름일 때 (아무것도 쓰지 않는다).
    """
    raise NotImplementedError


def write_projection_csv(
    inp: ExportInput, paths: Mapping[Role, Path] = PROJECTION_CSV
) -> dict[Role, Path]:
    """검토용 2027 예측 CSV를 쓴다 (모듈 docstring "검토용 CSV")."""
    raise NotImplementedError
