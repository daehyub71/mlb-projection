"""최종 검증(2026, 한 번)·2027 최종 예측·내보내기 입력 (SPEC F10·F11·N1, PLAN §2-3·§2-4). 레인 C가 구현한다.

최종 모델 (역할마다)
- 채택표에서 그 역할 지표(`EXT_ROLE_METRICS[role]`) 중 하나라도 True면 "extended", 아니면 "weighted".

평가 (`evaluate_target`, 가드 없음 — 가드는 명령 쪽)
- 역할마다 모델 "league"·"baseline"(`backtest.baseline_projections`), "weighted"
  (`models.extended.weighted_projection_for`), 그리고 최종 모델이 "extended"면 "extended"
  (`models.extended.extended_projection`)를 합쳐 `evaluate.evaluate`로 평가한다.
- 결과: 두 역할을 이은 `EVALUATIONS` 표, 키 순(안정 정렬).

최종 검증 (`run_final_test`) — PLAN §2-3 "파라미터 고정 절차"
1. `committed_head([params_path, adoption_path])`: 두 파일이 git에 추적되고 작업 트리·스테이징이 HEAD와
   같아야 한다. 아니면 FinalTestError(무엇이 문제인지 한국어로). 통과하면 HEAD 커밋 해시.
2. out_path가 이미 있고 rerun_reason이 없으면 FinalTestError("이미 한 번 돌렸다 … --rerun 사유").
3. `evaluate_target(…, FINAL_TEST_TARGET)`.
4. 기록(JSON, `export.dump_json`과 같은 직렬화 규칙이되 사람이 읽게 indent=1):
   {"target_season": 2026, "ran_at": 첫 실행 시각, "params_sha256", "adoption_sha256", "commit",
    "reruns": [{"ran_at", "reason", "commit"}…], "rows": [EVALUATIONS 행…(NaN → null)]}.
   다시 돌리면 ran_at은 첫 기록 값을 그대로 두고 reruns 끝에 이번 실행을 붙이고 rows를 새 결과로 바꾼다.
   반환은 이 기록(dict).

2027 예측 (`final_projections`): 역할마다 최종 모델의 목표 Y 예측 표 (그 모델 행만).

내보내기 입력 (`build_export_input`)
- evaluations = backtest_csv(`EVALUATIONS`, 2024·2025) 행 + 최종 검증 기록이 있으면 그 rows(2026).
- final_test = 기록이 있으면 {"target_season", "ran_at", "params_sha256", "reruns": [{"ran_at", "reason"}]}
  (스키마 ValidationReport.final_test 모양), 없으면 None.
- meta: schema_version = `WEB_SCHEMA_VERSION`, raw_collected_at = `raw_collected_at(raw_dir)`,
  generated_at = raw_collected_at(같은 원본이면 같은 바이트 — N1), params_version·params_sha256,
  target_season = Y, input_seasons = [Y−5 … Y−1], sources = `WEB_SOURCES`({name, url}), disclaimer = `WEB_DISCLAIMER`.

원본 수집 시각 (`raw_collected_at`): 원본 매니페스트(`collect.raw.load_manifest`) 항목의 받은 시각 중
가장 늦은 값, UTC "YYYY-MM-DDTHH:MM:SSZ". 매니페스트가 비었으면 ValueError.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from mlbproj.contracts import TARGET_SEASON, ModelName, Role
from mlbproj.export import ExportInput
from mlbproj.models.params import Adoption, Params
from mlbproj.tables import Tables

# git 인자 목록 → 표준 출력. 실패하면 예외 (테스트는 가짜를 넣는다)
GitRunner = Callable[[Sequence[str]], str]


class FinalTestError(RuntimeError):
    """최종 검증을 돌릴 수 없는 상태 (커밋 안 됨, 이미 돌림 등)."""


def final_model_name(adoption: Adoption, role: Role) -> ModelName:
    """역할의 최종 모델 이름 (모듈 docstring "최종 모델")."""
    raise NotImplementedError


def committed_head(paths: Sequence[Path], repo: Path, git: GitRunner | None = None) -> str:
    """paths가 모두 커밋된 그대로면 HEAD 커밋 해시. git=None이면 `git -C repo …`를 실제로 부른다.

    Raises:
        FinalTestError: 추적되지 않거나, 작업 트리·스테이징에 변경이 있거나, git을 부를 수 없을 때.
    """
    raise NotImplementedError


def evaluate_target(
    tables: Tables, params: Params, adoption: Adoption, target: int
) -> pd.DataFrame:
    """목표 시즌 하나의 평가표 (모듈 docstring "평가")."""
    raise NotImplementedError


def run_final_test(
    tables: Tables,
    params: Params,
    adoption: Adoption,
    *,
    repo: Path,
    params_path: Path,
    adoption_path: Path,
    out_path: Path,
    now: str,
    rerun_reason: str | None = None,
    git: GitRunner | None = None,
) -> dict[str, Any]:
    """2026 최종 검증을 한 번 돌리고 기록한다 (모듈 docstring "최종 검증").

    Args:
        now: 실행 시각 (UTC "YYYY-MM-DDTHH:MM:SSZ").
        rerun_reason: 다시 돌리는 사유 (빈 문자열은 사유 없음으로 본다).

    Raises:
        FinalTestError: 커밋 확인 실패, 또는 이미 돌렸는데 사유가 없을 때.
    """
    raise NotImplementedError


def final_projections(
    tables: Tables, params: Params, adoption: Adoption, target: int = TARGET_SEASON
) -> dict[Role, pd.DataFrame]:
    """역할마다 최종 모델의 목표 Y 예측 표."""
    raise NotImplementedError


def raw_collected_at(raw_dir: Path) -> str:
    """원본 수집 시각 (모듈 docstring "원본 수집 시각")."""
    raise NotImplementedError


def build_export_input(
    tables: Tables,
    params: Params,
    adoption: Adoption,
    projections: dict[Role, pd.DataFrame],
    *,
    raw_dir: Path,
    backtest_csv: Path,
    final_test_path: Path,
) -> ExportInput:
    """내보내기 입력 한 묶음 (모듈 docstring "내보내기 입력")."""
    raise NotImplementedError
