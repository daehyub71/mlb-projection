"""수비 시즌 표 (SPEC D21·E4·E18, §6 `fielding_seasons`). 레인 C가 구현한다.

과거 표시 전용이다 — 예측에 쓰지 않고, FRV와 OAA를 더하지 않는다(D21).
원본은 `collect.savant.load_savant`로만 읽는다.
"""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

# Savant FRV CSV 열 → 정제 열
FRV_COLUMNS: Mapping[str, str] = {
    "total_runs": "frv_total",
    "range_runs": "frv_range",
    "arm_runs": "frv_arm",
    "dp_runs": "frv_dp",
    "framing_runs": "frv_framing",
    "blocking_runs": "frv_blocking",
    "throwing_runs": "frv_throwing",
}


def build_fielding(raw_dir: Path) -> pd.DataFrame:
    """수비 시즌 표를 만든다 (`contracts.FIELDING_SEASONS`, `check_frame` 통과, season·player_id 순).

    - 시즌: `contracts.FIELDING_YEARS`. league = "MLB".
    - FRV(`load_savant(..., "frv")`): `FRV_COLUMNS`대로 바꾸고, `outs_2`~`outs_9`는 정수(빈 칸은 0).
    - OAA(`load_savant(..., "oaa")`): oaa = outs_above_average.
    - (player_id, season)으로 **바깥 결합**한다. OAA에만 있는 선수는 FRV 열이 결측·아웃 수 0,
      FRV에만 있는 선수는 oaa가 결측.
    - 같은 (player_id, season)이 한 파일에 두 번 있으면 값이 같을 때만 하나로, 다르면 ValueError.
    """
    raise NotImplementedError
