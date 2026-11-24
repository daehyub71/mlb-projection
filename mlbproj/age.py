"""시즌 나이 (SPEC F9). 확장 모델 특성과 웹 표시가 같은 규칙을 쓴다."""

import re

from mlbproj.contracts import SEASON_AGE_MONTH_DAY

_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def season_age(birth_date: str | None, season: int) -> int | None:
    """그 시즌 6월 30일 기준 만 나이 (`contracts.SEASON_AGE_MONTH_DAY`).

    생년월일은 시즌과 무관한 고정값이라 과거 목표에 써도 미래 정보가 섞이지 않는다(N2).

    Args:
        birth_date: `YYYY-MM-DD` 또는 None.
        season: 시즌 연도.

    Returns:
        만 나이. 생년월일이 없으면 None.

    Raises:
        ValueError: 형식이 `YYYY-MM-DD`가 아닐 때.
    """
    if birth_date is None:
        return None
    match = _ISO_DATE.fullmatch(birth_date)
    if match is None:
        raise ValueError(f"생년월일은 YYYY-MM-DD 형식이어야 한다: {birth_date!r}")
    year, month, day = (int(part) for part in match.groups())
    before_birthday = (month, day) > SEASON_AGE_MONTH_DAY
    return season - year - (1 if before_birthday else 0)
