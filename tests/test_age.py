"""시즌 나이 테스트 (SPEC F9, contracts.SEASON_AGE_MONTH_DAY = 6월 30일 기준)."""

import pytest

from mlbproj.age import season_age


@pytest.mark.parametrize(
    ("birth", "season", "expected"),
    [
        ("1994-07-05", 2027, 32),  # 7월 5일생 — 6월 30일엔 아직 32세
        ("1994-06-30", 2027, 33),  # 6월 30일생 — 그날 33세
        ("1994-01-01", 2027, 33),
        ("1994-12-31", 2027, 32),
        ("2000-02-29", 2027, 27),  # 윤일생
    ],
)
def test_age_season_age_june_30(birth: str, season: int, expected: int) -> None:
    assert season_age(birth, season) == expected


def test_age_season_age_missing_birth_date() -> None:
    assert season_age(None, 2027) is None


def test_age_season_age_rejects_bad_format() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        season_age("07/05/1994", 2027)
