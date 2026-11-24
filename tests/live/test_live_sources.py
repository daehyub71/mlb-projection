"""외부 출처 형식 확인 (PLAN §5 live 층). 기본 실행에서 빠지고 `venv/bin/pytest -m live`로 돈다.

출처마다 한 번씩만 요청해 필수 열·필드가 아직 있는지만 본다. 요청 사이 간격은 Fetcher가 지킨다.
"""

import pytest

from mlbproj import contracts as c
from mlbproj.collect import savant, statsapi
from mlbproj.collect.http import Fetcher

pytestmark = pytest.mark.live

SEASON = c.SEASON_LAST
OHTANI = 660271


@pytest.fixture(scope="module")
def fetcher() -> Fetcher:
    return Fetcher()


@pytest.mark.parametrize("role", c.ROLES)
def test_live_statsapi_season_stats(fetcher: Fetcher, role: c.Role) -> None:
    body = fetcher.get(c.statsapi_stats_url(SEASON, role))
    statsapi.check_stats_body(body)


def test_live_statsapi_people(fetcher: Fetcher) -> None:
    body = fetcher.get(c.statsapi_people_url([OHTANI]))
    statsapi.check_people_body(body, [OHTANI])


@pytest.mark.parametrize("role", c.ROLES)
def test_live_savant_leaderboard(fetcher: Fetcher, role: c.Role) -> None:
    kind: c.SavantKind = "batter" if role == "bat" else "pitcher"
    savant.check_csv_body(fetcher.get(c.savant_leaderboard_url(SEASON, role)), kind)


def test_live_savant_oaa(fetcher: Fetcher) -> None:
    savant.check_csv_body(fetcher.get(c.savant_oaa_url(SEASON)), "oaa")


def test_live_savant_frv(fetcher: Fetcher) -> None:
    savant.check_csv_body(fetcher.get(c.savant_frv_url(SEASON)), "frv")
