"""HTTP 요청 — 간격·재시도·일반 UA (SPEC N3). 레인 A가 구현한다.

외부 패키지 없이 표준 라이브러리 `urllib`만 쓴다(SPEC D15). 실제 네트워크는 `urllib_transport`만
만지고, `Fetcher`는 주입받은 transport·sleep·clock으로 동작하므로 테스트는 가짜를 넣는다.
"""

import time
from collections.abc import Callable, Mapping

from mlbproj.contracts import MIN_INTERVAL_SECONDS, USER_AGENT

# 다시 시도하는 HTTP 상태
RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# transport(url, headers, timeout) → (HTTP 상태, 응답 바이트).
# 연결 실패·시간 초과처럼 응답 자체가 없으면 TransientError를 낸다.
Transport = Callable[[str, Mapping[str, str], float], tuple[int, bytes]]


class TransientError(Exception):
    """응답을 받지 못한 일시적 실패 (연결 실패·시간 초과). 재시도 대상이다."""


class FetchError(Exception):
    """재시도해도 받지 못했거나 재시도 대상이 아닌 실패.

    Attributes:
        url: 요청 URL.
        status: 마지막 HTTP 상태. 응답이 없었으면 None.
        attempts: 시도 횟수.
    """

    def __init__(self, url: str, status: int | None, attempts: int, message: str) -> None:
        super().__init__(message)
        self.url = url
        self.status = status
        self.attempts = attempts


def urllib_transport(url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
    """실제 요청을 보낸다 (live 테스트로만 확인한다).

    HTTPError는 예외 대신 (상태, 본문)으로 돌려주고, URLError·시간 초과·연결 끊김은
    TransientError로 바꾼다.

    Args:
        url: 요청 URL (https).
        headers: 요청 헤더.
        timeout: 초.

    Returns:
        (HTTP 상태, 응답 바이트).
    """
    raise NotImplementedError


class Fetcher:
    """요청 간격과 재시도를 지키는 GET 클라이언트.

    규칙
    - 모든 요청에 `User-Agent: USER_AGENT`를 붙인다.
    - 직전 요청 시작 시각에서 `min_interval`초가 지나지 않았으면 남은 만큼 `sleep`한 뒤 보낸다.
      재시도도 요청이므로 간격을 지킨다. 첫 요청은 기다리지 않는다.
    - 200이면 본문을 돌려준다. 그 밖의 2xx도 성공이다.
    - `RETRY_STATUSES` 상태나 TransientError면 `backoff_base ** (시도 번호 - 1)`초
      (1·2·4…)를 `sleep`한 뒤 다시 시도한다. 최대 `max_attempts`번 시도하고, 다 실패하면 FetchError.
    - 그 밖의 상태(400·403·404 등)는 바로 FetchError (재시도하지 않는다).
    - `requests_made`는 실제로 transport를 부른 횟수다.
    """

    def __init__(
        self,
        transport: Transport = urllib_transport,
        *,
        user_agent: str = USER_AGENT,
        min_interval: float = MIN_INTERVAL_SECONDS,
        max_attempts: int = 4,
        backoff_base: float = 2.0,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        raise NotImplementedError

    @property
    def requests_made(self) -> int:
        """transport를 부른 횟수."""
        raise NotImplementedError

    def get(self, url: str) -> bytes:
        """URL을 받아 응답 바이트를 돌려준다.

        Args:
            url: 요청 URL.

        Returns:
            응답 바이트.

        Raises:
            FetchError: 재시도해도 실패했거나 재시도 대상이 아닌 상태일 때.
        """
        raise NotImplementedError
