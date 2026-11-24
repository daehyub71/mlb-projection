"""HTTP 요청 — 간격·재시도·일반 UA (SPEC N3). 레인 A가 구현한다.

외부 패키지 없이 표준 라이브러리 `urllib`만 쓴다(SPEC D15). 실제 네트워크는 `urllib_transport`만
만지고, `Fetcher`는 주입받은 transport·sleep·clock으로 동작하므로 테스트는 가짜를 넣는다.
"""

import http.client
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

from mlbproj.contracts import MIN_INTERVAL_SECONDS, USER_AGENT

# 다시 시도하는 HTTP 상태
RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# transport(url, headers, timeout) → (HTTP 상태, 응답 바이트).
# 연결 실패·시간 초과처럼 응답 자체가 없으면 TransientError를 낸다.
Transport = Callable[[str, Mapping[str, str], float], tuple[int, bytes]]

# 응답을 받지 못한 실패 — 모두 TransientError로 바꿔 재시도한다.
# - OSError: URLError(연결 실패·DNS·인증서), TimeoutError, ConnectionError(RemoteDisconnected·
#   ConnectionResetError 포함), ssl.SSLError(SSLEOFError 등)
# - http.client.HTTPException: IncompleteRead(본문을 읽다 끊김), BadStatusLine·LineTooLong(상태 줄·헤더가 깨짐)
# urlopen은 요청을 보내는 단계의 OSError만 URLError로 감싸고, 상태 줄을 읽는 단계(getresponse)와
# 본문을 읽는 단계(read)의 오류는 그대로 내보낸다. 그래서 URLError만이 아니라 두 계열을 다 잡는다.
# HTTPError도 OSError(URLError)의 하위 클래스이므로 먼저 따로 잡아야 한다.
# 테스트의 네트워크 차단(RuntimeError)이나 잘못된 URL(ValueError)은 여기에 들지 않아 그대로 올라간다.
_NO_RESPONSE_ERRORS: tuple[type[Exception], ...] = (OSError, http.client.HTTPException)


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
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        try:
            # 테스트가 모듈 속성을 바꿔 끼우므로 부를 때마다 urllib.request에서 찾는다
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                status: int = resp.status
                body: bytes = resp.read()
                return status, body
        except urllib.error.HTTPError as err:
            # 4xx·5xx도 응답이다. 판단은 Fetcher가 한다
            with err:
                return err.code, _read_error_body(err)
    except _NO_RESPONSE_ERRORS as err:
        raise TransientError(f"{url}: {err}") from err


def _read_error_body(err: urllib.error.HTTPError) -> bytes:
    """오류 응답의 본문을 읽는다. 읽다가 끊기면 빈 바이트를 돌려준다.

    상태 줄은 이미 받았으므로 '응답이 없는 실패'(TransientError)가 아니다. Fetcher는 2xx가 아닌
    응답의 본문을 쓰지 않으므로, 본문을 잃어도 상태(예: 404면 바로 실패)는 그대로 전한다.
    """
    try:
        return err.read()
    except _NO_RESPONSE_ERRORS:
        return b""


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
        if max_attempts < 1:
            raise ValueError(f"max_attempts는 1 이상이어야 한다: {max_attempts}")
        self._transport = transport
        self._headers: dict[str, str] = {"User-Agent": user_agent}
        self._min_interval = min_interval
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._timeout = timeout
        self._sleep = sleep
        self._clock = clock
        # 직전 transport 호출이 시작한 시각. None이면 아직 요청한 적이 없다
        # (시계가 0.0에서 시작할 수 있으므로 0.0으로 초기화하지 않는다)
        self._last_start: float | None = None
        self._requests_made = 0

    @property
    def requests_made(self) -> int:
        """transport를 부른 횟수."""
        return self._requests_made

    def _send(self, url: str) -> tuple[int, bytes]:
        """요청 간격을 지킨 뒤 transport를 한 번 부른다 (재시도도 이 길로 나간다)."""
        if self._last_start is not None:
            remaining = self._last_start + self._min_interval - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last_start = self._clock()
        self._requests_made += 1
        return self._transport(url, self._headers, self._timeout)

    def get(self, url: str) -> bytes:
        """URL을 받아 응답 바이트를 돌려준다.

        Args:
            url: 요청 URL.

        Returns:
            응답 바이트.

        Raises:
            FetchError: 재시도해도 실패했거나 재시도 대상이 아닌 상태일 때.
        """
        # 마지막으로 받은 HTTP 상태. 한 번도 응답을 받지 못했으면 None으로 남는다
        status: int | None = None
        last_problem = ""
        # 마지막 시도가 응답 없이 끝났을 때의 원인 (FetchError의 __cause__로 잇는다)
        last_error: TransientError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                status, body = self._send(url)
            except TransientError as err:
                last_problem = f"응답 없음 ({err})"
                last_error = err
            else:
                last_error = None
                if 200 <= status < 300:
                    return body
                if status not in RETRY_STATUSES:
                    raise FetchError(
                        url, status, attempt, f"{url}: HTTP {status} (재시도 대상이 아니다)"
                    )
                last_problem = f"HTTP {status}"
            # 마지막 시도가 실패한 뒤에는 기다리지 않는다
            if attempt < self._max_attempts:
                self._sleep(self._backoff_base ** (attempt - 1))
        raise FetchError(
            url,
            status,
            self._max_attempts,
            f"{url}: {self._max_attempts}번 시도했지만 받지 못했다 (마지막: {last_problem})",
        ) from last_error
