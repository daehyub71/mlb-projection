"""HTTP 요청(collect/http.py) 테스트 — 간격·재시도·일반 UA (SPEC N3).

실제 네트워크 없이 가짜 transport·시계·sleep으로 돈다. 가짜 sleep은 기록만 하고 가짜 시계를
그만큼 앞으로 민다. `urllib_transport`는 `urllib.request.urlopen`을 가짜로 바꿔 확인한다.
"""

from __future__ import annotations

import email.message
import http.client
import io
import itertools
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mlbproj.collect import raw
from mlbproj.collect.http import Fetcher, FetchError, TransientError, urllib_transport
from mlbproj.contracts import MIN_INTERVAL_SECONDS, USER_AGENT

URL = "https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season=2026"
URL2 = "https://baseballsavant.mlb.com/leaderboard/custom?year=2026&csv=true"
OK = (200, b"ok")

# 가짜 transport가 차례로 돌려줄 것: (상태, 본문) 또는 낼 예외
Response = tuple[int, bytes] | BaseException


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """주입한 sleep 대신 진짜 time.sleep을 부르면 바로 실패시킨다."""

    def guard(seconds: float) -> None:
        pytest.fail(f"진짜 time.sleep({seconds})을 불렀다 — 주입한 sleep을 써야 한다")

    monkeypatch.setattr(time, "sleep", guard)


class FakeClock:
    """가짜 시계. sleep은 기록하고 시계를 그만큼 민다."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def waits(self) -> list[float]:
        """0초 sleep(기다리지 않은 것과 같다)을 뺀 sleep 기록."""
        return [s for s in self.sleeps if s != 0]


@dataclass(frozen=True)
class Call:
    """transport 호출 하나."""

    url: str
    headers: dict[str, str]
    timeout: float
    at: float  # 호출 시작 시각 (가짜 시계)


class FakeTransport:
    """정해 둔 응답을 차례로 돌려주는 transport. 호출마다 시계를 latency만큼 민다."""

    def __init__(self, clock: FakeClock, responses: Sequence[Response], latency: float) -> None:
        self.clock = clock
        self.responses = list(responses)
        self.latency = latency
        self.calls: list[Call] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
        self.calls.append(Call(url, dict(headers), timeout, self.clock.now))
        if not self.responses:
            pytest.fail(f"정해 둔 응답보다 요청이 많다: {url}")
        response = self.responses.pop(0)
        self.clock.now += self.latency
        if isinstance(response, BaseException):
            raise response
        return response

    @property
    def times(self) -> list[float]:
        return [c.at for c in self.calls]


def make(
    responses: Sequence[Response],
    *,
    latency: float = 0.0,
    start: float = 1000.0,
    **kwargs: Any,
) -> tuple[Fetcher, FakeTransport, FakeClock]:
    """가짜 transport·시계·sleep을 넣은 Fetcher를 만든다."""
    clock = FakeClock(start)
    transport = FakeTransport(clock, responses, latency)
    fetcher = Fetcher(transport, sleep=clock.sleep, clock=clock.clock, **kwargs)
    return fetcher, transport, clock


def header(headers: Mapping[str, str], name: str) -> str | None:
    """헤더 값을 대소문자 구분 없이 찾는다 (같은 이름이 둘이면 실패)."""
    found = [v for k, v in headers.items() if k.lower() == name.lower()]
    assert len(found) <= 1, f"{name} 헤더가 여러 개다: {headers}"
    return found[0] if found else None


# ---------------------------------------------------------------------------
# User-Agent·timeout·요청 수
# ---------------------------------------------------------------------------


def test_http_fetcher_sends_user_agent_on_every_request() -> None:
    fetcher, transport, _ = make([OK, (503, b""), TransientError("reset"), OK, OK])
    assert fetcher.get(URL) == b"ok"
    assert fetcher.get(URL2) == b"ok"  # 503·연결 실패 뒤 성공 — 재시도에도 UA가 붙는다
    assert fetcher.get(URL) == b"ok"
    assert [c.url for c in transport.calls] == [URL, URL2, URL2, URL2, URL]
    for call in transport.calls:
        assert header(call.headers, "User-Agent") == USER_AGENT


def test_http_fetcher_custom_user_agent() -> None:
    fetcher, transport, _ = make([OK], user_agent="probe/9")
    fetcher.get(URL)
    assert header(transport.calls[0].headers, "User-Agent") == "probe/9"


def test_http_fetcher_passes_timeout_to_transport() -> None:
    fetcher, transport, _ = make([OK])
    fetcher.get(URL)
    assert transport.calls[0].timeout == 60.0
    custom, custom_transport, _ = make([(503, b""), OK], timeout=5.0)
    custom.get(URL)
    assert [c.timeout for c in custom_transport.calls] == [5.0, 5.0]


def test_http_fetcher_requests_made_counts_transport_calls() -> None:
    fetcher, _, _ = make([TransientError("timeout"), (503, b""), OK, (404, b""), OK])
    assert fetcher.requests_made == 0
    fetcher.get(URL)  # 예외를 낸 호출도 transport를 부른 것이다
    assert fetcher.requests_made == 3
    with pytest.raises(FetchError):
        fetcher.get(URL)
    assert fetcher.requests_made == 4
    fetcher.get(URL)
    assert fetcher.requests_made == 5


def test_http_fetcher_works_as_raw_getter(tmp_path: Path) -> None:
    fetcher, transport, _ = make([(200, b'{"a": 1}')])
    rel = Path("statsapi/2026_bat.json")
    outcome = raw.fetch_to_raw(fetcher, tmp_path, rel, URL)
    assert outcome.fetched
    assert (tmp_path / rel).read_bytes() == b'{"a": 1}'
    assert len(transport.calls) == 1


# ---------------------------------------------------------------------------
# 요청 간격 (N3: 1초 이상, 직전 요청 시작 시각 기준)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start", [0.0, 1000.0])
def test_http_fetcher_first_request_does_not_wait(start: float) -> None:
    fetcher, transport, clock = make([OK], start=start)
    fetcher.get(URL)
    assert clock.waits() == []
    assert transport.times == [start]


def test_http_fetcher_default_interval_is_at_least_one_second() -> None:
    assert MIN_INTERVAL_SECONDS >= 1.0
    fetcher, transport, clock = make([OK, OK, OK])
    for _ in range(3):
        fetcher.get(URL)
    assert clock.waits() == [MIN_INTERVAL_SECONDS, MIN_INTERVAL_SECONDS]
    assert transport.times == [
        1000.0,
        1000.0 + MIN_INTERVAL_SECONDS,
        1000.0 + 2 * MIN_INTERVAL_SECONDS,
    ]


def test_http_fetcher_waits_remaining_interval_from_previous_start() -> None:
    # 요청 하나에 0.25초 걸린다 → 다음 요청은 남은 0.75초만 기다린다 (끝 시각 기준이면 1초)
    fetcher, transport, clock = make([OK, OK, OK], latency=0.25)
    fetcher.get(URL)
    fetcher.get(URL)
    assert clock.waits() == [0.75]
    assert transport.times == [1000.0, 1001.0]
    clock.now += 0.5  # 부르는 쪽이 그 사이 다른 일을 했다 (지금 1001.75)
    fetcher.get(URL)
    assert clock.waits() == [0.75, 0.25]
    assert transport.times == [1000.0, 1001.0, 1002.0]


def test_http_fetcher_custom_min_interval() -> None:
    fetcher, transport, clock = make([OK, OK], latency=0.25, min_interval=2.5)
    fetcher.get(URL)
    fetcher.get(URL)
    assert clock.waits() == [2.25]
    assert transport.times == [1000.0, 1002.5]


def test_http_fetcher_no_wait_when_interval_already_passed() -> None:
    fetcher, transport, clock = make([OK, OK, OK], latency=1.5)
    fetcher.get(URL)
    fetcher.get(URL)  # 직전 요청이 1.5초 걸렸다
    clock.now += 3.0
    fetcher.get(URL)
    assert clock.waits() == []
    assert transport.times == [1000.0, 1001.5, 1006.0]


def test_http_fetcher_retries_obey_interval() -> None:
    # 간격 3초가 백오프(1·2초)보다 길다 → 재시도도 직전 시도 시작에서 3초 뒤에 나간다
    fetcher, transport, _ = make(
        [(503, b""), TransientError("reset"), OK, OK], latency=0.25, min_interval=3.0
    )
    assert fetcher.get(URL) == b"ok"
    fetcher.get(URL2)
    assert transport.times == [1000.0, 1003.0, 1006.0, 1009.0]


def test_http_fetcher_every_request_keeps_min_interval() -> None:
    responses: list[Response] = [
        OK,
        (503, b""),
        TransientError("timeout"),
        OK,
        (404, b""),
        OK,
        (429, b""),
        OK,
    ]
    fetcher, transport, _ = make(responses, latency=0.25, min_interval=1.5)
    fetcher.get(URL)
    fetcher.get(URL)
    with pytest.raises(FetchError):
        fetcher.get(URL)
    fetcher.get(URL)
    fetcher.get(URL)
    times = transport.times
    assert len(times) == 8
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert all(g >= 1.5 - 1e-9 for g in gaps), gaps


# ---------------------------------------------------------------------------
# 재시도·지수 백오프
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_http_fetcher_retries_retryable_status(status: int) -> None:
    fetcher, transport, clock = make([(status, b"busy"), OK])
    assert fetcher.get(URL) == b"ok"
    assert fetcher.requests_made == 2
    assert clock.waits() == [1.0]
    assert transport.responses == []


def test_http_fetcher_retries_transient_error() -> None:
    fetcher, _, clock = make([TransientError("reset"), TransientError("timeout"), OK])
    assert fetcher.get(URL) == b"ok"
    assert fetcher.requests_made == 3
    assert clock.waits() == [1.0, 2.0]


def test_http_fetcher_backoff_doubles_between_attempts() -> None:
    fetcher, _, clock = make([(503, b""), (503, b""), (503, b""), OK], latency=0.25)
    assert fetcher.get(URL) == b"ok"
    assert clock.waits() == [1.0, 2.0, 4.0]
    assert fetcher.requests_made == 4


def test_http_fetcher_backoff_base_is_configurable() -> None:
    fetcher, _, clock = make([(503, b"")] * 4, backoff_base=3.0)
    with pytest.raises(FetchError):
        fetcher.get(URL)
    # 3 ** 0, 3 ** 1, 3 ** 2 — 마지막 실패 뒤에는 기다리지 않는다
    assert clock.waits() == [1.0, 3.0, 9.0]


def test_http_fetcher_gives_up_after_max_attempts() -> None:
    fetcher, transport, clock = make([(503, b""), (502, b""), (500, b""), (429, b""), OK])
    with pytest.raises(FetchError) as excinfo:
        fetcher.get(URL)
    err = excinfo.value
    assert err.url == URL
    assert err.status == 429  # 마지막 시도의 상태
    assert err.attempts == 4
    assert fetcher.requests_made == 4
    assert clock.waits() == [1.0, 2.0, 4.0]
    assert transport.responses == [OK]  # 다섯 번째는 보내지 않는다


def test_http_fetcher_gives_up_without_response_has_no_status() -> None:
    fetcher, _, _ = make([TransientError("timeout")] * 4)
    with pytest.raises(FetchError) as excinfo:
        fetcher.get(URL2)
    assert excinfo.value.url == URL2
    assert excinfo.value.status is None
    assert excinfo.value.attempts == 4


@pytest.mark.parametrize(("max_attempts", "waits"), [(1, []), (2, [1.0]), (3, [1.0, 2.0])])
def test_http_fetcher_max_attempts_is_configurable(max_attempts: int, waits: list[float]) -> None:
    fetcher, transport, clock = make([(503, b"")] * 5, max_attempts=max_attempts)
    with pytest.raises(FetchError) as excinfo:
        fetcher.get(URL)
    assert excinfo.value.attempts == max_attempts
    assert excinfo.value.status == 503
    assert fetcher.requests_made == max_attempts
    assert clock.waits() == waits
    assert len(transport.responses) == 5 - max_attempts


@pytest.mark.parametrize("status", [301, 304, 400, 401, 403, 404, 410, 501])
def test_http_fetcher_non_retryable_status_fails_at_once(status: int) -> None:
    fetcher, transport, clock = make([(status, b"nope"), OK])
    with pytest.raises(FetchError) as excinfo:
        fetcher.get(URL)
    err = excinfo.value
    assert (err.url, err.status, err.attempts) == (URL, status, 1)
    assert fetcher.requests_made == 1
    assert clock.waits() == []
    assert transport.responses == [OK]  # 재시도하지 않는다


@pytest.mark.parametrize("status", [200, 201, 203, 204, 299])
def test_http_fetcher_any_2xx_returns_body(status: int) -> None:
    body = "﻿last_name, first_name\n".encode() if status != 204 else b""
    fetcher, _, clock = make([(status, body), OK])
    assert fetcher.get(URL) == body  # 받은 바이트 그대로
    assert fetcher.requests_made == 1
    assert clock.waits() == []


def test_http_fetcher_non_retryable_status_after_retry_stops() -> None:
    fetcher, transport, clock = make([TransientError("reset"), (404, b""), OK])
    with pytest.raises(FetchError) as excinfo:
        fetcher.get(URL)
    assert (excinfo.value.status, excinfo.value.attempts) == (404, 2)
    assert clock.waits() == [1.0]  # 404 뒤에는 기다리지도 다시 보내지도 않는다
    assert transport.responses == [OK]


def test_http_fetcher_status_is_last_received_http_status() -> None:
    # 마지막 시도가 응답 없이 끝나도 status는 그 전에 받은 마지막 HTTP 상태다.
    # None은 응답을 한 번도 받지 못했을 때뿐이다 (그 경우는 gives_up_without_response 테스트)
    fetcher, _, _ = make([(503, b""), (502, b""), TransientError("reset"), TransientError("t/o")])
    with pytest.raises(FetchError) as excinfo:
        fetcher.get(URL)
    assert excinfo.value.status == 502
    assert excinfo.value.attempts == 4
    assert "응답 없음" in str(excinfo.value)  # 메시지는 마지막 시도를 적는다


def test_http_fetcher_error_messages_name_url_and_reason() -> None:
    fetcher, _, _ = make([(404, b""), *[(503, b"")] * 4])
    with pytest.raises(FetchError) as rejected:
        fetcher.get(URL)
    assert URL in str(rejected.value)
    assert "HTTP 404" in str(rejected.value)
    with pytest.raises(FetchError) as exhausted:
        fetcher.get(URL2)
    message = str(exhausted.value)
    assert URL2 in message
    assert "4번" in message
    assert "HTTP 503" in message


def test_http_fetcher_chains_last_transient_error_as_cause() -> None:
    last = TransientError("timed out")
    fetcher, _, _ = make([(503, b""), TransientError("reset"), (503, b""), last])
    with pytest.raises(FetchError) as excinfo:
        fetcher.get(URL)
    assert excinfo.value.__cause__ is last
    # 마지막 시도가 HTTP 상태로 끝났으면 이을 원인이 없다
    fetcher, _, _ = make([TransientError("reset"), *[(503, b"")] * 3])
    with pytest.raises(FetchError) as excinfo:
        fetcher.get(URL)
    assert excinfo.value.__cause__ is None


def test_http_fetcher_unexpected_transport_error_is_not_swallowed() -> None:
    # TransientError가 아닌 예외는 버그다 — 재시도하거나 FetchError로 감추지 않는다
    fetcher, transport, clock = make([RuntimeError("bug in transport"), OK])
    with pytest.raises(RuntimeError, match="bug in transport"):
        fetcher.get(URL)
    assert fetcher.requests_made == 1
    assert clock.waits() == []
    assert transport.responses == [OK]


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_http_fetcher_rejects_max_attempts_below_one(max_attempts: int) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        make([OK], max_attempts=max_attempts)


# ---------------------------------------------------------------------------
# urllib_transport — urlopen을 가짜로 바꿔 네트워크 없이 확인한다
# ---------------------------------------------------------------------------


class FakeResponse:
    """urlopen이 돌려주는 응답 흉내 (with 문·read(n) 조각 읽기 지원)."""

    def __init__(self, status: int, body: bytes, read_error: BaseException | None = None) -> None:
        self.status = status
        self.code = status
        self.reason = "OK"
        self.headers = email.message.Message()
        self._stream = io.BytesIO(body)
        self._read_error = read_error
        self.closed = False

    def getcode(self) -> int:
        return self.status

    def info(self) -> email.message.Message:
        return self.headers

    def read(self, amt: int | None = None) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._stream.read(amt)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FakeUrlopen:
    """urllib.request.urlopen 대역. 받은 요청과 timeout을 기록한다."""

    def __init__(self, outcome: FakeResponse | BaseException) -> None:
        self.outcome = outcome
        self.requests: list[Any] = []
        self.timeouts: list[Any] = []

    def __call__(
        self, url: Any, data: Any = None, timeout: Any = None, **kwargs: Any
    ) -> FakeResponse:
        self.requests.append(url)
        self.timeouts.append(timeout)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def install(monkeypatch: pytest.MonkeyPatch, outcome: FakeResponse | BaseException) -> FakeUrlopen:
    fake = FakeUrlopen(outcome)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, code, "error", email.message.Message(), io.BytesIO(body))


def test_http_urllib_transport_sends_get_with_headers_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"x" * 200_000 + b"end"  # 조각으로 읽어도 끝까지 읽어야 한다
    fake = install(monkeypatch, FakeResponse(200, body))
    result = urllib_transport(URL, {"User-Agent": USER_AGENT, "X-Probe": "1"}, 7.5)
    assert result == (200, body)
    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == URL
    assert request.get_method() == "GET"
    assert request.get_header("User-agent") == USER_AGENT  # Request가 이름을 이렇게 바꿔 둔다
    assert request.get_header("X-probe") == "1"
    assert fake.timeouts == [7.5]
    assert isinstance(fake.outcome, FakeResponse)
    assert fake.outcome.closed  # 응답을 닫는다


def test_http_urllib_transport_returns_other_2xx_status(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeResponse(203, b"partial"))
    assert urllib_transport(URL, {"User-Agent": USER_AGENT}, 1.0) == (203, b"partial")


@pytest.mark.parametrize(
    ("code", "body"),
    [(404, b"not found"), (403, b""), (429, b"slow down"), (503, b"busy")],
)
def test_http_urllib_transport_http_error_becomes_status(
    monkeypatch: pytest.MonkeyPatch, code: int, body: bytes
) -> None:
    install(monkeypatch, http_error(code, body))
    assert urllib_transport(URL, {"User-Agent": USER_AGENT}, 1.0) == (code, body)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(urllib.error.URLError("connection refused"), id="url-error"),
        pytest.param(urllib.error.URLError(TimeoutError("timed out")), id="url-error-timeout"),
        pytest.param(TimeoutError("timed out"), id="timeout"),
        pytest.param(ConnectionResetError("reset by peer"), id="connection-reset"),
        pytest.param(http.client.RemoteDisconnected("closed"), id="remote-disconnected"),
        # urlopen은 상태 줄을 읽는 단계(getresponse)의 오류를 URLError로 감싸지 않는다
        pytest.param(ssl.SSLEOFError("EOF in violation of protocol"), id="ssl-eof"),
        pytest.param(http.client.BadStatusLine("garbage"), id="bad-status-line"),
        pytest.param(http.client.LineTooLong("header line"), id="line-too-long"),
    ],
)
def test_http_urllib_transport_no_response_is_transient(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    install(monkeypatch, error)
    with pytest.raises(TransientError) as excinfo:
        urllib_transport(URL, {"User-Agent": USER_AGENT}, 1.0)
    assert excinfo.value.__cause__ is error  # 원인을 잃지 않는다


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(TimeoutError("read timed out"), id="timeout"),
        pytest.param(ConnectionResetError("reset by peer"), id="connection-reset"),
        pytest.param(http.client.IncompleteRead(b"par", 10), id="incomplete-read"),
        pytest.param(ssl.SSLError("record layer failure"), id="ssl-error"),
    ],
)
def test_http_urllib_transport_failure_while_reading_is_transient(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    install(monkeypatch, FakeResponse(200, b"", read_error=error))
    with pytest.raises(TransientError):
        urllib_transport(URL, {"User-Agent": USER_AGENT}, 1.0)


class BrokenBody(io.BytesIO):
    """읽으면 연결이 끊기는 오류 응답 본문."""

    def read(self, size: int | None = -1, /) -> bytes:
        raise ConnectionResetError("reset while reading error body")


@pytest.mark.parametrize("code", [404, 503])
def test_http_urllib_transport_http_error_body_failure_keeps_status(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    # 상태 줄은 받았으므로 '응답 없음'이 아니다 — 404는 재시도 없이 404로 끝나야 한다
    error = urllib.error.HTTPError(URL, code, "error", email.message.Message(), BrokenBody())
    install(monkeypatch, error)
    assert urllib_transport(URL, {"User-Agent": USER_AGENT}, 1.0) == (code, b"")


def test_http_urllib_transport_does_not_hide_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, ValueError("unknown url type: 'statsapi'"))
    with pytest.raises(ValueError, match="unknown url type"):
        urllib_transport("statsapi", {"User-Agent": USER_AGENT}, 1.0)


def test_http_urllib_transport_network_guard_is_not_retryable() -> None:
    # conftest의 네트워크 차단(RuntimeError)이 TransientError로 바뀌어 조용히 재시도되면 안 된다.
    # 차단이 걸려 있으므로 실제로는 아무 연결도 나가지 않는다 (.invalid는 존재하지 않는 도메인)
    with pytest.raises(RuntimeError, match="외부 네트워크"):
        urllib_transport("https://example.invalid/", {"User-Agent": USER_AGENT}, 0.1)
