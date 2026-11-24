"""테스트 공통 설정 — 외부 네트워크 차단 (`live` 마커가 붙은 테스트만 허용, SPEC N5)."""

import socket
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return

    def guard(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("테스트에서 외부 네트워크를 쓸 수 없다 (live 테스트만 허용)")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
