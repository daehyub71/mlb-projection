"""네트워크 차단 통제군 — 차단이 실제로 동작하는지 확인한다."""

import socket
import urllib.request

import pytest


def test_conftest_blocks_socket_connect() -> None:
    with pytest.raises(RuntimeError, match="외부 네트워크"):
        socket.create_connection(("example.com", 443), timeout=1)


def test_conftest_blocks_urllib() -> None:
    with pytest.raises(Exception, match="외부 네트워크"):
        urllib.request.urlopen("https://example.com", timeout=1)
