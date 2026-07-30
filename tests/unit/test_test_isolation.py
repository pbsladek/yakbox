from __future__ import annotations

import socket

import pytest
from pytest_socket import SocketConnectBlockedError


def test_test_network_policy_allows_local_ipc_and_blocks_external_connections() -> None:
    left, right = socket.socketpair()
    left.close()
    right.close()

    external = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with (
            pytest.warns(UserWarning, match="A test tried to use socket"),
            pytest.raises(SocketConnectBlockedError),
        ):
            external.connect(("192.0.2.1", 443))
    finally:
        external.close()
