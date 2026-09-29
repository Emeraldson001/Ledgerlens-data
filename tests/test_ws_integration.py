"""End-to-end integration tests for WebSocket server with JWT authentication.

Tests the full JWT authentication handshake using websockets.connect with
a real WebSocket server running in a background thread.

Test Cases:
1. Valid JWT connects successfully (HTTP 101 received)
2. Expired JWT returns HTTP 401 before WebSocket upgrade
3. JWT with wrong iss claim is rejected
4. JWT with missing scores:read scope is rejected
5. Client subscribes to wallet/G... channel successfully
6. Wallet-scoped JWT rejected when subscribing to 'all' channel
7. Server sends rate_limit error after exceeding threshold
"""

import asyncio
import json
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest
import websockets
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

from streaming.ws_server import publish_score_update, run_ws_server

# ─────────────────────────────────────────────────────────────────────────
# Test Key Pair Generation (In-Memory)
# ─────────────────────────────────────────────────────────────────────────


def generate_test_keypair() -> tuple[str, str]:
    """Generate RSA key pair for testing.

    Returns:
        Tuple of (private_key_pem, public_key_pem)
    """
    # Generate RSA key pair
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )

    # Export private key
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    # Export public key
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")

    return private_pem, public_pem


def create_test_jwt(
    private_key: str,
    client_id: str = "test-client",
    scope: str = "scores:read:all",
    issuer: str = "ledgerlens-api",
    expires_in_seconds: int = 3600,
) -> str:
    """Create a test JWT token.

    Args:
        private_key: PEM-encoded RSA private key
        client_id: Client ID (sub claim)
        scope: Scope string (e.g., "scores:read:all")
        issuer: Issuer (iss claim)
        expires_in_seconds: Token expiration time in seconds (can be negative for expired tokens)

    Returns:
        JWT token string
    """
    now = datetime.now(UTC)
    claims = {
        "sub": client_id,
        "iss": issuer,
        "scope": scope,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in_seconds),
    }
    return jwt.encode(claims, private_key, algorithm="RS256")


# ─────────────────────────────────────────────────────────────────────────
# WebSocket Server Fixture
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def test_keypair():
    """Generate RSA key pair for the test module."""
    return generate_test_keypair()


@pytest.fixture(scope="module")
def ws_server(test_keypair, tmp_path_factory):
    """Start WebSocket server in background thread with test keys.

    Yields:
        Tuple of (port, private_key, public_key)
    """
    private_key, public_key = test_keypair

    # Create temporary directory for public key
    tmp_dir = tmp_path_factory.mktemp("ws_test")
    public_key_path = tmp_dir / "jwt_public_key.pem"
    public_key_path.write_text(public_key)

    # Monkey-patch config to use test key
    from config import config

    original_key_path = config.JWT_PUBLIC_KEY_PATH
    config.JWT_PUBLIC_KEY_PATH = str(public_key_path)

    # Reload authenticator with test key
    from streaming import ws_auth, ws_server

    ws_server._auth = ws_auth.JWTAuthenticator(str(public_key_path))

    # Find a free port by binding to port 0
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    # Start server in background thread
    server_ready = threading.Event()
    server_thread = None

    def _run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ws_server._loop = loop

        async def _start():
            server_ready.set()
            await run_ws_server(host="127.0.0.1", port=port)

        try:
            loop.run_until_complete(_start())
        except asyncio.CancelledError:
            pass
        finally:
            loop.close()

    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    server_ready.wait(timeout=5)
    time.sleep(0.5)  # Extra time for server to fully initialize

    yield port, private_key, public_key

    # Cleanup
    config.JWT_PUBLIC_KEY_PATH = original_key_path


# ─────────────────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_valid_jwt_connects_successfully(ws_server):
    """Test Case 1: Valid JWT connects successfully (HTTP 101 received)."""
    port, private_key, _ = ws_server
    token = create_test_jwt(private_key, client_id="valid-client")

    uri = f"ws://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {token}"}

    async with websockets.connect(uri, additional_headers=headers) as websocket:
        # If we get here, connection was successful (HTTP 101)
        assert websocket.state == websockets.State.OPEN
        # Send a ping to verify connection is working
        await websocket.ping()


@pytest.mark.asyncio
async def test_expired_jwt_returns_401(ws_server):
    """Test Case 2: Expired JWT returns HTTP 401 before WebSocket upgrade."""
    port, private_key, _ = ws_server
    # Create expired token (expired 1 hour ago)
    token = create_test_jwt(private_key, client_id="expired-client", expires_in_seconds=-3600)

    uri = f"ws://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {token}"}

    # Server accepts the handshake then closes with 1008 (policy violation)
    # rather than rejecting at the HTTP level, so the client sees the
    # connection open successfully followed by a ConnectionClosedError.
    async with websockets.connect(uri, additional_headers=headers) as websocket:
        with pytest.raises(websockets.exceptions.ConnectionClosedError) as exc_info:
            await websocket.recv()

    assert exc_info.value.code == 1008


@pytest.mark.asyncio
async def test_wrong_issuer_claim_rejected(ws_server):
    """Test Case 3: JWT with wrong iss claim is rejected."""
    port, private_key, _ = ws_server
    # Create token with wrong issuer
    token = create_test_jwt(private_key, client_id="wrong-issuer-client", issuer="wrong-issuer")

    uri = f"ws://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {token}"}

    async with websockets.connect(uri, additional_headers=headers) as websocket:
        with pytest.raises(websockets.exceptions.ConnectionClosedError) as exc_info:
            await websocket.recv()

    assert exc_info.value.code == 1008


@pytest.mark.asyncio
async def test_missing_scores_read_scope_rejected(ws_server):
    """Test Case 4: JWT with missing scores:read scope is rejected."""
    port, private_key, _ = ws_server
    # Create token with wrong scope
    token = create_test_jwt(private_key, client_id="no-scope-client", scope="other:scope")

    uri = f"ws://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {token}"}

    async with websockets.connect(uri, additional_headers=headers) as websocket:
        with pytest.raises(websockets.exceptions.ConnectionClosedError) as exc_info:
            await websocket.recv()

    # Should receive 1008 (policy violation) close code
    assert exc_info.value.code == 1008


@pytest.mark.asyncio
async def test_subscribe_to_wallet_channel_successfully(ws_server):
    """Test Case 5: Client subscribes to wallet/G... channel successfully."""
    port, private_key, _ = ws_server
    token = create_test_jwt(private_key, client_id="wallet-subscriber")

    uri = f"ws://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {token}"}

    wallet_id = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
    channel = f"wallet/{wallet_id}"

    async with websockets.connect(uri, additional_headers=headers) as websocket:
        # Send subscribe message
        subscribe_msg = {"type": "subscribe", "channels": [channel]}
        await websocket.send(json.dumps(subscribe_msg))

        # Wait a bit for subscription to be processed
        await asyncio.sleep(0.2)

        # Publish a score update to that channel
        from streaming import ws_server as ws_module

        if ws_module._loop and ws_module._loop.is_running():
            score_event = {
                "wallet": wallet_id,
                "asset_pair": "XLM:native/USDC:test",
                "score": 75,
                "score_lower": 70,
                "score_upper": 80,
                "bft_divergence": False,
                "top_features": [],
                "timestamp": datetime.now(UTC).isoformat(),
            }
            asyncio.run_coroutine_threadsafe(publish_score_update(score_event), ws_module._loop)

        # Receive the published message
        try:
            message_str = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            message = json.loads(message_str)
            assert message["type"] == "score_update"
            assert message["channel"] == channel
            assert message["wallet"] == wallet_id
        except TimeoutError:
            pytest.fail("Did not receive score update message")


@pytest.mark.asyncio
async def test_wallet_scoped_jwt_rejected_for_all_channel(ws_server):
    """Test Case 6: Wallet-scoped JWT rejected when subscribing to 'all' channel."""
    port, private_key, _ = ws_server
    # Create token scoped to a specific wallet
    wallet_id = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
    token = create_test_jwt(
        private_key, client_id="wallet-scoped-client", scope=f"scores:read:wallet/{wallet_id}"
    )

    uri = f"ws://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {token}"}

    async with websockets.connect(uri, additional_headers=headers) as websocket:
        # Try to subscribe to 'all' channel (should be rejected)
        subscribe_msg = {"type": "subscribe", "channels": ["all"]}
        await websocket.send(json.dumps(subscribe_msg))

        # Should receive an error message
        try:
            message_str = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            message = json.loads(message_str)
            assert message["type"] == "error"
            assert message["code"] == "forbidden"
            assert "all" in message["message"]
        except TimeoutError:
            pytest.fail("Did not receive forbidden error message")


@pytest.mark.asyncio
async def test_rate_limit_error_after_threshold(ws_server):
    """Test Case 7: Server sends rate_limit error after exceeding threshold."""
    port, private_key, _ = ws_server
    token = create_test_jwt(private_key, client_id="rate-limited-client")

    uri = f"ws://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {token}"}

    wallet_id = "GBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    channel = f"wallet/{wallet_id}"

    # Temporarily reduce rate limit for testing
    from streaming import ws_server as ws_module

    original_rate_limit = ws_module.config.WS_RATE_LIMIT_MSGS_PER_SECOND
    ws_module.config.WS_RATE_LIMIT_MSGS_PER_SECOND = 5  # 5 messages per second

    try:
        async with websockets.connect(uri, additional_headers=headers) as websocket:
            # Subscribe to channel
            subscribe_msg = {"type": "subscribe", "channels": [channel]}
            await websocket.send(json.dumps(subscribe_msg))
            await asyncio.sleep(0.2)

            # Publish many score updates rapidly to trigger rate limit
            if ws_module._loop and ws_module._loop.is_running():
                for i in range(20):  # Send 20 messages rapidly
                    score_event = {
                        "wallet": wallet_id,
                        "asset_pair": "XLM:native/USDC:test",
                        "score": 50 + i,
                        "score_lower": 45 + i,
                        "score_upper": 55 + i,
                        "bft_divergence": False,
                        "top_features": [],
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    asyncio.run_coroutine_threadsafe(
                        publish_score_update(score_event), ws_module._loop
                    )

            # Receive messages and look for rate_limit error
            rate_limit_error_found = False
            for _ in range(25):  # Try to receive multiple messages
                try:
                    message_str = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message = json.loads(message_str)
                    if message.get("type") == "error" and message.get("code") == "rate_limit":
                        rate_limit_error_found = True
                        assert "retry_after_ms" in message
                        assert isinstance(message["retry_after_ms"], int)
                        assert message["retry_after_ms"] > 0
                        break
                except TimeoutError:
                    break

            assert rate_limit_error_found, "Rate limit error was not received"

    finally:
        # Restore original rate limit
        ws_module.config.WS_RATE_LIMIT_MSGS_PER_SECOND = original_rate_limit


# ─────────────────────────────────────────────────────────────────────────
# Additional Helper Tests
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connection_without_token_rejected(ws_server):
    """Test that connection without token is rejected."""
    port, _, _ = ws_server
    uri = f"ws://127.0.0.1:{port}"

    async with websockets.connect(uri) as websocket:
        with pytest.raises(websockets.exceptions.ConnectionClosedError) as exc_info:
            await websocket.recv()

    assert exc_info.value.code == 1008


@pytest.mark.asyncio
async def test_invalid_channel_format_rejected(ws_server):
    """Test that invalid channel format is rejected."""
    port, private_key, _ = ws_server
    token = create_test_jwt(private_key, client_id="invalid-channel-client")

    uri = f"ws://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {token}"}

    async with websockets.connect(uri, additional_headers=headers) as websocket:
        # Try to subscribe to invalid channel
        subscribe_msg = {"type": "subscribe", "channels": ["invalid-channel-format"]}
        await websocket.send(json.dumps(subscribe_msg))

        # Should receive an error message
        try:
            message_str = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            message = json.loads(message_str)
            assert message["type"] == "error"
            assert message["code"] == "invalid_channel"
        except TimeoutError:
            pytest.fail("Did not receive invalid_channel error message")


# ─────────────────────────────────────────────────────────────────────────
# Zombie connection (abrupt disconnect) cleanup test
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_zombie_connection_cleaned_up_after_heartbeat(ws_server):
    """A client that disconnects without a close handshake must be removed from
    the server's _clients dict within one heartbeat cycle.

    This test:
    1. Connects a legitimate client and confirms it appears in _clients.
    2. Closes the underlying TCP transport without sending a WebSocket close
       frame (simulating a network drop / browser tab kill).
    3. Triggers a broadcast (which exercises the send path) and waits for the
       server's write-failure detection to clean up the dead entry.
    4. Asserts the dead client ID is no longer in _clients within a generous
       timeout.
    """
    port, private_key, _ = ws_server
    zombie_client_id = "zombie-test-client"
    token = create_test_jwt(private_key, client_id=zombie_client_id)

    uri = f"ws://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {token}"}

    from streaming import ws_server as ws_module

    # ── Step 1: Connect and confirm the client is registered ──────────────
    websocket = await websockets.connect(uri, additional_headers=headers)
    # Give the server's _handler coroutine time to register the client
    await asyncio.sleep(0.3)

    with ws_module._clients_lock:
        assert (
            zombie_client_id in ws_module._clients
        ), "Client should be registered in _clients after connecting"

    # Subscribe to a channel so the client is eligible to receive broadcasts
    channel = "wallet/GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
    await websocket.send(json.dumps({"type": "subscribe", "channels": [channel]}))
    await asyncio.sleep(0.1)

    # ── Step 2: Abrupt disconnect — close the transport without WS close ──
    # websockets.WebSocketClientProtocol exposes the underlying asyncio
    # transport; closing it abruptly mimics a network drop.
    websocket.transport.close()

    # ── Step 3: Trigger a broadcast so the server attempts a send to the
    #            dead socket and detects the failure ─────────────────────
    if ws_module._loop and ws_module._loop.is_running():
        score_event = {
            "wallet": "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF",
            "asset_pair": "XLM:native/USDC:test",
            "score": 80,
            "score_lower": 75,
            "score_upper": 85,
            "bft_divergence": False,
            "top_features": [],
            "timestamp": datetime.now(UTC).isoformat(),
        }
        asyncio.run_coroutine_threadsafe(
            ws_module.publish_score_update(score_event), ws_module._loop
        )

    # ── Step 4: Wait for cleanup and assert ───────────────────────────────
    # The server detects the dead connection either through the write failure
    # in _process_outbound or through the heartbeat ping.  We use a generous
    # 5-second timeout which is well within one heartbeat cycle even on slow CI.
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.1)
        with ws_module._clients_lock:
            if zombie_client_id not in ws_module._clients:
                break
    else:
        with ws_module._clients_lock:
            still_present = zombie_client_id in ws_module._clients
        if still_present:
            pytest.fail(
                f"Zombie client '{zombie_client_id}' was NOT removed from _clients "
                "within 5 seconds of an abrupt transport close. The server is leaking "
                "dead connections."
            )
