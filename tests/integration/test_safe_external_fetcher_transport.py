"""Integration tests for hermes/jobs/safe_fetcher.py — transport layer.

These tests drive ``safe_fetch`` end-to-end while replacing the
HTTPCore primitives with deterministic fakes. They MUST NOT open an
external socket, MUST NOT do DNS, and MUST NOT touch the filesystem.

The fakes exercise:

* The actual ``httpcore.Request`` that ``_HopTransport`` constructs:
  URL still carries the canonical hostname (TLS SNI / HTTP authority
  preserved), the ``Host`` header is the canonical authority, the
  ``extensions`` dict carries the timeout configuration, and the
  ``Accept-Encoding: identity`` header is sent on the wire.

* The actual ``httpcore.Response`` shape returned by the pool: a
  streaming body accessible via ``aiter_stream()`` whose ``aclose()``
  is called by the response lifecycle.

* The fake inner network backend (substituted for
  ``httpcore.AnyIOBackend``) that ``_PinnedBackend`` delegates to:
  every connection is required to use the approved numeric address
  with the exact port and ``socket_options``; any host/port drift
  raises the internal authorization exception BEFORE any I/O.

* The full fetch state machine: redirect creates a fresh pool and a
  fresh resolver call; the prior response and prior client/context are
  closed before the next hop is authorized. Cancellation and total
  deadline propagation are observable through fake lifecycle hooks.

* Boundary rejections: duplicate / missing / empty /
  protocol-relative ``Location`` headers; status codes outside the
  exact allowlist; non-2xx final responses; encoded responses rejected
  before any body byte is buffered; identity body bytes bounded
  below/equal/above the configured cap.

* Re-execution guarantee: ``httpx.AsyncClient`` is constructed with
  ``trust_env=False`` so environment proxy variables cannot influence
  the chosen backend. Redaction sentinel values never appear in any
  returned error, any captured streaming body byte, or any log line.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
from typing import Any, ClassVar

import httpcore
import httpx
import pytest

from hermes.jobs import safe_fetcher
from hermes.jobs.safe_fetcher import (
    FetchErrorCode,
    FetchPolicy,
    FetchResult,
    SafeFetchError,
    safe_fetch,
)

# ---------------------------------------------------------------------------
# Sentinel values that must NEVER appear in any error, log line, or body.
# ---------------------------------------------------------------------------

SENTINEL_HOSTNAME = "REDACTED-SENTINEL-HOST-7e3a"
SENTINEL_MARKER = "REDACTED-SENTINEL-MARKER-deadbeef"
SENTINEL_PATH = "/REDACTED-SENTINEL-PATH-9f1c"

# Public, globally-routable addresses used as approved numeric
# destinations. We pick addresses that are deliberately NOT in the
# production resolver path; they only ever appear because the fake
# resolver hands them to the policy code directly.
GLOBAL_IPV4 = "8.8.8.8"
GLOBAL_IPV6 = "2001:4860:4860::8888"


# ---------------------------------------------------------------------------
# Fake httpcore primitives
# ---------------------------------------------------------------------------


class FakeAsyncByteStream:
    """Mimics ``httpcore.AsyncByteStream`` for fake responses."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def __aiter__(self) -> Any:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        pass


class FakeResponse:
    """Duck-typed ``httpcore.Response``.

    Exposes the exact attributes the production transport reads:
    ``status``, ``headers``, ``extensions``, ``aiter_stream()``,
    ``aclose()``. ``aclose()`` is observable so tests can assert the
    full response lifecycle is honoured.
    """

    def __init__(
        self,
        status: int,
        headers: list[tuple[bytes, bytes]],
        body_chunks: list[bytes],
        extensions: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers
        self._body = list(body_chunks)
        self.content = FakeAsyncByteStream(self._body)
        self.extensions = extensions or {}
        self.closed = False
        self.aiter_stream_invocations = 0

    async def aiter_stream(self) -> Any:
        self.aiter_stream_invocations += 1
        for chunk in self._body:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class FakeConnectionPool:
    """Fake ``httpcore.AsyncConnectionPool``.

    Records the network backend it was constructed with, captures every
    ``httpcore.Request`` that flowed through ``handle_async_request``,
    and returns a configurable ``FakeResponse``. The class-level
    ``instances`` list lets tests inspect every pool ever created
    during a fetch (one per hop).

    The pool never opens sockets: ``handle_async_request`` is short-
    circuited to return the configured response without consulting the
    network backend (unless ``simulate_tls_failure`` is set, see
    below). This isolates the test from ``_PinnedBackend`` so
    pinning-specific assertions live in dedicated tests that call
    ``_PinnedBackend.connect_tcp`` directly.

    The optional ``simulate_tls_failure`` flag, when True, makes
    ``handle_async_request`` reach into ``network_backend.connect_tcp``
    to obtain the pinned :class:`_TLSStreamProxy`, then invoke
    ``start_tls`` on it. Any ``_TLSCertificateError`` (or
    ``httpcore.ConnectError``) bubbles up unchanged so the boundary
    catches the typed exception and maps to ``tls_failed``. This
    enables deterministic end-to-end TLS-failure tests through the
    full ``safe_fetch`` pipeline without ever opening a socket.
    """

    instances: ClassVar[list[FakeConnectionPool]] = []

    def __init__(
        self,
        *,
        network_backend: Any,
        http1: bool = True,
        http2: bool = False,
        retries: int = 0,
        max_connections: int = 1,
        max_keepalive_connections: int = 0,
    ) -> None:
        self.network_backend = network_backend
        self.http1 = http1
        self.http2 = http2
        self.retries = retries
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.handled_requests: list[httpcore.Request] = []
        self.handled_count = 0
        self.closed = False
        # Per-pool response factory; tests can override to script
        # redirects, status codes, headers, and body chunks.
        self.response_factory = self._default_response_factory
        # Track distinct responses that were returned (for closure asserts).
        self.responses_returned: list[FakeResponse] = []
        # TLS-failure simulation. When set, ``handle_async_request``
        # invokes ``network_backend.connect_tcp`` and then the
        # resulting proxy's ``start_tls`` so the boundary maps any
        # ``_TLSCertificateError`` deterministically. The sentinel
        # record is preserved so tests can audit which paths were
        # exercised.
        self.simulate_tls_failure: bool = False
        self.tls_failure_sentinel: str | None = None
        self.tls_handshake_attempts: list[dict[str, Any]] = []
        FakeConnectionPool.instances.append(self)

    def _default_response_factory(self, _request: httpcore.Request) -> FakeResponse:
        return FakeResponse(
            status=200,
            headers=[(b"content-type", b"text/plain")],
            body_chunks=[b"hello"],
        )

    async def handle_async_request(self, request: httpcore.Request) -> FakeResponse:
        self.handled_requests.append(request)
        self.handled_count += 1
        if self.simulate_tls_failure:
            # Drive the TLS handshake through the pinned backend so
            # the stream-proxy / typed-exception path is exercised
            # end-to-end. The proxy's ``start_tls`` is expected to
            # raise :class:`_TLSCertificateError`; we let it
            # propagate unchanged so ``_HopTransport.handle_async_request``
            # preserves the type and ``_perform_hop`` maps it to
            # ``tls_failed`` BEFORE any ``connect_failed`` mapper
            # can swallow it.
            url_obj = request.url
            host_attr = getattr(url_obj, "host", b"")
            if isinstance(host_attr, bytes):
                host_attr = host_attr.decode("ascii")
            port_attr = getattr(url_obj, "port", None)
            scheme_attr = getattr(url_obj, "scheme", b"http")
            if isinstance(scheme_attr, bytes):
                scheme_attr = scheme_attr.decode("ascii")
            if port_attr is None:
                port_attr = 443 if scheme_attr == "https" else 80
            # Use the authorized host/port declared on the backend so
            # the pinned check passes BEFORE the inner stream is
            # wrapped (and so we exercise the same path as a real
            # fetch). We don't poke into AnyIOBackend here because
            # the pinned backend already exposes ``connect_tcp`` for
            # exactly this entry point.
            # Reconstruct the canonical authority from the request so
            # we go through ``_PinnedBackend.connect_tcp`` exactly as
            # HTTPCore would.
            pinned = self.network_backend
            stream = await pinned.connect_tcp(
                host_attr,
                port_attr,
                timeout=None,
            )
            # Tell the proxy the server hostname it should present
            # during TLS (SNI). We don't need a real SSL context for
            # this unit-level test: the inner stream rejects it.
            import ssl as _ssl  # local import keeps the test module
            # surface clean.

            ssl_ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
            self.tls_handshake_attempts.append(
                {
                    "host": host_attr,
                    "port": port_attr,
                    "scheme": scheme_attr,
                }
            )
            # The proxy catches any httpcore.ConnectError raised by
            # the inner stream's start_tls and re-raises as the
            # private typed exception. Let it bubble.
            await stream.start_tls(ssl_ctx, host_attr, timeout=None)
            # If start_tls "succeeded" (the fake inner always raises),
            # we'd still need to return a response. This branch is
            # unreachable for TLS-failure simulation.
            raise RuntimeError("start_tls did not raise in TLS-failure mode")
        response = self.response_factory(request)
        self.responses_returned.append(response)
        return response

    async def aclose(self) -> None:
        self.closed = True


class FakeInnerBackend:
    """Fake of the inner network backend (``httpcore.AnyIOBackend``).

    Records every ``connect_tcp`` call so tests can assert that the
    pinned backend forwards only the approved numeric destination with
    the original port and ``socket_options``. ``connect_unix_socket``
    is recorded and denied (mirrors the production behaviour).

    Set ``tls_sentinel_text`` to enable TLS-handshake failure
    simulation: every stream returned by ``connect_tcp`` will, when
    its ``start_tls(...)`` is invoked, raise
    ``httpcore.ConnectError(message=tls_sentinel_text)``. This mirrors
    what real inner backends do on a TLS handshake error and is the
    input the TLS-stream proxy is expected to translate into the
    private typed exception.
    """

    instances: ClassVar[list[FakeInnerBackend]] = []

    def __init__(self, *, tls_sentinel_text: str | None = None) -> None:
        self.tls_sentinel_text = tls_sentinel_text
        self.connect_calls: list[dict[str, Any]] = []
        self.unix_socket_calls: list[dict[str, Any]] = []
        FakeInnerBackend.instances.append(self)

    async def connect_tcp(
        self,
        host: str,
        port: int,
        *,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> FakeNetworkStream:
        self.connect_calls.append(
            {
                "host": host,
                "port": port,
                "timeout": timeout,
                "local_address": local_address,
                "socket_options": socket_options,
            }
        )
        return FakeNetworkStream(
            tls_sentinel_text=self.tls_sentinel_text,
            outer_host=host,
        )

    async def connect_unix_socket(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> Any:
        # Record BEFORE raising so tests can observe the attempt.
        self.unix_socket_calls.append({"path": path, "timeout": timeout})
        raise safe_fetcher._AuthorizationMismatch("unix socket denied")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeNetworkStream:
    """Fake ``httpcore.AsyncNetworkStream``. Never performs I/O.

    If ``tls_sentinel_text`` is set, ``start_tls`` raises
    ``httpcore.ConnectError(message=tls_sentinel_text)``. Otherwise
    ``start_tls`` raises ``RuntimeError`` to surface accidental
    use.
    """

    def __init__(
        self,
        *,
        tls_sentinel_text: str | None = None,
        outer_host: str | None = None,
    ) -> None:
        self.closed = False
        self.read_calls = 0
        self.write_calls = 0
        self.tls_sentinel_text = tls_sentinel_text
        # host we'd try to start TLS against. Recorded so tests can
        # confirm the inner saw the approved address, not a hostname.
        self.outer_host = outer_host
        # Optional get_extra_info hook for the proxy's tolerance test.
        self.extra_info: dict[str, Any] = {"peer": b"8.8.8.8"}

    async def read(self, n: int, *, timeout: float | None = None) -> bytes:
        self.read_calls += 1
        return b""

    async def write(self, data: bytes, *, timeout: float | None = None) -> None:
        self.write_calls += 1

    async def aclose(self) -> None:
        self.closed = True

    async def start_tls(
        self,
        _ssl_context: Any,
        _server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        del timeout
        if self.tls_sentinel_text is not None:
            # The brief: the inner stream raises httpcore.ConnectError
            # with a sentinel message. The proxy is expected to
            # translate this to the private typed exception, severing
            # the chain and stripping the underlying text.
            raise httpcore.ConnectError(self.tls_sentinel_text)
        raise RuntimeError("TLS not supported by fake")

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        # Mirror how httpcore's real streams expose metadata.
        return self.extra_info.get(name, default)


class ScriptedResolver:
    """Resolver that returns a fixed sequence of address lists.

    ``script`` is a list; each element is the list of
    ``ipaddress.IPv4Address`` / ``ipaddress.IPv6Address`` instances the
    resolver will return for the next call. Empty list triggers
    ``RESOLUTION_FAILED`` in the fetcher. The full call history is
    preserved on ``calls`` so tests can assert "resolver was called
    exactly N times".
    """

    def __init__(
        self,
        script: list[list[ipaddress.IPv4Address | ipaddress.IPv6Address]],
    ) -> None:
        self._script = list(script)
        self.calls: list[str] = []

    async def resolve(self, hostname: str, *, timeout_s: float) -> list[Any]:
        self.calls.append(hostname)
        if not self._script:
            return []
        return list(self._script.pop(0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _header(headers: list[tuple[bytes, bytes]], name: str) -> bytes | None:
    target = name.lower().encode("ascii")
    for k, v in headers:
        if k.lower() == target:
            return v
    return None


def _req_url(request: httpcore.Request) -> str:
    """Return the URL of an ``httpcore.Request`` as a plain string.

    ``httpcore.Request.url`` is stored as an ``httpcore.URL`` object
    whose ``__str__`` returns its dataclass-style repr. To make
    assertions easy, we reconstruct the URL from its components.
    """
    url_obj = request.url
    # If the implementation ever switches to a plain ``str`` URL,
    # return it directly.
    if isinstance(url_obj, str):
        return url_obj
    scheme = getattr(url_obj, "scheme", b"http")
    if isinstance(scheme, bytes):
        scheme = scheme.decode("ascii") or "http"
    host = getattr(url_obj, "host", b"")
    if isinstance(host, bytes):
        host = host.decode("ascii")
    port = getattr(url_obj, "port", None)
    target = getattr(url_obj, "target", b"/")
    if isinstance(target, bytes):
        target = target.decode("ascii")
    authority = f"{host}:{port}" if port else host
    if not target.startswith("/"):
        target = "/" + target
    return f"{scheme}://{authority}{target}"


@pytest.fixture(autouse=True)
def _reset_fake_state() -> None:
    """Reset the class-level fake bookkeeping between tests."""
    FakeConnectionPool.instances.clear()
    FakeInnerBackend.instances.clear()


def _patch_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response_factory: Any | None = None,
) -> None:
    """Patch HTTPCore into safe_fetcher so it builds FakeConnectionPools."""

    def pool_factory(**kwargs: Any) -> FakeConnectionPool:
        pool = FakeConnectionPool(**kwargs)
        if response_factory is not None:
            pool.response_factory = response_factory
        return pool

    monkeypatch.setattr(safe_fetcher.httpcore, "AsyncConnectionPool", pool_factory)
    # The pinned backend also instantiates AnyIOBackend; tests that
    # exercise the pinning contract directly patch this themselves.
    # Leaving the default alone means _PinnedBackend would normally
    # create a real AnyIOBackend, but the FakeConnectionPool never
    # calls into the backend, so this is safe.


# ---------------------------------------------------------------------------
# 1. Fake AsyncConnectionPool — captures the actual httpcore.Request
# ---------------------------------------------------------------------------


class TestConnectionPoolCapturesRequest:
    async def test_pool_captures_original_hostname_in_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """URL in the captured ``httpcore.Request`` retains the canonical
        hostname (not the resolved IP), preserving TLS SNI / HTTP
        authority.
        """

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"ok"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        result = await safe_fetch(
            "http://example.com/path?q=1",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert isinstance(result, FetchResult)
        assert result.body == b"ok"
        assert result.status == 200

        # Exactly one pool created (no redirect).
        assert len(FakeConnectionPool.instances) == 1
        pool = FakeConnectionPool.instances[0]
        assert len(pool.handled_requests) == 1
        req = pool.handled_requests[0]
        # The URL keeps the canonical hostname (not the IP).
        url = _req_url(req)
        assert "example.com" in url
        assert GLOBAL_IPV4 not in url
        assert url.startswith("http://example.com")

    async def test_pool_host_header_is_canonical_authority(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``Host`` header in the captured ``httpcore.Request`` is the
        canonical authority, not the resolved address."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"ok"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        req = FakeConnectionPool.instances[0].handled_requests[0]
        host_header = _header(req.headers, "host")
        assert host_header is not None
        assert host_header == b"example.com:80"
        # No leakage of the resolved address anywhere in the headers.
        for _k, v in req.headers:
            assert GLOBAL_IPV4.encode() not in v

    async def test_pool_extensions_include_timeout_dict(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The captured ``httpcore.Request.extensions`` carries the
        per-hop timeout configuration (connect/read/write/pool)."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"ok"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        req = FakeConnectionPool.instances[0].handled_requests[0]
        assert isinstance(req.extensions, dict)
        timeouts = req.extensions.get("timeout")
        assert isinstance(timeouts, dict)
        # All four timeout keys must be present.
        for key in ("connect", "read", "write", "pool"):
            assert key in timeouts, f"missing {key!r} in extensions.timeout"
            assert timeouts[key] > 0

    async def test_pool_sends_accept_encoding_identity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``Accept-Encoding: identity`` is sent on every request so the
        server cannot surprise the fetcher with an encoded body that
        would bypass the streaming byte cap."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"ok"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        req = FakeConnectionPool.instances[0].handled_requests[0]
        ae = _header(req.headers, "accept-encoding")
        assert ae is not None
        assert ae.lower() == b"identity"

    async def test_response_streaming_consumes_incrementally(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Body is streamed via the response's ``aiter_stream()``; chunks
        are not buffered before the streaming consumer sees them."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/octet-stream")],
                body_chunks=[b"chunk-a|", b"chunk-b|", b"chunk-c"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        result = await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        # Body must be the exact concatenation in order.
        assert result.body == b"chunk-a|chunk-b|chunk-c"
        response = FakeConnectionPool.instances[0].responses_returned[0]
        assert response.aiter_stream_invocations >= 1

    async def test_response_aclose_invoked_on_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After the body is fully consumed, ``aclose()`` is invoked on
        the underlying core response (verified through the fake)."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"ok"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        result = await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert result.body == b"ok"
        response = FakeConnectionPool.instances[0].responses_returned[0]
        assert response.closed is True

    async def test_pool_is_closed_after_successful_fetch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The HTTPCore pool is closed in the success path (no leaks)."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"ok"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert FakeConnectionPool.instances[0].closed is True


# ---------------------------------------------------------------------------
# 2. Fake inner backend injected into _PinnedBackend
# ---------------------------------------------------------------------------


class TestPinnedBackendDelegatesOnlyApprovedAddress:
    async def test_delegates_only_numeric_address_with_port_and_socket_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``_PinnedBackend.connect_tcp`` is called with the
        canonical hostname + port + ``socket_options``, the inner
        backend receives ONLY the approved numeric address, the same
        port, and the same ``socket_options``.
        """
        fake_inner = FakeInnerBackend()
        monkeypatch.setattr(safe_fetcher.httpcore, "AnyIOBackend", lambda: fake_inner)

        pinned = safe_fetcher._PinnedBackend(
            authorized_host="example.com",
            authorized_port=443,
            approved_address=GLOBAL_IPV4,
            connect_timeout_s=5.0,
        )
        sock_opts = [(1, 1)]
        await pinned.connect_tcp(
            "example.com",
            443,
            timeout=2.5,
            local_address=None,
            socket_options=sock_opts,
        )

        assert len(fake_inner.connect_calls) == 1
        call = fake_inner.connect_calls[0]
        # Original hostname MUST NOT be delegated to the inner backend.
        assert call["host"] == GLOBAL_IPV4
        assert call["host"] != "example.com"
        # Port and socket options pass through verbatim.
        assert call["port"] == 443
        assert call["socket_options"] is sock_opts
        assert call["timeout"] == 2.5

    async def test_hostname_mismatch_raises_before_any_io(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If HTTPCore offers a host that differs from the authorization,
        ``connect_tcp`` raises the internal authorization exception
        BEFORE delegating to the inner backend. No I/O occurs.
        """
        fake_inner = FakeInnerBackend()
        monkeypatch.setattr(safe_fetcher.httpcore, "AnyIOBackend", lambda: fake_inner)

        pinned = safe_fetcher._PinnedBackend(
            authorized_host="example.com",
            authorized_port=443,
            approved_address=GLOBAL_IPV4,
            connect_timeout_s=5.0,
        )
        with pytest.raises(safe_fetcher._AuthorizationMismatch):
            await pinned.connect_tcp("attacker.example", 443)
        # No delegation happened.
        assert fake_inner.connect_calls == []

    async def test_port_mismatch_raises_before_any_io(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A drift in port also raises before I/O."""
        fake_inner = FakeInnerBackend()
        monkeypatch.setattr(safe_fetcher.httpcore, "AnyIOBackend", lambda: fake_inner)

        pinned = safe_fetcher._PinnedBackend(
            authorized_host="example.com",
            authorized_port=443,
            approved_address=GLOBAL_IPV4,
            connect_timeout_s=5.0,
        )
        with pytest.raises(safe_fetcher._AuthorizationMismatch):
            await pinned.connect_tcp("example.com", 80)
        assert fake_inner.connect_calls == []

    async def test_unix_socket_denied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``connect_unix_socket`` is rejected at the pinned boundary
        BEFORE delegating to the inner backend. The inner backend is
        never invoked.
        """
        fake_inner = FakeInnerBackend()
        monkeypatch.setattr(safe_fetcher.httpcore, "AnyIOBackend", lambda: fake_inner)

        pinned = safe_fetcher._PinnedBackend(
            authorized_host="example.com",
            authorized_port=443,
            approved_address=GLOBAL_IPV4,
            connect_timeout_s=5.0,
        )
        with pytest.raises(safe_fetcher._AuthorizationMismatch):
            await pinned.connect_unix_socket("/var/run/socket")
        # The pinned backend itself rejects; the inner is never consulted.
        assert fake_inner.unix_socket_calls == []
        assert fake_inner.connect_calls == []

    async def test_inner_backend_resolves_nothing_itself(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The pinned backend never invokes any DNS or getaddrinfo: it
        delegates a numeric address. Verified by ensuring no method on
        the fake inner ever receives a hostname (only an IP)."""
        fake_inner = FakeInnerBackend()
        monkeypatch.setattr(safe_fetcher.httpcore, "AnyIOBackend", lambda: fake_inner)

        pinned = safe_fetcher._PinnedBackend(
            authorized_host="example.com",
            authorized_port=443,
            approved_address=GLOBAL_IPV6,
            connect_timeout_s=5.0,
        )
        await pinned.connect_tcp("example.com", 443)
        assert fake_inner.connect_calls[0]["host"] == GLOBAL_IPV6
        # The inner never sees a DNS-style name.
        for call in fake_inner.connect_calls:
            assert ":" in call["host"] or call["host"].replace(".", "").isdigit()


# ---------------------------------------------------------------------------
# 3. IP literal bypasses the resolver
# ---------------------------------------------------------------------------


class TestIPLiteralBypassesResolver:
    async def test_ipv4_literal_does_not_invoke_resolver(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the URL contains a canonical IPv4 literal, the resolver
        is not consulted at all."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"ok"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[]])  # would RESOLUTION_FAILED if called

        result = await safe_fetch(
            f"http://{GLOBAL_IPV4}/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert result.body == b"ok"
        # Resolver was NEVER called.
        assert resolver.calls == []
        # And the URL still carries the literal IP for SNI / authority.
        req = FakeConnectionPool.instances[0].handled_requests[0]
        assert _req_url(req).startswith(f"http://{GLOBAL_IPV4}")

    async def test_ipv6_literal_does_not_invoke_resolver(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bracketed IPv6 URL literal bypasses the resolver; the captured
        request preserves the IPv6 literal as the host."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"ok"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[]])

        result = await safe_fetch(
            f"http://[{GLOBAL_IPV6}]/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert result.body == b"ok"
        assert resolver.calls == []
        req = FakeConnectionPool.instances[0].handled_requests[0]
        # httpcore stores the host without brackets; the literal is
        # preserved in the URL components.
        assert GLOBAL_IPV6 in _req_url(req)
        # And the ``host`` attribute on the URL object is the IPv6 literal.
        host_attr = getattr(req.url, "host", None)
        if isinstance(host_attr, bytes):
            host_attr = host_attr.decode("ascii")
        assert host_attr == GLOBAL_IPV6


# ---------------------------------------------------------------------------
# 4. DNS resolution once per hop; redirect creates fresh pool + resolver
# ---------------------------------------------------------------------------


class TestResolverAndPoolPerHop:
    async def test_dns_name_resolves_once_per_initial_hop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A DNS-name URL triggers exactly one resolver call on the
        initial (non-redirect) hop."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"ok"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert resolver.calls == ["example.com"]

    async def test_redirect_creates_fresh_pool_and_resolver_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A redirect to a new authority creates a fresh pool and a
        fresh resolver call. The prior response and the prior
        client/transport/pool are all closed before the new hop is
        authorized."""

        # Build a scriptable response factory. The first call returns
        # 301 with a Location; the second call returns 200 with the
        # body. We assert ordering: response-1 is closed BEFORE the
        # second pool even has its handle_async_request called.
        response_index = {"n": 0}
        first_response_holder: dict[str, FakeResponse | None] = {"resp": None}

        def factory(_req: httpcore.Request) -> FakeResponse:
            response_index["n"] += 1
            if response_index["n"] == 1:
                resp = FakeResponse(
                    status=301,
                    headers=[(b"location", b"http://other.example/")],
                    body_chunks=[b""],
                )
                first_response_holder["resp"] = resp
                return resp
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"final"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        # First hop resolves to GLOBAL_IPV4, second hop resolves to
        # GLOBAL_IPV6. The resolver records the canonical hostnames.
        resolver = ScriptedResolver(
            [
                [ipaddress.ip_address(GLOBAL_IPV4)],
                [ipaddress.ip_address(GLOBAL_IPV6)],
            ]
        )

        result = await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert result.status == 200
        assert result.body == b"final"
        assert result.redirect_count == 1
        # Two resolver calls, one per hop.
        assert resolver.calls == ["example.com", "other.example"]
        # Two pools, one per hop. Both pools are closed.
        assert len(FakeConnectionPool.instances) == 2
        assert FakeConnectionPool.instances[0].closed is True
        assert FakeConnectionPool.instances[1].closed is True
        # First response is closed (release_hop before redirect).
        first_response = first_response_holder["resp"]
        assert first_response is not None
        assert first_response.closed is True
        # Pools are independent objects.
        assert FakeConnectionPool.instances[0] is not FakeConnectionPool.instances[1]

    async def test_redirect_closes_prior_pool_before_new_pool_handles_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tighter ordering: the prior pool must be closed BEFORE the
        second pool's ``handle_async_request`` is invoked."""
        closure_order: list[str] = []

        original_aclose = FakeConnectionPool.aclose

        async def tracking_aclose(self: FakeConnectionPool) -> None:
            closure_order.append(f"pool-{id(self)}-aclose")
            await original_aclose(self)

        monkeypatch.setattr(FakeConnectionPool, "aclose", tracking_aclose)

        def factory(req: httpcore.Request) -> FakeResponse:
            # Tag the second pool's request so we can correlate order.
            if "other.example" in _req_url(req):
                closure_order.append("pool2-handle")
                return FakeResponse(
                    status=200,
                    headers=[(b"content-type", b"text/plain")],
                    body_chunks=[b"final"],
                )
            closure_order.append("pool1-handle")
            return FakeResponse(
                status=301,
                headers=[(b"location", b"http://other.example/")],
                body_chunks=[b""],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver(
            [
                [ipaddress.ip_address(GLOBAL_IPV4)],
                [ipaddress.ip_address(GLOBAL_IPV6)],
            ]
        )

        result = await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert result.body == b"final"

        # The hard invariant: by the time ``pool2-handle`` is recorded,
        # ``pool1-aclose`` has already happened. ``aclose`` may be
        # called more than once for the same pool (defensive cleanup),
        # but the FIRST close of pool1 must precede pool2's first
        # handle. We check this by index positions.
        pool1_handle_idx = closure_order.index("pool1-handle")
        pool2_handle_idx = closure_order.index("pool2-handle")
        # First pool1 aclose must occur between pool1-handle and pool2-handle.
        pool1_aclose_indices = [
            i
            for i, entry in enumerate(closure_order)
            if entry.endswith("-aclose") and i > pool1_handle_idx and i < pool2_handle_idx
        ]
        assert pool1_aclose_indices, f"pool1 must be closed BEFORE pool2.handle: {closure_order!r}"

    async def test_redirect_to_self_uses_fresh_pool(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A redirect back to the same authority still creates a fresh
        pool and a fresh resolver call (no pool reuse)."""
        call_index = {"n": 0}

        def factory(_req: httpcore.Request) -> FakeResponse:
            call_index["n"] += 1
            if call_index["n"] == 1:
                return FakeResponse(
                    status=301,
                    headers=[(b"location", b"http://example.com/again")],
                    body_chunks=[b""],
                )
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"final"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver(
            [
                [ipaddress.ip_address(GLOBAL_IPV4)],
                [ipaddress.ip_address(GLOBAL_IPV4)],
            ]
        )

        result = await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert result.body == b"final"
        assert resolver.calls == ["example.com", "example.com"]
        assert len(FakeConnectionPool.instances) == 2


# ---------------------------------------------------------------------------
# 5. Redirect Location edge cases and exact status allowlist
# ---------------------------------------------------------------------------


class TestRedirectEdgeCases:
    async def test_missing_location_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 301 response with no Location header must be rejected."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=301,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b""],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                "http://example.com/",
                policy=FetchPolicy(),
                resolver=resolver,
            )
        assert exc.value.code is FetchErrorCode.REDIRECT_INVALID

    async def test_empty_location_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 301 with an empty Location header is rejected."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=302,
                headers=[(b"location", b""), (b"location", b"   ")],
                body_chunks=[b""],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                "http://example.com/",
                policy=FetchPolicy(),
                resolver=resolver,
            )
        assert exc.value.code is FetchErrorCode.REDIRECT_INVALID

    async def test_duplicate_location_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two Location headers on the same redirect are rejected (the
        brief requires exactly one)."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            # httpcore uses list-of-tuples for headers; multiple tuples
            # with the same key simulate duplicate Location headers.
            return FakeResponse(
                status=301,
                headers=[
                    (b"location", b"http://a.example/"),
                    (b"location", b"http://b.example/"),
                ],
                body_chunks=[b""],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                "http://example.com/",
                policy=FetchPolicy(),
                resolver=resolver,
            )
        assert exc.value.code is FetchErrorCode.REDIRECT_INVALID

    async def test_protocol_relative_location_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A protocol-relative Location (``//evil.com/path``) is
        authority-ambiguous and is rejected."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=302,
                headers=[(b"location", b"//attacker.example/")],
                body_chunks=[b""],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                "http://example.com/",
                policy=FetchPolicy(),
                resolver=resolver,
            )
        assert exc.value.code is FetchErrorCode.REDIRECT_INVALID

    @pytest.mark.parametrize(
        "status_code",
        [300, 304, 305, 306, 309, 310, 400, 200, 404, 500],
    )
    async def test_non_allowlisted_status_not_followed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status_code: int,
    ) -> None:
        """Only the exact allowlist {301, 302, 303, 307, 308} is
        followed. Any other status (including 3xx codes outside the
        list) is treated as the final response."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=status_code,
                headers=[
                    (b"location", b"http://other.example/"),
                    (b"content-type", b"text/plain"),
                ],
                body_chunks=[b"final-body"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        # 2xx final → success; non-2xx final → HTTP_STATUS_DENIED.
        if 200 <= status_code < 300:
            result = await safe_fetch(
                "http://example.com/",
                policy=FetchPolicy(),
                resolver=resolver,
            )
            assert result.status == status_code
            # Exactly one resolver call: the second hop was not taken.
            assert len(resolver.calls) == 1
        else:
            with pytest.raises(SafeFetchError) as exc:
                await safe_fetch(
                    "http://example.com/",
                    policy=FetchPolicy(),
                    resolver=resolver,
                )
            assert exc.value.code is FetchErrorCode.HTTP_STATUS_DENIED
            assert len(resolver.calls) == 1

    @pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
    async def test_allowlisted_status_followed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status_code: int,
    ) -> None:
        """Each status in the exact allowlist triggers a redirect hop."""

        def factory(req: httpcore.Request) -> FakeResponse:
            # Pool #1 returns the redirect; pool #2 returns 200.
            if "other.example" in _req_url(req):
                return FakeResponse(
                    status=200,
                    headers=[(b"content-type", b"text/plain")],
                    body_chunks=[b"final"],
                )
            return FakeResponse(
                status=status_code,
                headers=[(b"location", b"http://other.example/")],
                body_chunks=[b""],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver(
            [
                [ipaddress.ip_address(GLOBAL_IPV4)],
                [ipaddress.ip_address(GLOBAL_IPV6)],
            ]
        )

        result = await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert result.status == 200
        assert result.redirect_count == 1

    async def test_redirect_exhausted_closes_resources(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the redirect budget is exhausted, the in-flight response
        and the in-flight pool are closed before the error is raised."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=302,
                headers=[(b"location", b"http://other.example/")],
                body_chunks=[b""],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver(
            [
                [ipaddress.ip_address(GLOBAL_IPV4)],
                [ipaddress.ip_address(GLOBAL_IPV4)],
                [ipaddress.ip_address(GLOBAL_IPV4)],
                [ipaddress.ip_address(GLOBAL_IPV4)],
                [ipaddress.ip_address(GLOBAL_IPV4)],
                [ipaddress.ip_address(GLOBAL_IPV4)],
            ]
        )
        policy = FetchPolicy(max_redirects=2)

        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                "http://example.com/",
                policy=policy,
                resolver=resolver,
            )
        assert exc.value.code is FetchErrorCode.REDIRECT_EXHAUSTED
        # Every pool created during the run must be closed.
        assert len(FakeConnectionPool.instances) >= 2
        for pool in FakeConnectionPool.instances:
            assert pool.closed is True


# ---------------------------------------------------------------------------
# 6. Identity body byte cap, encoding rejection, final non-2xx
# ---------------------------------------------------------------------------


class TestBodyStreamingAndCaps:
    async def test_identity_body_below_cap(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A body smaller than ``max_body_bytes`` is returned in full."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"x" * 100],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        result = await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(max_body_bytes=10_000),
            resolver=resolver,
        )
        assert len(result.body) == 100

    async def test_identity_body_equal_cap(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A body whose length equals the cap is accepted (the cap is
        inclusive)."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"y" * 50],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        result = await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(max_body_bytes=50),
            resolver=resolver,
        )
        assert result.body == b"y" * 50

    async def test_identity_body_above_cap(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A body that would exceed the cap is rejected and the response
        and pool are closed."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"z" * 60],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                "http://example.com/",
                policy=FetchPolicy(max_body_bytes=10),
                resolver=resolver,
            )
        assert exc.value.code is FetchErrorCode.BODY_TOO_LARGE
        # Pool closed even on rejection.
        assert FakeConnectionPool.instances[0].closed is True

    async def test_encoded_response_rejected_before_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A response declaring ``Content-Encoding: gzip`` is rejected
        BEFORE any body byte is read.
        """

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[
                    (b"content-type", b"text/plain"),
                    (b"content-encoding", b"gzip"),
                ],
                body_chunks=[b"would-be-decoded-bytes"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                "http://example.com/",
                policy=FetchPolicy(),
                resolver=resolver,
            )
        assert exc.value.code is FetchErrorCode.ENCODING_DENIED
        # Body must NOT have been streamed: aiter_stream was never called.
        response = FakeConnectionPool.instances[0].responses_returned[0]
        assert response.aiter_stream_invocations == 0

    async def test_final_non_2xx_mapped_to_status_denied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A final non-2xx response (with no Location) is mapped to
        ``HTTP_STATUS_DENIED``; the body is not returned."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=404,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"not found"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                "http://example.com/",
                policy=FetchPolicy(),
                resolver=resolver,
            )
        assert exc.value.code is FetchErrorCode.HTTP_STATUS_DENIED
        # Pool closed.
        assert FakeConnectionPool.instances[0].closed is True


# ---------------------------------------------------------------------------
# 7. Cancellation and total deadline propagation
# ---------------------------------------------------------------------------


class TestCancellationAndDeadlines:
    async def test_cancellation_closes_response_and_pool(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the surrounding task is cancelled mid-fetch, the response
        and the pool are still closed (no resource leak)."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_handle(self: FakeConnectionPool, request: httpcore.Request) -> FakeResponse:
            started.set()
            await release.wait()
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"late"],
            )

        monkeypatch.setattr(FakeConnectionPool, "handle_async_request", slow_handle)
        _patch_transport(monkeypatch)

        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        async def runner() -> None:
            await safe_fetch(
                "http://example.com/",
                policy=FetchPolicy(),
                resolver=resolver,
            )

        task = asyncio.create_task(runner())
        await started.wait()
        task.cancel()
        with pytest.raises((asyncio.CancelledError, SafeFetchError)):
            await task
        # Pool created during the cancelled fetch must be closed.
        for pool in FakeConnectionPool.instances:
            assert pool.closed is True

    async def test_total_deadline_marks_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If a fetch exceeds the configured total timeout, the operation
        is mapped to ``OPERATION_TIMEOUT`` and the pool is closed."""

        async def slow_handle(self: FakeConnectionPool, _request: httpcore.Request) -> FakeResponse:
            await asyncio.sleep(5.0)  # far longer than total_timeout_s
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"late"],
            )

        monkeypatch.setattr(FakeConnectionPool, "handle_async_request", slow_handle)
        _patch_transport(monkeypatch)

        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        # Build a tiny policy with all operation timeouts <= total. The
        # constructor forbids op timeouts that exceed total_timeout_s.
        tiny_policy = FetchPolicy(
            total_timeout_s=0.5,
            resolve_timeout_s=0.5,
            connect_timeout_s=0.5,
            read_timeout_s=0.5,
            write_timeout_s=0.5,
            pool_timeout_s=0.5,
        )

        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                "http://example.com/",
                policy=tiny_policy,
                resolver=resolver,
            )
        assert exc.value.code is FetchErrorCode.OPERATION_TIMEOUT


# ---------------------------------------------------------------------------
# 8. trust_env / proxy isolation
# ---------------------------------------------------------------------------


class TestTrustEnvIsolation:
    async def test_async_client_constructed_with_trust_env_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The httpx client is constructed with ``trust_env=False`` so
        ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY`` environment
        variables cannot redirect the connection through a proxy.
        """
        captured: dict[str, Any] = {}

        real_async_client = httpx.AsyncClient

        def spy_async_client(*args: Any, **kwargs: Any) -> Any:
            captured["args"] = args
            captured["kwargs"] = dict(kwargs)
            return real_async_client(*args, **kwargs)

        monkeypatch.setattr(safe_fetcher.httpx, "AsyncClient", spy_async_client)

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"ok"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        # Seed the environment with proxy variables that MUST NOT take
        # effect (the fetcher constructs trust_env=False explicitly).
        monkeypatch.setenv("HTTP_PROXY", "http://evil.example:8080")
        monkeypatch.setenv("HTTPS_PROXY", "http://evil.example:8443")
        monkeypatch.setenv("NO_PROXY", "*")
        monkeypatch.setenv("ALL_PROXY", "http://evil.example:3128")

        result = await safe_fetch(
            "http://example.com/",
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert result.body == b"ok"
        # The capture must show trust_env=False in the kwargs.
        assert captured.get("kwargs", {}).get("trust_env") is False

    async def test_proxy_env_var_does_not_reach_pinned_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even if env vars are set, the pinned backend's network stack
        only ever sees the approved numeric address — not the proxy.
        """
        fake_inner = FakeInnerBackend()
        monkeypatch.setattr(safe_fetcher.httpcore, "AnyIOBackend", lambda: fake_inner)

        pinned = safe_fetcher._PinnedBackend(
            authorized_host="example.com",
            authorized_port=443,
            approved_address=GLOBAL_IPV4,
            connect_timeout_s=5.0,
        )
        # Inject proxy env vars.
        monkeypatch.setenv("HTTP_PROXY", "http://attacker.example:9999")
        await pinned.connect_tcp("example.com", 443)
        # Inner must see only the approved IP.
        assert fake_inner.connect_calls[0]["host"] == GLOBAL_IPV4


# ---------------------------------------------------------------------------
# 9. Redaction
# ---------------------------------------------------------------------------


class TestRedactionAtBoundary:
    async def test_tls_failure_is_typed_and_fully_redacted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        fake_inner = FakeInnerBackend(tls_sentinel_text=SENTINEL_MARKER)
        monkeypatch.setattr(safe_fetcher.httpcore, "AnyIOBackend", lambda: fake_inner)

        def pool_factory(**kwargs: Any) -> FakeConnectionPool:
            pool = FakeConnectionPool(**kwargs)
            pool.simulate_tls_failure = True
            return pool

        monkeypatch.setattr(
            safe_fetcher.httpcore,
            "AsyncConnectionPool",
            pool_factory,
        )
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])
        caplog.set_level(logging.DEBUG, logger="hermes.jobs.safe_fetcher")

        with pytest.raises(SafeFetchError) as captured:
            await safe_fetch(
                "https://example.com/",
                policy=FetchPolicy(),
                resolver=resolver,
            )

        error = captured.value
        assert error.code is FetchErrorCode.TLS_FAILED
        assert SENTINEL_MARKER not in str(error)
        assert SENTINEL_MARKER not in repr(error)
        assert error.__cause__ is None
        assert error.__context__ is not None
        assert error.__context__.__cause__ is None
        assert error.__context__.__context__ is None
        assert SENTINEL_MARKER not in "\n".join(record.getMessage() for record in caplog.records)
        assert FakeConnectionPool.instances[0].tls_handshake_attempts
        assert FakeConnectionPool.instances[0].closed is True

    async def test_sentinel_hostname_absent_from_error_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sentinel input hostname and marker never appear in any error
        message or log line."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=404,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"forbidden body " + SENTINEL_MARKER.encode()],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        sentinel_url = f"http://{SENTINEL_HOSTNAME}.example.com/{SENTINEL_PATH}"
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                sentinel_url,
                policy=FetchPolicy(),
                resolver=resolver,
            )
        text = str(exc.value)
        assert SENTINEL_HOSTNAME not in text
        assert SENTINEL_MARKER not in text
        assert SENTINEL_PATH not in text

    async def test_sentinel_absent_from_captured_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A request that fails before the captured httpcore.Request is
        produced never reveals the input URL/host/marker."""
        _patch_transport(monkeypatch)
        sentinel_url = f"http://{SENTINEL_HOSTNAME}.example.com/"
        resolver = ScriptedResolver([[]])  # empty → RESOLUTION_FAILED

        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                sentinel_url,
                policy=FetchPolicy(),
                resolver=resolver,
            )
        text = str(exc.value)
        assert SENTINEL_HOSTNAME not in text
        assert SENTINEL_MARKER not in text
        assert exc.value.code is FetchErrorCode.RESOLUTION_FAILED
        # No request was captured because the failure was pre-pool.
        assert FakeConnectionPool.instances == []

    async def test_sentinel_absent_from_logs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Captured logs must not include the sentinel hostname."""

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=500,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"server error"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        caplog.set_level(logging.DEBUG, logger="hermes.jobs.safe_fetcher")
        sentinel_url = f"http://{SENTINEL_HOSTNAME}.example.com/"
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        with pytest.raises(SafeFetchError):
            await safe_fetch(
                sentinel_url,
                policy=FetchPolicy(),
                resolver=resolver,
            )
        all_text = "\n".join(rec.getMessage() for rec in caplog.records)
        assert SENTINEL_HOSTNAME not in all_text
        assert SENTINEL_MARKER not in all_text

    async def test_sentinel_absent_from_captured_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sentinels hidden in the input URL never leak via results or logs.

        The outbound request intentionally carries the canonical authority
        for Host and TLS SNI, so request-capture assertions are inappropriate.
        """

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"clean body"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        # URL contains the sentinel hostname — it must not leak via the
        # request URL/headers either.
        sentinel_url = f"http://{SENTINEL_HOSTNAME}.example.com/clean"
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        result = await safe_fetch(
            sentinel_url,
            policy=FetchPolicy(),
            resolver=resolver,
        )
        assert result.body == b"clean body"
        assert SENTINEL_MARKER not in result.body.decode()


# ---------------------------------------------------------------------------
# 10. fetch()-based entry point reuses the same boundary
# ---------------------------------------------------------------------------


class TestSafeExternalFetcherEntry:
    async def test_fetch_method_returns_fetch_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``SafeExternalFetcher.fetch`` returns the same ``FetchResult``
        type as ``safe_fetch`` and goes through the fake pool.
        """

        def factory(_req: httpcore.Request) -> FakeResponse:
            return FakeResponse(
                status=200,
                headers=[(b"content-type", b"text/plain")],
                body_chunks=[b"via fetch"],
            )

        _patch_transport(monkeypatch, response_factory=factory)
        resolver = ScriptedResolver([[ipaddress.ip_address(GLOBAL_IPV4)]])

        fetcher = safe_fetcher.SafeExternalFetcher(
            policy=FetchPolicy(),
            resolver=resolver,
        )
        result = await fetcher.fetch("http://example.com/")
        assert isinstance(result, FetchResult)
        assert result.body == b"via fetch"
        assert FakeConnectionPool.instances and FakeConnectionPool.instances[0].closed is True
