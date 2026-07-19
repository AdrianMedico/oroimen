"""Slice 1B: SafeExternalFetcher.

A reusable, fail-closed external GET fetch boundary that defeats
redirect SSRF and DNS rebinding, preserves TLS hostname verification
and HTTP authority, enforces a total deadline and streamed size
limit, ignores inherited proxy configuration, and emits only
stable redacted failures.

The fetcher is not wired into Deep Research in this slice. The
unsafe existing Phase 2 path remains unreachable because Deep
Research is disabled and unwired.

Public surface (selected):

    FetchPolicy                       - frozen, validated policy.
    FetchResult                       - successful fetch result.
    SafeFetchError(code)              - redacted error type.
    FetchErrorCode                    - closed stable enum.
    AddressResolver                   - async resolver protocol.
    DualAddressResolver               - production resolver.
    SafeExternalFetcher(policy, ...)  - the boundary.
    safe_fetch(url, policy, ...)      - convenience entry point.

All logging and all returned errors are redacted: no input URL,
hostname, address, header, body bytes, or underlying exception
text is ever included.
"""

from __future__ import annotations

import asyncio
import enum
import ipaddress
import logging
import socket
import ssl
from contextlib import suppress
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error surface
# ---------------------------------------------------------------------------


class FetchErrorCode(enum.StrEnum):
    """Closed, stable set of failure codes for the public boundary."""

    INVALID_URL = "invalid_url"
    SCHEME_DENIED = "scheme_denied"
    AUTHORITY_DENIED = "authority_denied"
    PORT_DENIED = "port_denied"
    RESOLUTION_FAILED = "resolution_failed"
    DESTINATION_DENIED = "destination_denied"
    AUTHORIZATION_MISMATCH = "authorization_mismatch"
    REDIRECT_INVALID = "redirect_invalid"
    REDIRECT_EXHAUSTED = "redirect_exhausted"
    CONNECT_FAILED = "connect_failed"
    TLS_FAILED = "tls_failed"
    HTTP_STATUS_DENIED = "http_status_denied"
    ENCODING_DENIED = "encoding_denied"
    BODY_TOO_LARGE = "body_too_large"
    OPERATION_TIMEOUT = "operation_timeout"
    INTERNAL_FAILURE = "internal_failure"


# Stable, ordered list of every code. Tests assert membership against this.
ALL_FETCH_ERROR_CODES: tuple[str, ...] = tuple(c.value for c in FetchErrorCode)


class _AuthorizationMismatch(Exception):
    """Internal exception raised when HTTPCore's host/port drifts from auth.

    Caught at the boundary and mapped to FetchErrorCode.AUTHORIZATION_MISMATCH.
    The string form intentionally carries no identifying data.
    """


class _TLSCertificateError(Exception):
    """Internal typed exception raised when the TLS handshake fails.

    The pinned :class:`httpcore.AsyncNetworkStream` proxy translates the
    raw underlying ``httpcore.ConnectError`` raised inside
    ``start_tls(...)`` into this private typed exception so the boundary
    can map it deterministically to :data:`FetchErrorCode.TLS_FAILED`
    BEFORE the broader :class:`httpcore.ConnectError` catch in
    :class:`_HopTransport` swallows it as ``connect_failed``.

    The exception carries a stable, generic code only. NO underlying
    text — the literal error message, hostname, certificate subject, or
    any other identifying data from the underlying ssl/httpcore
    exception — is included. The accompanying :class:`FetchErrorCode`
    distinguishes TLS handshake failures from generic connectivity
    failures so the caller can decide how to surface them.
    """

    code = FetchErrorCode.TLS_FAILED


class SafeFetchError(Exception):
    """Public failure type. Carries a FetchErrorCode only.

    The message is intentionally generic so that no input URL, hostname,
    address, header, body bytes, or underlying exception text leaks out.
    """

    def __init__(self, code: FetchErrorCode) -> None:
        self.code = code
        super().__init__(f"safe_fetch_failed:{code.value}")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


# Version 1 default port map (no other ports are accepted).
_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    """Frozen, strict policy for one safe_fetch invocation.

    All caps and timeouts are strictly positive. An operation timeout
    cannot exceed the total timeout. Construction rejects invalid
    combinations rather than silently clamping them.

    The dataclass is `frozen=True`, but because the port map is a
    `dict` value, this only prevents attribute reassignment. We
    additionally deep-validate and deep-freeze the port mapping in
    `__post_init__` so callers cannot mutate it through the field.
    """

    allowed_schemes: frozenset[str] = field(
        default=frozenset({"http", "https"}),
    )
    scheme_default_ports: dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_PORTS),
    )
    max_redirects: int = 5
    max_body_bytes: int = 2_000_000
    resolve_timeout_s: float = 5.0
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 10.0
    write_timeout_s: float = 5.0
    pool_timeout_s: float = 5.0
    total_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        # Only the canonical {http:80, https:443} pair is acceptable.
        if not self.allowed_schemes:
            raise ValueError("allowed_schemes must not be empty")
        if self.allowed_schemes != frozenset({"http", "https"}):
            raise ValueError("allowed_schemes must equal {'http', 'https'}")
        # Copy + freeze the port mapping. The dataclass stores its own
        # private copy wrapped in MappingProxyType so callers cannot
        # mutate after construction. Equality with a plain dict still
        # works through MappingProxyType.__eq__.
        ports = dict(self.scheme_default_ports)
        expected_ports = {"http": 80, "https": 443}
        if ports != expected_ports:
            raise ValueError("scheme_default_ports must equal {'http': 80, 'https': 443}")
        for scheme, port_value in ports.items():
            if not isinstance(port_value, int) or port_value <= 0 or port_value > 65535:
                raise ValueError(f"port for {scheme!r} must be a valid integer")
        object.__setattr__(self, "scheme_default_ports", MappingProxyType(ports))
        # max_redirects: strictly positive AND <= brief cap.
        if self.max_redirects < 1:
            raise ValueError("max_redirects must be >= 1")
        if self.max_redirects > 5:
            raise ValueError("max_redirects must be <= 5")
        if self.max_body_bytes < 1:
            raise ValueError("max_body_bytes must be >= 1")
        for name, timeout_value in (
            ("resolve_timeout_s", self.resolve_timeout_s),
            ("connect_timeout_s", self.connect_timeout_s),
            ("read_timeout_s", self.read_timeout_s),
            ("write_timeout_s", self.write_timeout_s),
            ("pool_timeout_s", self.pool_timeout_s),
            ("total_timeout_s", self.total_timeout_s),
        ):
            if timeout_value <= 0:
                raise ValueError(f"{name} must be > 0")
        # Operation timeouts cannot exceed the total. `resolve_timeout_s`
        # is the one we previously forgot; include it here.
        for name, timeout_value in (
            ("resolve_timeout_s", self.resolve_timeout_s),
            ("connect_timeout_s", self.connect_timeout_s),
            ("read_timeout_s", self.read_timeout_s),
            ("write_timeout_s", self.write_timeout_s),
            ("pool_timeout_s", self.pool_timeout_s),
        ):
            if timeout_value > self.total_timeout_s:
                raise ValueError(
                    f"{name} ({timeout_value}) cannot exceed total_timeout_s ({self.total_timeout_s})"
                )


# ---------------------------------------------------------------------------
# Authorization / result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuthorizedHop:
    """One authorized hop: canonical scheme, hostname, port, selected address."""

    scheme: str
    hostname: str
    port: int
    target: str
    selected_address: str


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Successful fetch result. Bounded bytes and redacted of identifying data."""

    body: bytes
    media_type: str
    status: int
    redirect_count: int


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


@runtime_checkable
class AddressResolver(Protocol):
    """Asynchronous resolver protocol.

    The production implementation requests both address families and
    normalizes results through `ipaddress`.
    """

    async def resolve(
        self,
        hostname: str,
        *,
        timeout_s: float,
    ) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]: ...


class DualAddressResolver:
    """Production resolver.

    Requests both address families through getaddrinfo and returns the
    canonical numeric results. No name, address, or exception detail
    is logged on failure.
    """

    async def resolve(
        self,
        hostname: str,
        *,
        timeout_s: float,
    ) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        loop = asyncio.get_running_loop()
        # Use AF_UNSPEC to request both v4 and v6.
        try:
            infos = await asyncio.wait_for(
                loop.getaddrinfo(
                    hostname,
                    None,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                    proto=socket.IPPROTO_TCP,
                ),
                timeout=timeout_s,
            )
        except (socket.gaierror, OSError, TimeoutError):
            # Redacted: no name, no exception text.
            return []
        out: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
        for info in infos:
            sockaddr = info[4]
            if not sockaddr:
                continue
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            out.append(ip)
        return out


# ---------------------------------------------------------------------------
# Canonicalization / validation
# ---------------------------------------------------------------------------


_BRACKET_LEFT = "["
_BRACKET_RIGHT = "]"


def _strip_brackets_ipv6(host: str) -> str:
    """Strip IPv6 literal brackets, if present, before literal parsing.

    Brackets are URL syntax. They MUST NOT be sent to the resolver; they
    MUST NOT appear on the wire to DNS. If the brackets are mismatched,
    reject.
    """
    if host.startswith(_BRACKET_LEFT) and host.endswith(_BRACKET_RIGHT) and len(host) >= 2:
        return host[1:-1]
    if _BRACKET_LEFT in host or _BRACKET_RIGHT in host:
        raise ValueError("mismatched brackets in IPv6 literal")
    return host


_DNS_LABEL_MAX_LEN = 63  # RFC 1035 §2.3.4: each label 1-63 octets.
_DNS_NAME_MAX_LEN = 253  # RFC 1035 §2.3.4: full name <=253 octets (incl. dots).


def _validate_strict_dns_label(label: str) -> None:
    """Validate that `label` is a strict LDH (letter/digit/hyphen) DNS label.

    Rules (RFC 1035 §2.3.4 + RFC 5890/5891 IDN A-label constraints):

    * length 1..63 octets;
    * only ASCII letters, digits, and ``-``;
    * underscore (``_``) is rejected: DNS permits it in some queries but it
      is not part of the canonical hostname syntax this boundary accepts;
    * no leading or trailing hyphen (LDH-only-apart-from-NN);
    * IDN A-labels begin with ``xn--``; the rest after ``xn--`` is opaque
      bytes that are implicitly validated by the IDNA encoder above and
      not re-checked here label-by-label (the encoder rejects malformed
      A-labels).
    """
    if not label:
        raise ValueError("empty dns label")
    if len(label) > _DNS_LABEL_MAX_LEN:
        raise ValueError("dns label exceeds 63 octets")
    if "_" in label:
        raise ValueError("underscore in dns label rejected")
    if label.startswith("-") or label.endswith("-"):
        raise ValueError("leading or trailing hyphen in dns label rejected")
    # ASCII letters / digits / hyphen only. ``X`` rejects anything outside
    # ``[A-Za-z0-9-]`` so e.g. ``xn--bcher-kva.example`` passes.
    for ch in label:
        if not (("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "-"):
            raise ValueError("non-ldh character in dns label rejected")


def canonicalize_hostname(raw: str) -> str:
    """Canonicalize an authority hostname.

    - IDN is converted via strict IDNA (Uts#46).
    - DNS names are lowercased and one terminal dot is removed.
    - Zone identifiers (RFC 6874) are rejected.
    - Percent-encoded authority is rejected.
    - Alternate / shortened / decimal / octal / hex address forms are
      rejected in favor of the canonical numeric form.
    - An IP literal must parse as either an IPv4Address or an IPv6Address
      in its canonical form. IPv4-mapped IPv6 is rejected.
    - DNS names are validated label-by-label under strict LDH rules:
      1-63 ASCII letters/digits/hyphen per label, no underscore, no
      leading/trailing hyphen, total length <=253 octets.

    Returns the canonical ASCII hostname. Raises ValueError on anything
    not acceptable.
    """
    if raw is None:
        raise ValueError("hostname is None")
    if not raw:
        raise ValueError("hostname is empty")
    if "%" in raw:
        # RFC 6874 zone identifier or percent-encoded authority.
        raise ValueError("percent-encoded authority or zone identifier rejected")
    if _BRACKET_LEFT in raw or _BRACKET_RIGHT in raw:
        # Brackets are URL syntax; they don't belong in the authority
        # passed to the resolver. The URL parser should have stripped
        # them before us; if any are present we reject.
        raise ValueError("bracket in authority rejected")

    # Try as literal IP first, but reject non-canonical numeric spellings.
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        ip = None
    if ip is not None:
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            raise ValueError("ipv4-mapped ipv6 literal rejected")
        # The brief requires an `is_global` check only. No carve-outs for
        # documentation or special-purpose ranges.
        if not ip.is_global:
            raise ValueError("non-global address literal rejected")
        if isinstance(ip, ipaddress.IPv4Address) and raw != str(ip):
            raise ValueError("non-canonical ipv4 literal rejected")
        if isinstance(ip, ipaddress.IPv6Address) and raw.lower() != ip.compressed.lower():
            raise ValueError("non-canonical ipv6 literal rejected")
        return ip.compressed

    # Reject alternate numeric IPv4 spellings (shortened, decimal, octal,
    # and hexadecimal) instead of treating them as DNS names.
    if raw.replace(".", "").isdigit() or raw.lower().startswith(("0x", "0o")):
        raise ValueError("alternate numeric address form rejected")

    # DNS name. Strict IDNA: convert Unicode to ASCII A-label form.
    try:
        ascii_name = raw.encode("idna").decode("ascii")
    except UnicodeError:
        raise ValueError("idna encode failed") from None

    # Lowercase and strip one terminal dot.
    ascii_name = ascii_name.lower()
    if ascii_name.endswith("."):
        ascii_name = ascii_name[:-1]
    if not ascii_name:
        raise ValueError("hostname became empty after canonicalization")

    # Defensive: reject embedded nulls, spaces, controls, slashes.
    if any(c <= " " or c == "/" for c in ascii_name):
        raise ValueError("control or whitespace in hostname")
    if "\x00" in ascii_name:
        raise ValueError("null byte in hostname")

    # Strict DNS label validation. Every label is 1..63 ASCII letters /
    # digits / hyphen, no leading/trailing hyphen, no underscore. The
    # total name (dots included) must be <=253 octets per RFC 1035.
    if len(ascii_name) > _DNS_NAME_MAX_LEN:
        raise ValueError("dns name exceeds 253 octets")
    for label in ascii_name.split("."):
        _validate_strict_dns_label(label)

    return ascii_name


def _validate_address_for_destination(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Return True iff this address is acceptable as a single hop target."""
    # Reject any address that is mapped, scoped, ambiguous, or in a
    # transition/special range not explicitly accepted. The brief says:
    # "accept only addresses where `is_global` is True."
    if isinstance(ip, ipaddress.IPv6Address):
        if (
            ip.ipv4_mapped is not None
            or ip.is_site_local
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_unspecified
            or ip.is_multicast
            or ip.scope_id is not None
        ):
            return False
    elif isinstance(ip, ipaddress.IPv4Address) and (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return False
    return bool(ip.is_global)


# ---------------------------------------------------------------------------
# TLS-stream proxy: deterministic mapping of TLS handshake failures
# ---------------------------------------------------------------------------


class _TLSStreamProxy(httpcore.AsyncNetworkStream):
    """Public-stream contract wrapper that maps TLS handshake failures.

    Why: HTTPCore drives the TLS handshake by calling
    ``inner.start_tls(ssl_context, server_hostname, timeout)`` on the
    stream returned by :meth:`_PinnedBackend.connect_tcp`. If the
    handshake fails, httpcore raises :class:`httpcore.ConnectError`
    from inside that call. Without an intercept, that ``ConnectError``
    would be indistinguishable from a generic TCP-connect failure at our
    boundary, and the failure would be mapped to
    :data:`FetchErrorCode.CONNECT_FAILED` rather than
    :data:`FetchErrorCode.TLS_FAILED`.

    The proxy keeps the public
    :class:`httpcore.AsyncNetworkStream` contract intact
    (``read``/``write``/``aclose``/``get_extra_info``) and translates
    any :class:`httpcore.ConnectError` raised inside ``start_tls`` into
    the private typed exception :class:`_TLSCertificateError`. The
    boundary catches that typed exception BEFORE the generic
    ``httpcore.ConnectError`` handler and maps it deterministically to
    :data:`FetchErrorCode.TLS_FAILED`.

    On successful TLS, the returned TLS stream is itself wrapped in a
    fresh :class:`_TLSStreamProxy` so the same conversion applies to
    any later operation.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: httpcore.AsyncNetworkStream) -> None:
        self._inner = inner

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return await self._inner.read(max_bytes, timeout=timeout)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        await self._inner.write(buffer, timeout=timeout)

    async def aclose(self) -> None:
        await self._inner.aclose()

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        # httpcore stream protocols expose a ``get_extra_info`` lookup;
        # forward the call so callers retain visibility into the
        # underlying socket / TLS state. We tolerate both the two-arg
        # and single-arg forms used by different stream implementations.
        inner_get = getattr(self._inner, "get_extra_info", None)
        if inner_get is None:
            return default
        try:
            return inner_get(name, default)
        except TypeError:
            try:
                return inner_get(name)
            except Exception:
                return default

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> _TLSStreamProxy:
        """Delegate the TLS handshake, translating failures to typed exn.

        On success, the new TLS stream is itself wrapped so the same
        diagnostic translation covers any subsequent operations.
        """
        try:
            tls_stream = await self._inner.start_tls(
                ssl_context,
                server_hostname,
                timeout=timeout,
            )
        except httpcore.ConnectError:
            # Leave the exception handler before raising the typed error.
            # This prevents Python from retaining the identifying
            # ConnectError as ``__context__`` at the public boundary.
            pass
        else:
            return _TLSStreamProxy(tls_stream)
        raise _TLSCertificateError()


# ---------------------------------------------------------------------------
# Pinned network backend
# ---------------------------------------------------------------------------


class _PinnedBackend(httpcore.AsyncNetworkBackend):
    """Pinned network backend for one authorized hop.

    - `connect_tcp()` validates the host/port offered by HTTPCore against
      the authorization before any I/O, then delegates ONLY the approved
      numeric address to the underlying AnyIO backend.
    - The original hostname is never resolved by this backend.
    - The original hostname remains in the HTTPCore request URL so TLS
      SNI, certificate verification, and HTTP authority are correct.
    - The stream returned to HTTPCore is wrapped in :class:`_TLSStreamProxy`
      so a TLS handshake failure surfaces as the typed
      :class:`_TLSCertificateError` (mapping to ``tls_failed``) rather
      than being absorbed as a generic
      :class:`httpcore.ConnectError`.
    """

    def __init__(
        self,
        *,
        authorized_host: str,
        authorized_port: int,
        approved_address: str,
        connect_timeout_s: float,
    ) -> None:
        self._authorized_host = authorized_host
        self._authorized_port = authorized_port
        self._approved_address = approved_address
        self._connect_timeout_s = connect_timeout_s
        self._inner = httpcore.AnyIOBackend()

    async def connect_tcp(  # type: ignore[override]
        self,
        host: str,
        port: int,
        *,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        # Compare the host and port supplied by HTTPCore with the
        # authorization BEFORE any I/O.
        if host != self._authorized_host or port != self._authorized_port:
            raise _AuthorizationMismatch("host/port drift")
        effective_timeout = (
            self._connect_timeout_s if timeout is None else min(timeout, self._connect_timeout_s)
        )
        # Delegate ONLY the approved numeric destination. The original
        # hostname is never resolved here. socket_options is forwarded
        # verbatim so the inner AnyIO backend can apply per-socket
        # tuning (TCP_NODELAY, etc.) without us re-implementing it.
        inner_stream = await self._inner.connect_tcp(
            self._approved_address,
            port,
            timeout=effective_timeout,
            local_address=local_address,
            socket_options=socket_options,
        )
        # Wrap the raw stream so any TLS handshake failure inside
        # ``start_tls`` surfaces as :class:`_TLSCertificateError` at the
        # boundary, never as a bare :class:`httpcore.ConnectError`. The
        # proxy is opaque to HTTPCore: it implements the full stream
        # contract.
        return _TLSStreamProxy(inner_stream)

    async def connect_unix_socket(  # type: ignore[override]
        self,
        path: str,
        *,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        # Unix sockets are never an authorized destination. We accept
        # the ``socket_options`` keyword so the signature matches the
        # pinned httpcore 1.0.9 contract — HTTPCore may pass it through
        # even on unix-socket paths.
        del path, timeout, socket_options
        raise _AuthorizationMismatch("unix socket denied")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


# ---------------------------------------------------------------------------
# HTTPX transport adapter wrapping a fresh pool per hop
# ---------------------------------------------------------------------------


class _HttpcoreByteStream(httpx.AsyncByteStream):
    """Adapter from an httpcore response stream to httpx.AsyncByteStream.

    Bytes are never buffered past the size cap; callers iterate with
    `aiter_bytes()` and bound the consumer themselves. `aclose()` closes
    the underlying httpcore response via `aclose()`.
    """

    def __init__(self, core_response: Any) -> None:
        self._core_response = core_response
        self._closed = False

    async def __aiter__(self) -> Any:
        # httpcore 1.0.9 exposes the body via `aiter_stream()`. Yielding
        # here wires the httpcore response into httpx's streaming
        # consumer without ever buffering past the caller's chunk size.
        async for chunk in self._core_response.aiter_stream():
            yield chunk

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            await self._core_response.aclose()


class _HopTransport(httpx.AsyncBaseTransport):
    """Wraps a fresh httpcore.AsyncConnectionPool for one authorized hop.

    The pool is closed in all success, failure, timeout, and cancellation
    paths, via `aclose()`. A hop transport is never reused.
    """

    def __init__(
        self,
        *,
        pool: httpcore.AsyncConnectionPool,
        connect_timeout_s: float,
        read_timeout_s: float,
        write_timeout_s: float,
        pool_timeout_s: float,
    ) -> None:
        self._pool = pool
        self._timeouts: dict[str, float] = {
            "connect": connect_timeout_s,
            "read": read_timeout_s,
            "write": write_timeout_s,
            "pool": pool_timeout_s,
        }
        self._closed = False

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        if self._closed:
            raise RuntimeError("transport closed")
        # Pinned httpcore 1.0.9 contract:
        #   - construct `httpcore.Request` with the original hostname
        #     in the URL (TLS SNI + HTTP authority remain correct);
        #   - put the connect/read/write/pool timeouts in `extensions`;
        #   - call `pool.handle_async_request(...)` so the response is
        #     streamed lazily and closed by `aclose()` (we MUST NOT
        #     call `pool.request(...)` which eagerly reads/closes).
        core_request = httpcore.Request(
            method=request.method,
            url=str(request.url),
            headers=list(request.headers.raw),
            content=request.stream,
            extensions={**request.extensions, "timeout": dict(self._timeouts)},
        )
        try:
            core_response = await self._pool.handle_async_request(core_request)
        except _TLSCertificateError:
            # A TLS handshake failure is NOT a connectivity failure. It
            # carries diagnostic value (the server's TLS stack rejected
            # the handshake) that callers may want to surface separately
            # from a generic connect failure. We deliberately use
            # ``raise ... from None`` here too so the inner exception's
            # redaction chain remains severed.
            raise
        except httpcore.ConnectError:
            raise httpx.ConnectError("connect_failed") from None
        except httpcore.ConnectTimeout:
            raise httpx.ConnectTimeout("connect_timeout") from None
        except httpcore.ReadError:
            raise httpx.ReadError("read_failed") from None
        except httpcore.ReadTimeout:
            raise httpx.ReadTimeout("read_timeout") from None
        except httpcore.WriteError:
            raise httpx.WriteError("write_failed") from None
        except httpcore.WriteTimeout:
            raise httpx.WriteTimeout("write_timeout") from None
        except httpcore.PoolTimeout:
            raise httpx.PoolTimeout("pool_timeout") from None
        except httpcore.NetworkError:
            raise httpx.NetworkError("network_failed") from None
        except httpcore.RemoteProtocolError:
            raise httpx.RemoteProtocolError("remote_protocol_failed") from None
        except httpcore.LocalProtocolError:
            raise httpx.LocalProtocolError("local_protocol_failed") from None
        except httpcore.TimeoutException:
            raise httpx.TimeoutException("timeout") from None

        return httpx.Response(
            status_code=core_response.status,
            headers=core_response.headers,
            stream=_HttpcoreByteStream(core_response),
            extensions=core_response.extensions,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            await self._pool.aclose()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_and_authorize_url(
    url: str,
    policy: FetchPolicy,
) -> tuple[str, str, int]:
    """Return (scheme, canonical_host, port) or raise SafeFetchError.

    The input URL string is never included in the error.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        raise SafeFetchError(FetchErrorCode.INVALID_URL) from None
    if parts.scheme and parts.scheme not in policy.allowed_schemes:
        raise SafeFetchError(FetchErrorCode.SCHEME_DENIED)
    if not parts.scheme or not parts.netloc:
        raise SafeFetchError(FetchErrorCode.INVALID_URL)
    if parts.username is not None:
        raise SafeFetchError(FetchErrorCode.AUTHORITY_DENIED)
    # netloc might contain userinfo in pathological cases; reject.
    if "@" in parts.netloc:
        raise SafeFetchError(FetchErrorCode.AUTHORITY_DENIED)

    raw_authority = parts.netloc
    # Strip any trailing slash or path that snuck in.
    if "/" in raw_authority:
        raise SafeFetchError(FetchErrorCode.INVALID_URL)

    # Split host:port. urlsplit puts host:port into netloc.
    host_part = raw_authority
    port_part: str | None = None
    if raw_authority.startswith("["):
        # IPv6 literal: find matching ']'
        end = raw_authority.find("]")
        if end == -1:
            raise SafeFetchError(FetchErrorCode.INVALID_URL)
        host_part = raw_authority[1:end]
        rest = raw_authority[end + 1 :]
        if rest.startswith(":"):
            port_part = rest[1:]
        elif rest:
            raise SafeFetchError(FetchErrorCode.INVALID_URL)
    else:
        if ":" in raw_authority:
            host_part, port_part = raw_authority.rsplit(":", 1)
        else:
            host_part = raw_authority
            port_part = None

    # Port
    if port_part is None or port_part == "":
        port = policy.scheme_default_ports[parts.scheme]
    else:
        try:
            port = int(port_part)
        except ValueError:
            raise SafeFetchError(FetchErrorCode.PORT_DENIED) from None
        if port <= 0 or port > 65535:
            raise SafeFetchError(FetchErrorCode.PORT_DENIED)
        # Strict: only the default port per scheme is allowed.
        if port != policy.scheme_default_ports[parts.scheme]:
            raise SafeFetchError(FetchErrorCode.PORT_DENIED)

    try:
        canonical_host = canonicalize_hostname(host_part)
    except ValueError:
        # Anything the canonicalizer doesn't like is an authority denial.
        raise SafeFetchError(FetchErrorCode.AUTHORITY_DENIED) from None

    if not canonical_host:
        raise SafeFetchError(FetchErrorCode.AUTHORITY_DENIED)
    return parts.scheme, canonical_host, port


async def _safe_close(response: httpx.Response | None) -> None:
    """Close a response without raising. Safe to call on None."""
    if response is None:
        return
    with suppress(Exception):
        await response.aclose()


# ---------------------------------------------------------------------------
# Streaming consumer
# ---------------------------------------------------------------------------


async def _stream_within_cap(
    response: httpx.Response,
    cap: int,
    deadline: float,
    read_timeout_s: float,
) -> bytes:
    """Consume response body bytes without ever exceeding `cap`.

    Uses an explicit `anext` loop over `response.aiter_bytes()`. Before
    EACH `anext`, recomputes `remaining = deadline - loop.time()` and
    bounds the wait with `asyncio.timeout(min(read_timeout_s, remaining))`.
    Handles `StopAsyncIteration` as a normal end-of-body. Raises
    SafeFetchError(BODY_TOO_LARGE) when the cap would be exceeded.
    Never buffers the full body.
    """
    buf = bytearray()
    iterator = response.aiter_bytes()
    loop = asyncio.get_running_loop()
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise SafeFetchError(FetchErrorCode.OPERATION_TIMEOUT)
            try:
                async with asyncio.timeout(min(read_timeout_s, remaining)):
                    chunk = await anext(iterator)
            except StopAsyncIteration:
                break
            if chunk:
                if len(buf) + len(chunk) > cap:
                    raise SafeFetchError(FetchErrorCode.BODY_TOO_LARGE)
                buf.extend(chunk)
        return bytes(buf)
    finally:
        await _safe_close(response)


# ---------------------------------------------------------------------------
# Hop execution
# ---------------------------------------------------------------------------


async def _resolve_and_select(
    *,
    scheme: str,
    canonical_host: str,
    port: int,
    resolver: AddressResolver,
    timeout_s: float,
) -> AuthorizedHop:
    """Resolve the hostname and pick the first globally-routable address.

    If `canonical_host` is already a canonical IP literal (parsed by
    `ipaddress.ip_address`), it is used directly and the resolver is
    NOT consulted. The literal is still validated against the
    `_validate_address_for_destination` policy. DNS names go through
    the resolver.

    Fails closed if resolution is empty, contains a denied address, or
    yields no globally-routable candidate. No identifying data is
    surfaced in the resulting error.
    """
    target = f"{scheme}://{canonical_host}:{port}/"

    # Literal fast-path: when the canonical host is already an IP in
    # canonical numeric form, do not call the resolver. The literal
    # still must satisfy the global-routability policy.
    literal_ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        literal_ip = ipaddress.ip_address(canonical_host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        if not _validate_address_for_destination(literal_ip):
            raise SafeFetchError(FetchErrorCode.DESTINATION_DENIED)
        return AuthorizedHop(
            scheme=scheme,
            hostname=canonical_host,
            port=port,
            target=target,
            selected_address=literal_ip.compressed,
        )

    try:
        addresses = await resolver.resolve(canonical_host, timeout_s=timeout_s)
    except Exception:
        # Resolver must not raise, but if it does, treat as resolution failed.
        raise SafeFetchError(FetchErrorCode.RESOLUTION_FAILED) from None

    if not addresses:
        raise SafeFetchError(FetchErrorCode.RESOLUTION_FAILED)

    accepted: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for ip in addresses:
        if _validate_address_for_destination(ip):
            accepted.append(ip)

    if not accepted or len(accepted) != len(addresses):
        # If any returned address is denied, the entire destination fails.
        raise SafeFetchError(FetchErrorCode.DESTINATION_DENIED)

    accepted.sort(key=lambda a: (a.version, int(a)))
    selected = accepted[0]
    return AuthorizedHop(
        scheme=scheme,
        hostname=canonical_host,
        port=port,
        target=target,
        selected_address=selected.compressed,
    )


@dataclass
class _HopContext:
    """Owns the live HTTPX client (and therefore transport + pool) for one hop.

    Lifecycle: created by `_perform_hop` and returned alongside the
    response. The caller MUST close the response first and then this
    context. Closing this context closes the client, which closes the
    transport, which closes the HTTPCore pool. That ordering matters:
    if the pool is closed before the response, the body stream becomes
    invalid and `aiter_bytes()` raises.
    """

    client: httpx.AsyncClient
    transport: _HopTransport

    async def aclose(self) -> None:
        # Closing the client cascades: client → transport → pool.
        # Be defensive in every path so cancellation never leaks a pool.
        with suppress(Exception):
            await self.client.aclose()
        # Belt-and-suspenders: even if httpx is patched in a way that
        # doesn't propagate, close the transport (and therefore pool)
        # directly.
        with suppress(Exception):
            await self.transport.aclose()


def _is_ipv6_literal(host: str) -> bool:
    """True if the canonical hostname is an IPv6 literal (contains ':')."""
    return ":" in host


def _authority_for_url(host: str) -> str:
    """Return the URL authority component for the canonical hostname.

    DNS names are emitted unchanged. IPv6 literals are bracketed because
    URL syntax requires `[host]:port` to disambiguate colons inside the
    literal from the port separator.
    """
    if _is_ipv6_literal(host):
        return f"[{host}]"
    return host


async def _perform_hop(
    *,
    hop: AuthorizedHop,
    url_path_and_query: str,
    method: str,
    policy: FetchPolicy,
    deadline: float,
    send_accept_encoding_identity: bool,
) -> tuple[httpx.Response, _HopContext]:
    """Perform one authorized hop and return (response, context).

    The context owns the HTTPX client (and therefore transport + pool).
    It is returned alive so the caller can stream the response body
    before closing everything. On any failure inside this function the
    context is closed before the exception propagates.
    """
    pinned = _PinnedBackend(
        authorized_host=hop.hostname,
        authorized_port=hop.port,
        approved_address=hop.selected_address,
        connect_timeout_s=policy.connect_timeout_s,
    )
    pool = httpcore.AsyncConnectionPool(
        network_backend=pinned,
        http1=True,
        http2=False,
        retries=0,
        max_connections=1,
        max_keepalive_connections=0,
    )
    transport = _HopTransport(
        pool=pool,
        connect_timeout_s=policy.connect_timeout_s,
        read_timeout_s=policy.read_timeout_s,
        write_timeout_s=policy.write_timeout_s,
        pool_timeout_s=policy.pool_timeout_s,
    )
    # Build the URL. The hostname is preserved verbatim so TLS SNI,
    # certificate verification, and HTTP `Host` authority remain
    # correct. IPv6 literals are bracketed because URL syntax requires
    # `[host]:port` to disambiguate the colons inside the literal.
    host_authority = _authority_for_url(hop.hostname)
    full_url = f"{hop.scheme}://{host_authority}:{hop.port}{url_path_and_query}"
    headers: dict[str, str] = {
        "Host": f"{host_authority}:{hop.port}",
        "Accept": "*/*",
        "Connection": "close",
    }
    if send_accept_encoding_identity:
        headers["Accept-Encoding"] = "identity"

    loop = asyncio.get_running_loop()
    remaining = deadline - loop.time()
    if remaining <= 0:
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.OPERATION_TIMEOUT)

    # Honor distinct connect/read/write/pool timeouts; each is bounded
    # by `remaining` so a near-deadline operation cannot exceed the
    # total. These values are passed to HTTPX (which forwards them
    # to HTTPCore via the request `extensions` dict).
    connect_t = min(policy.connect_timeout_s, remaining)
    read_t = min(policy.read_timeout_s, remaining)
    write_t = min(policy.write_timeout_s, remaining)
    pool_t = min(policy.pool_timeout_s, remaining)

    client = httpx.AsyncClient(
        transport=transport,
        trust_env=False,
        timeout=httpx.Timeout(
            read_t,
            connect=connect_t,
            write=write_t,
            pool=pool_t,
        ),
    )

    try:
        request = client.build_request(method, full_url, headers=headers)
        # ``follow_redirects`` is explicitly disabled here even though the
        # HTTPX default is ``False``. This is a defense-in-depth guard:
        # if a future httpx release ever changes the default to ``True``,
        # the fetcher's own re-authorization state machine (which
        # resolves, validates and pins each redirected hop) will remain
        # the single source of truth for redirect policy.
        response = await client.send(request, stream=True, follow_redirects=False)
    except _TLSCertificateError:
        # TLS-handshake failures must map to the stable ``tls_failed``
        # code BEFORE any ``connect_failed`` mapper can swallow the
        # exception. The brief requires the typed exception to leave
        # the boundary as ``tls_failed`` deterministically. The
        # exception's string carries no underlying text; we use
        # ``from None`` to ensure the chained exception (which may
        # carry identifying details from the ssl/httpcore stack) does
        # not leak into our error or logs.
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.TLS_FAILED) from None
    except _AuthorizationMismatch:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.AUTHORIZATION_MISMATCH) from None
    except httpx.ConnectError:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.CONNECT_FAILED) from None
    except httpx.ConnectTimeout:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.CONNECT_FAILED) from None
    except httpx.ReadTimeout:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.OPERATION_TIMEOUT) from None
    except httpx.WriteTimeout:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.OPERATION_TIMEOUT) from None
    except httpx.PoolTimeout:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.OPERATION_TIMEOUT) from None
    except httpx.ReadError:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.CONNECT_FAILED) from None
    except httpx.WriteError:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.CONNECT_FAILED) from None
    except httpx.NetworkError:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.CONNECT_FAILED) from None
    except (
        httpx.RemoteProtocolError,
        httpx.LocalProtocolError,
        httpx.TimeoutException,
        httpx.HTTPError,
    ):
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.CONNECT_FAILED) from None
    except ssl.SSLError:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.TLS_FAILED) from None
    except TimeoutError:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.OPERATION_TIMEOUT) from None
    except asyncio.CancelledError:
        # Always release the pool on cancellation. The caller may not
        # have a chance to close anything if the loop is being torn down.
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise
    except Exception:
        with suppress(Exception):
            await client.aclose()
        with suppress(Exception):
            await transport.aclose()
        raise SafeFetchError(FetchErrorCode.INTERNAL_FAILURE) from None

    return response, _HopContext(client=client, transport=transport)


# ---------------------------------------------------------------------------
# Top-level safe_fetch
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeExternalFetcher:
    """The boundary. Construct once with a policy and resolver, reuse."""

    policy: FetchPolicy = field(default_factory=FetchPolicy)
    resolver: AddressResolver = field(default_factory=DualAddressResolver)

    async def fetch(self, url: str, *, method: str = "GET") -> FetchResult:
        return await safe_fetch(url, policy=self.policy, resolver=self.resolver, method=method)


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def _remaining(deadline: float) -> float:
    """Return remaining seconds until the monotonic deadline."""
    return deadline - asyncio.get_running_loop().time()


async def safe_fetch(
    url: str,
    *,
    policy: FetchPolicy | None = None,
    resolver: AddressResolver | None = None,
    method: str = "GET",
) -> FetchResult:
    """Perform a single fail-closed external GET (or other method).

    The URL string, hostname, resolved address, headers, body bytes, and
    any underlying exception text are never returned or logged.
    """
    if method != "GET":
        # The brief says GET only.
        raise SafeFetchError(FetchErrorCode.INVALID_URL)

    _policy = policy or FetchPolicy()
    _resolver = resolver or DualAddressResolver()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _policy.total_timeout_s

    # Defense in depth: even if an operation escapes our checks, the
    # whole state machine is enclosed by a wall-clock timeout.
    try:
        async with asyncio.timeout_at(deadline):
            return await _do_safe_fetch(url, _policy, _resolver, deadline)
    except TimeoutError:
        raise SafeFetchError(FetchErrorCode.OPERATION_TIMEOUT) from None
    except SafeFetchError:
        raise
    except Exception:
        raise SafeFetchError(FetchErrorCode.INTERNAL_FAILURE) from None


async def _do_safe_fetch(
    url: str,
    policy: FetchPolicy,
    resolver: AddressResolver,
    deadline: float,
) -> FetchResult:
    current_url = url
    redirects_used = 0
    # Resource handles for the in-flight hop. We hold them across the
    # redirect-or-stream decision so that nothing is closed before the
    # caller finishes with it.
    response: httpx.Response | None = None
    context: _HopContext | None = None

    async def _release_hop() -> None:
        """Close response (if any) then the context, in that order."""
        nonlocal response, context
        if response is not None:
            with suppress(Exception):
                await response.aclose()
            response = None
        if context is not None:
            with suppress(Exception):
                await context.aclose()
            context = None

    try:
        while True:
            # 1. Parse and authorize the URL (canonical hostname, allowed port).
            scheme, canonical_host, port = _parse_and_authorize_url(current_url, policy)

            # 2. Re-extract path/query so we preserve them across hops.
            parts = urlsplit(current_url)
            url_path_and_query = parts.path or "/"
            if parts.query:
                url_path_and_query = url_path_and_query + "?" + parts.query

            # 3. Resolve and pick the first globally-routable address.
            remaining = _remaining(deadline)
            if remaining <= 0:
                raise SafeFetchError(FetchErrorCode.OPERATION_TIMEOUT)
            resolve_timeout = min(policy.resolve_timeout_s, remaining)

            hop = await _resolve_and_select(
                scheme=scheme,
                canonical_host=canonical_host,
                port=port,
                resolver=resolver,
                timeout_s=resolve_timeout,
            )

            # 4. Perform the hop with its own pool / transport / client.
            #    The new ownership pattern returns a live context that
            #    we MUST close after the response, never before.
            try:
                response, context = await _perform_hop(
                    hop=hop,
                    url_path_and_query=url_path_and_query,
                    method="GET",
                    policy=policy,
                    deadline=deadline,
                    send_accept_encoding_identity=True,
                )
            except SafeFetchError:
                await _release_hop()
                raise

            status = response.status_code

            # 5. Redirect?
            if status in _REDIRECT_STATUSES:
                if redirects_used >= policy.max_redirects:
                    await _release_hop()
                    raise SafeFetchError(FetchErrorCode.REDIRECT_EXHAUSTED)
                location = _extract_location(response)
                await _release_hop()
                if location is None:
                    raise SafeFetchError(FetchErrorCode.REDIRECT_INVALID)
                # Reject protocol-relative targets ('//host/...').
                # They are authority-ambiguous: the host would silently
                # switch to a different scheme than the request came
                # from, and the next iteration of re-authorization is
                # not a substitute for explicit rejection here.
                if location.startswith("//"):
                    raise SafeFetchError(FetchErrorCode.REDIRECT_INVALID)
                try:
                    current_url = _absolutize(current_url, location)
                except ValueError:
                    raise SafeFetchError(FetchErrorCode.REDIRECT_INVALID) from None
                redirects_used += 1
                continue

            # 6. Final response. Content-Encoding policy.
            content_encoding = response.headers.get("content-encoding")
            if content_encoding is not None and content_encoding.lower() != "identity":
                await _release_hop()
                raise SafeFetchError(FetchErrorCode.ENCODING_DENIED)

            if not (200 <= status < 300):
                await _release_hop()
                raise SafeFetchError(FetchErrorCode.HTTP_STATUS_DENIED)

            media_type = response.headers.get("content-type", "application/octet-stream")
            # `_stream_within_cap` reads the body incrementally and
            # closes the response on its own. The context outlives the
            # body read, which is the only correct ordering.
            try:
                body = await _stream_within_cap(
                    response,
                    cap=policy.max_body_bytes,
                    deadline=deadline,
                    read_timeout_s=policy.read_timeout_s,
                )
            finally:
                # Whether the body read raised or not, the pool/client
                # must be released.
                if context is not None:
                    with suppress(Exception):
                        await context.aclose()
                    context = None
                    response = None
            return FetchResult(
                body=body,
                media_type=media_type,
                status=status,
                redirect_count=redirects_used,
            )
    except BaseException:
        # Catch CancelledError too so we never leak the pool/transport.
        await _release_hop()
        raise


def _extract_location(response: httpx.Response) -> str | None:
    """Return the redirect Location iff exactly one valid header is present."""
    values: list[str] = response.headers.get_list("location")
    if len(values) != 1:
        return None
    value = values[0].strip()
    if not value:
        return None
    return value


def _absolutize(base: str, target: str) -> str:
    """Resolve a possibly-relative redirect Location against the base URL.

    Calls :func:`urllib.parse.urljoin` with the current hop URL as the
    base and the redirect ``Location`` value as the reference. Behaviour:

    * absolute targets (``http://…``, ``https://…``) pass through unchanged;
    * path-only targets (``/path``) are resolved against the base origin;
    * relative targets (``path``) are resolved against the base URL.

    Protocol-relative targets (``//host/...``) are rejected earlier in the
    caller's state machine, so this helper does not need to special-case
    them. Any ``urljoin`` failure (e.g. control characters) is surfaced
    as ``ValueError`` so the caller can map it to
    ``FetchErrorCode.REDIRECT_INVALID``.
    """
    return urljoin(base, target)
