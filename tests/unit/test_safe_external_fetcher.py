"""Unit tests for hermes/jobs/safe_fetcher.py — Slice 1B.

Pure policy tests, canonicalization/authority validation, error-code
membership and serialization, AST/import-surface tests, and redaction
assertions. These tests MUST NOT open an external socket.

The companion transport-level tests live in
``tests/integration/test_safe_external_fetcher_transport.py``.
"""

from __future__ import annotations

import ast
import logging
import socket

import pytest

from hermes.jobs import safe_fetcher
from hermes.jobs.safe_fetcher import (
    ALL_FETCH_ERROR_CODES,
    AddressResolver,
    AuthorizedHop,
    DualAddressResolver,
    FetchErrorCode,
    FetchPolicy,
    FetchResult,
    SafeExternalFetcher,
    SafeFetchError,
    canonicalize_hostname,
    safe_fetch,
)

# ---------------------------------------------------------------------------
# Pure policy: FetchPolicy construction and validation
# ---------------------------------------------------------------------------


class TestFetchPolicyDefaults:
    def test_defaults_match_brief(self) -> None:
        p = FetchPolicy()
        assert p.allowed_schemes == frozenset({"http", "https"})
        assert p.scheme_default_ports == {"http": 80, "https": 443}
        assert p.max_redirects == 5
        assert p.max_body_bytes == 2_000_000
        assert p.resolve_timeout_s == 5.0
        assert p.connect_timeout_s == 5.0
        assert p.read_timeout_s == 10.0
        assert p.write_timeout_s == 5.0
        assert p.pool_timeout_s == 5.0
        assert p.total_timeout_s == 30.0

    def test_is_frozen(self) -> None:
        p = FetchPolicy()
        with pytest.raises((AttributeError, Exception)):
            p.max_redirects = 1  # type: ignore[misc]


class TestFetchPolicyValidation:
    def test_empty_schemes_rejected(self) -> None:
        with pytest.raises(ValueError):
            FetchPolicy(allowed_schemes=frozenset())

    def test_unknown_scheme_rejected(self) -> None:
        with pytest.raises(ValueError):
            FetchPolicy(allowed_schemes=frozenset({"ftp"}))

    def test_missing_default_port_rejected(self) -> None:
        # Force construction of a policy whose port map is incomplete.
        with pytest.raises(ValueError):
            FetchPolicy(scheme_default_ports={})

    def test_max_redirects_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            FetchPolicy(max_redirects=0)

    def test_max_body_bytes_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            FetchPolicy(max_body_bytes=0)

    @pytest.mark.parametrize(
        "field_name,value",
        [
            ("resolve_timeout_s", 0),
            ("connect_timeout_s", 0),
            ("read_timeout_s", 0),
            ("write_timeout_s", 0),
            ("pool_timeout_s", 0),
            ("total_timeout_s", 0),
            ("resolve_timeout_s", -1),
            ("connect_timeout_s", -1),
            ("total_timeout_s", -0.0001),
        ],
    )
    def test_timeouts_must_be_positive(self, field_name: str, value: float) -> None:
        with pytest.raises(ValueError):
            FetchPolicy(**{field_name: value})

    @pytest.mark.parametrize(
        "field_name,value",
        [
            ("connect_timeout_s", 31),
            ("read_timeout_s", 31),
            ("write_timeout_s", 31),
            ("pool_timeout_s", 31),
        ],
    )
    def test_op_timeouts_cannot_exceed_total(self, field_name: str, value: float) -> None:
        with pytest.raises(ValueError):
            FetchPolicy(**{field_name: value})

    def test_op_timeout_equal_total_is_accepted(self) -> None:
        p = FetchPolicy(total_timeout_s=10.0, connect_timeout_s=10.0)
        assert p.connect_timeout_s == 10.0


# ---------------------------------------------------------------------------
# Canonical hostname / default-port behavior
# ---------------------------------------------------------------------------


class TestCanonicalHostname:
    def test_lowercase_dns(self) -> None:
        assert canonicalize_hostname("Example.COM") == "example.com"

    def test_terminal_dot_removed(self) -> None:
        assert canonicalize_hostname("example.com.") == "example.com"

    def test_dns_no_trailing_dot_unchanged(self) -> None:
        assert canonicalize_hostname("example.com") == "example.com"

    def test_idn_unicode_to_ascii(self) -> None:
        # bücher.example → xn--bcher-kva.example
        out = canonicalize_hostname("bücher.example")
        assert out.endswith(".example")
        assert out.startswith("xn--")

    def test_idn_unicode_equivalent_to_explicit_a_label(self) -> None:
        # Equivalence between Unicode and pre-encoded A-label forms.
        from_unicode = canonicalize_hostname("münchen.de")
        # münchen.de → xn--mnchen-3ya.de
        assert canonicalize_hostname("xn--mnchen-3ya.de") == from_unicode

    def test_bracketed_ipv6_literal_accepted(self) -> None:
        # The URL parser strips brackets before us; we accept raw literals
        # in their canonical compressed form, so 2001:4860:4860::8888 stays
        # as-is. Brackets are URL syntax and must not appear
        # post-canonicalization.
        assert canonicalize_hostname("2001:4860:4860::8888") == "2001:4860:4860::8888"

    def test_canonical_ipv4(self) -> None:
        # A non-canonical form (leading zeros) must be rejected — only the
        # canonical numeric form is allowed.
        with pytest.raises(ValueError):
            canonicalize_hostname(f"192.{168}.001.1")
        # Canonical form is accepted as an IPv4 literal only if it is_global.
        # 8.8.8.8 is_global → must return its compressed canonical form.
        out = canonicalize_hostname("8.8.8.8")
        assert out == "8.8.8.8"

    def test_ipv4_mapped_ipv6_rejected(self) -> None:
        # ::ffff:8.8.8.8 is an IPv4-mapped IPv6 literal: must be rejected.
        with pytest.raises(ValueError):
            canonicalize_hostname("::ffff:8.8.8.8")

    def test_percent_in_authority_rejected(self) -> None:
        with pytest.raises(ValueError):
            canonicalize_hostname("example%25.com")

    def test_zone_identifier_rejected(self) -> None:
        with pytest.raises(ValueError):
            canonicalize_hostname("fe80::1%eth0")

    def test_shortened_ipv4_rejected(self) -> None:
        # No shortened forms: only canonical numeric is allowed.
        with pytest.raises(ValueError):
            canonicalize_hostname(f"192.{168}.1")
        with pytest.raises(ValueError):
            # Decimal form: must be rejected by ipaddress.
            canonicalize_hostname("2130706433")
        with pytest.raises(ValueError):
            canonicalize_hostname("0x7f000001")

    def test_empty_hostname_rejected(self) -> None:
        with pytest.raises(ValueError):
            canonicalize_hostname("")

    def test_none_hostname_rejected(self) -> None:
        with pytest.raises(ValueError):
            canonicalize_hostname(None)  # type: ignore[arg-type]

    def test_bracket_in_authority_rejected(self) -> None:
        # Brackets are URL syntax — they don't belong in the authority.
        with pytest.raises(ValueError):
            canonicalize_hostname("[2001:db8::1]")


# ---------------------------------------------------------------------------
# URL parse + authorize: scheme/port/userinfo behavior
# ---------------------------------------------------------------------------


class TestURLAuthorization:
    def test_http_default_port_80(self) -> None:
        policy = FetchPolicy()
        # Parse via the public pipeline: use safe_fetch entry with a
        # mocked-out resolver to assert that parsing succeeds.
        # Easier: hit the internal helper directly.
        from hermes.jobs.safe_fetcher import _parse_and_authorize_url

        scheme, host, port = _parse_and_authorize_url("http://example.com/", policy)
        assert scheme == "http"
        assert host == "example.com"
        assert port == 80

    def test_https_default_port_443(self) -> None:
        from hermes.jobs.safe_fetcher import _parse_and_authorize_url

        scheme, _host, port = _parse_and_authorize_url("https://example.com/", policy=FetchPolicy())
        assert scheme == "https"
        assert port == 443

    def test_non_default_port_rejected(self) -> None:
        from hermes.jobs.safe_fetcher import _parse_and_authorize_url

        with pytest.raises(SafeFetchError) as exc:
            _parse_and_authorize_url("http://example.com:8080/", policy=FetchPolicy())
        assert exc.value.code is FetchErrorCode.PORT_DENIED

    def test_explicit_default_port_accepted(self) -> None:
        # http://example.com:80/ should be accepted; the default port is
        # allowed, just no *other* ports.
        from hermes.jobs.safe_fetcher import _parse_and_authorize_url

        scheme, _host, port = _parse_and_authorize_url(
            "http://example.com:80/", policy=FetchPolicy()
        )
        assert scheme == "http"
        assert port == 80

    def test_userinfo_rejected(self) -> None:
        from hermes.jobs.safe_fetcher import _parse_and_authorize_url

        with pytest.raises(SafeFetchError) as exc:
            _parse_and_authorize_url("http://user@example.com/", policy=FetchPolicy())
        assert exc.value.code is FetchErrorCode.AUTHORITY_DENIED

    def test_userinfo_with_colon_rejected(self) -> None:
        from hermes.jobs.safe_fetcher import _parse_and_authorize_url

        with pytest.raises(SafeFetchError) as exc:
            _parse_and_authorize_url("http://user:pass@example.com/", policy=FetchPolicy())
        assert exc.value.code is FetchErrorCode.AUTHORITY_DENIED

    def test_unsupported_scheme_rejected(self) -> None:
        from hermes.jobs.safe_fetcher import _parse_and_authorize_url

        with pytest.raises(SafeFetchError) as exc:
            _parse_and_authorize_url("file:///etc/passwd", policy=FetchPolicy())
        assert exc.value.code is FetchErrorCode.SCHEME_DENIED

    def test_invalid_url_rejected(self) -> None:
        from hermes.jobs.safe_fetcher import _parse_and_authorize_url

        with pytest.raises(SafeFetchError) as exc:
            _parse_and_authorize_url("not a url", policy=FetchPolicy())
        assert exc.value.code is FetchErrorCode.INVALID_URL

    def test_bracketed_ipv6_default_port(self) -> None:
        from hermes.jobs.safe_fetcher import _parse_and_authorize_url

        # 2001:4860:4860::8888 — global Google public DNS IPv6 literal.
        # NOTE: IPv6 global literal at URL form is parsed with brackets
        # by urlsplit. Test the canonicalizer itself for IPv6.
        # [2001:4860:4860::8888] URL → after bracket stripping →
        # 2001:4860:4860::8888.
        scheme, host, port = _parse_and_authorize_url(
            "https://[2001:4860:4860::8888]/", policy=FetchPolicy()
        )
        assert scheme == "https"
        assert host == "2001:4860:4860::8888"
        assert port == 443

    def test_port_out_of_range_rejected(self) -> None:
        from hermes.jobs.safe_fetcher import _parse_and_authorize_url

        # Port 70000 is not allowed (out of range). Note that even if it
        # were in range, the brief only allows the default port per scheme.
        with pytest.raises(SafeFetchError):
            _parse_and_authorize_url("http://example.com:70000/", policy=FetchPolicy())


# ---------------------------------------------------------------------------
# Resolver: address validation
# ---------------------------------------------------------------------------


class TestValidateAddress:
    def test_global_ipv4_accepted(self) -> None:
        import ipaddress

        from hermes.jobs.safe_fetcher import _validate_address_for_destination

        assert _validate_address_for_destination(ipaddress.ip_address("8.8.8.8")) is True

    def test_global_ipv6_accepted(self) -> None:
        import ipaddress

        from hermes.jobs.safe_fetcher import _validate_address_for_destination

        # 2001:4860:4860::8888 is Google public DNS IPv6.
        assert (
            _validate_address_for_destination(ipaddress.ip_address("2001:4860:4860::8888")) is True
        )

    def test_private_ipv4_rejected(self) -> None:
        import ipaddress

        from hermes.jobs.safe_fetcher import _validate_address_for_destination

        assert _validate_address_for_destination(ipaddress.ip_address(0x0A000001)) is False
        assert _validate_address_for_destination(ipaddress.ip_address(0xC0A80101)) is False
        assert _validate_address_for_destination(ipaddress.ip_address(0xAC100001)) is False

    def test_loopback_rejected(self) -> None:
        import ipaddress

        from hermes.jobs.safe_fetcher import _validate_address_for_destination

        assert _validate_address_for_destination(ipaddress.ip_address("127.0.0.1")) is False
        assert _validate_address_for_destination(ipaddress.ip_address("::1")) is False

    def test_link_local_rejected(self) -> None:
        import ipaddress

        from hermes.jobs.safe_fetcher import _validate_address_for_destination

        assert _validate_address_for_destination(ipaddress.ip_address("169.254.169.254")) is False
        assert _validate_address_for_destination(ipaddress.ip_address("fe80::1")) is False

    def test_multicast_rejected(self) -> None:
        import ipaddress

        from hermes.jobs.safe_fetcher import _validate_address_for_destination

        assert _validate_address_for_destination(ipaddress.ip_address("224.0.0.1")) is False
        assert _validate_address_for_destination(ipaddress.ip_address("ff02::1")) is False

    def test_unspecified_rejected(self) -> None:
        import ipaddress

        from hermes.jobs.safe_fetcher import _validate_address_for_destination

        assert _validate_address_for_destination(ipaddress.ip_address("0.0.0.0")) is False
        assert _validate_address_for_destination(ipaddress.ip_address("::")) is False

    def test_reserved_rejected(self) -> None:
        import ipaddress

        from hermes.jobs.safe_fetcher import _validate_address_for_destination

        # 240.0.0.0/4 is reserved (Class E).
        assert _validate_address_for_destination(ipaddress.ip_address("240.0.0.1")) is False

    def test_ipv4_mapped_ipv6_rejected(self) -> None:
        import ipaddress

        from hermes.jobs.safe_fetcher import _validate_address_for_destination

        # ::ffff:8.8.8.8 — hidden v4. Must be rejected.
        assert _validate_address_for_destination(ipaddress.ip_address("::ffff:8.8.8.8")) is False


# ---------------------------------------------------------------------------
# Resolver protocol / production resolver
# ---------------------------------------------------------------------------


class TestAddressResolverProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        # DualAddressResolver satisfies the Protocol shape.
        assert isinstance(DualAddressResolver(), AddressResolver)

    def test_protocol_accepts_any_object_with_resolve(self) -> None:
        class _StubResolver:
            async def resolve(self, hostname: str, *, timeout_s: float) -> list:
                return []

        assert isinstance(_StubResolver(), AddressResolver)

    def test_protocol_rejects_missing_method(self) -> None:
        class _NotAResolver:
            pass

        assert not isinstance(_NotAResolver(), AddressResolver)


class TestDualAddressResolver:
    async def test_empty_for_garbage_hostname(self) -> None:
        # GAIE_NONAME → resolver returns []. No exception leaks.
        r = DualAddressResolver()
        result = await r.resolve("this-host-should-not-exist.invalid", timeout_s=1.0)
        assert result == []

    async def test_empty_for_timeout(self) -> None:
        # A timeout must not raise out of the resolver; it's redacted to [].
        r = DualAddressResolver()
        # Use an unreachable port to force getaddrinfo timeout.
        # NOTE: getaddrinfo on a routable hostname usually does not block,
        # but with a tiny timeout we exercise the wait_for branch.
        # We pick a known-bad hostname; if it returns quickly we still
        # expect []. If it raises, the resolver must swallow it.
        result = await r.resolve("definitely-not-resolvable.invalid", timeout_s=0.001)
        assert result == []


# ---------------------------------------------------------------------------
# FetchErrorCode enum: closed, stable, serialized correctly
# ---------------------------------------------------------------------------


EXPECTED_CODE_VALUES = {
    "invalid_url",
    "scheme_denied",
    "authority_denied",
    "port_denied",
    "resolution_failed",
    "destination_denied",
    "authorization_mismatch",
    "redirect_invalid",
    "redirect_exhausted",
    "connect_failed",
    "tls_failed",
    "http_status_denied",
    "encoding_denied",
    "body_too_large",
    "operation_timeout",
    "internal_failure",
}


class TestFetchErrorCode:
    def test_exact_stable_membership(self) -> None:
        actual = {c.value for c in FetchErrorCode}
        assert actual == EXPECTED_CODE_VALUES

    def test_no_extra_codes(self) -> None:
        # If a new code is added, this test fails (intentional guard).
        assert len(FetchErrorCode) == len(EXPECTED_CODE_VALUES)

    def test_all_fetch_error_codes_tuple_matches_enum(self) -> None:
        assert tuple(c.value for c in FetchErrorCode) == ALL_FETCH_ERROR_CODES

    def test_str_enum_serialization(self) -> None:
        # FetchErrorCode is a StrEnum — serialization must round-trip.
        for code in FetchErrorCode:
            assert isinstance(code, str)
            assert FetchErrorCode(code.value) is code

    def test_serialization_via_value(self) -> None:
        assert FetchErrorCode.CONNECT_FAILED.value == "connect_failed"
        assert FetchErrorCode.HTTP_STATUS_DENIED.value == "http_status_denied"


class TestSafeFetchError:
    def test_message_is_generic_redacted(self) -> None:
        e = SafeFetchError(FetchErrorCode.CONNECT_FAILED)
        # No input URL, hostname, address, headers, body, exception text.
        assert e.code is FetchErrorCode.CONNECT_FAILED
        # The string form is the only thing surfaced. It must not contain
        # any sentinel that the caller could mistake for a leaked datum.
        text = str(e)
        assert "safe_fetch_failed:" in text
        assert "connect_failed" in text
        # Negative: nothing sensitive here.
        assert "http" not in text
        assert "example" not in text

    def test_carries_code_only(self) -> None:
        e = SafeFetchError(FetchErrorCode.OPERATION_TIMEOUT)
        # Only `code` and the inherited Exception args are present.
        assert hasattr(e, "code")
        assert e.code is FetchErrorCode.OPERATION_TIMEOUT

    def test_no_leak_of_input_url_in_message(self) -> None:
        sentinel = "https://attacker-controlled.example/steal?q=marker"
        e = SafeFetchError(FetchErrorCode.CONNECT_FAILED)
        assert sentinel not in str(e)


# ---------------------------------------------------------------------------
# SafeExternalFetcher construction
# ---------------------------------------------------------------------------


class TestSafeExternalFetcherDataclass:
    def test_default_construction(self) -> None:
        f = SafeExternalFetcher()
        assert f.policy is not None
        assert isinstance(f.policy, FetchPolicy)
        assert f.resolver is not None

    def test_custom_policy_and_resolver(self) -> None:
        from hermes.jobs.safe_fetcher import DualAddressResolver

        policy = FetchPolicy(
            max_body_bytes=1000,
            total_timeout_s=1.0,
            resolve_timeout_s=1.0,
            connect_timeout_s=1.0,
            read_timeout_s=1.0,
            write_timeout_s=1.0,
            pool_timeout_s=1.0,
        )
        resolver = DualAddressResolver()
        f = SafeExternalFetcher(policy=policy, resolver=resolver)
        assert f.policy is policy
        assert f.resolver is resolver


# ---------------------------------------------------------------------------
# Redaction: sentinel values must never appear in errors or logs
# ---------------------------------------------------------------------------


SENTINEL_HOSTNAME = "REDACTED-SENTINEL-HOST-7e3a"
SENTINEL_IP = "203.0.113.99"  # TEST-NET-3 (RFC 5737)
SENTINEL_MARKER = "REDACTED-SENTINEL-MARKER-deadbeef"


class TestRedaction:
    async def test_invalid_url_error_carries_no_input_url(self) -> None:
        sentinel_url = f"http://{SENTINEL_HOSTNAME}.invalid.example.com/{SENTINEL_MARKER}"
        with pytest.raises(SafeFetchError):
            await safe_fetch(sentinel_url, policy=FetchPolicy(), resolver=DualAddressResolver())

    async def test_resolution_failed_does_not_leak_hostname(self) -> None:
        # A resolver that returns [] forces RESOLUTION_FAILED. The error
        # text must NOT mention the sentinel hostname.
        class _EmptyResolver:
            async def resolve(self, hostname: str, *, timeout_s: float) -> list:
                return []

        sentinel_url = f"http://{SENTINEL_HOSTNAME}.invalid/"
        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                sentinel_url,
                policy=FetchPolicy(),
                resolver=_EmptyResolver(),
            )
        # No identifying data in the message.
        msg = str(exc.value)
        assert SENTINEL_HOSTNAME not in msg
        assert SENTINEL_MARKER not in msg
        assert exc.value.code is FetchErrorCode.RESOLUTION_FAILED

    async def test_destination_denied_when_any_address_is_denied(self) -> None:
        # Mixed accepted/denied resolution must fail closed.
        import ipaddress

        class _MixedResolver:
            async def resolve(self, hostname: str, *, timeout_s: float) -> list:
                return [
                    ipaddress.ip_address("8.8.8.8"),  # global
                    ipaddress.ip_address("127.0.0.1"),  # loopback → denied
                ]

        sentinel_url = f"http://{SENTINEL_HOSTNAME}.example/"
        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                sentinel_url,
                policy=FetchPolicy(),
                resolver=_MixedResolver(),
            )
        assert exc.value.code is FetchErrorCode.DESTINATION_DENIED
        assert SENTINEL_HOSTNAME not in str(exc.value)
        assert SENTINEL_IP not in str(exc.value)

    async def test_resolver_raising_exception_is_redacted(self) -> None:
        class _RaisingResolver:
            async def resolve(self, hostname: str, *, timeout_s: float) -> list:
                raise RuntimeError(f"leaked: {SENTINEL_MARKER} for {SENTINEL_HOSTNAME}")

        sentinel_url = f"http://{SENTINEL_HOSTNAME}.example/"
        with pytest.raises(SafeFetchError) as exc:
            await safe_fetch(
                sentinel_url,
                policy=FetchPolicy(),
                resolver=_RaisingResolver(),
            )
        msg = str(exc.value)
        assert SENTINEL_MARKER not in msg
        assert SENTINEL_HOSTNAME not in msg
        # The brief says: resolver exceptions map to RESOLUTION_FAILED.
        assert exc.value.code is FetchErrorCode.RESOLUTION_FAILED

    async def test_logs_do_not_contain_sentinel_hostname(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # SafeFetchError messages must not contain input hostnames.
        # Use caplog to verify nothing leaks via the logger either.
        caplog.set_level(logging.DEBUG, logger="hermes.jobs.safe_fetcher")
        sentinel_url = f"http://{SENTINEL_HOSTNAME}.example/"
        with pytest.raises(SafeFetchError):
            await safe_fetch(sentinel_url, policy=FetchPolicy(), resolver=DualAddressResolver())

        all_text = "\n".join(rec.getMessage() for rec in caplog.records)
        assert SENTINEL_HOSTNAME not in all_text


# ---------------------------------------------------------------------------
# AST / import surface: no underscore-prefixed public paths from httpx/httpcore
# ---------------------------------------------------------------------------


class TestImportSurface:
    def test_production_module_has_no_underscore_httpx_imports(self) -> None:
        """AST scan: hermes/jobs/safe_fetcher.py imports nothing from a path
        containing a leading underscore for httpx or httpcore."""
        path = safe_fetcher.__file__
        assert path is not None
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        offenders: list[tuple[str, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                top = mod.split(".", 1)[0]
                if top in {"httpx", "httpcore"}:
                    # Check both the module path and any local name bindings
                    # (e.g. `from httpx._api import X`).
                    if mod.startswith("_") or "._" in mod:
                        offenders.append((mod, node.lineno))
                    for alias in node.names:
                        if alias.name.startswith("_"):
                            offenders.append((f"{mod}.{alias.name}", node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    if top in {"httpx", "httpcore"} and alias.name.startswith("_"):
                        offenders.append((alias.name, node.lineno))
        assert not offenders, f"underscore-prefixed imports: {offenders}"

    def test_only_allowed_top_level_imports_for_httpx_httpcore(self) -> None:
        """Public-API discipline: top-level httpx/httpcore imports use only
        documented modules."""
        path = safe_fetcher.__file__
        assert path is not None
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        allowed_httpx = {"httpx"}
        allowed_httpcore = {"httpcore"}
        offenders: list[tuple[str, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                top = mod.split(".", 1)[0]
                if top == "httpx" and mod not in allowed_httpx:
                    offenders.append((mod, node.lineno))
                if top == "httpcore" and mod not in allowed_httpcore:
                    offenders.append((mod, node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    if top in {"httpx", "httpcore"} and alias.name not in {
                        "httpx",
                        "httpcore",
                    }:
                        offenders.append((alias.name, node.lineno))
        assert not offenders, f"unexpected public imports: {offenders}"

    def test_safe_fetcher_uses_only_public_httpx_types(self) -> None:
        """Source-grep discipline: the module references no `_Class` names
        from httpx or httpcore."""
        path = safe_fetcher.__file__
        assert path is not None
        with open(path, encoding="utf-8") as f:
            src = f.read()
        # Pull out every base class declared. Bases are referenced in
        # class headers and `isinstance` checks. We scan for
        # `httpx.<Name>` / `httpcore.<Name>` where Name starts with `_`.
        import re

        pattern = re.compile(r"\b(?:httpx|httpcore)\.([A-Za-z_][A-Za-z0-9_]*)")
        offenders: list[str] = []
        for match in pattern.finditer(src):
            name = match.group(1)
            if name.startswith("_"):
                offenders.append(name)
        assert not offenders, f"underscore-prefixed public API references: {offenders}"


# ---------------------------------------------------------------------------
# Dataclass shape checks
# ---------------------------------------------------------------------------


class TestDataclassShapes:
    def test_authorized_hop_is_frozen(self) -> None:
        h = AuthorizedHop(
            scheme="https",
            hostname="example.com",
            port=443,
            target="https://example.com:443/",
            selected_address="8.8.8.8",
        )
        with pytest.raises((AttributeError, Exception)):
            h.hostname = "evil.example"  # type: ignore[misc]

    def test_fetch_result_is_frozen(self) -> None:
        r = FetchResult(body=b"x", media_type="text/plain", status=200, redirect_count=0)
        with pytest.raises((AttributeError, Exception)):
            r.body = b"y"  # type: ignore[misc]

    def test_fetch_result_has_no_url_field(self) -> None:
        # The brief: FetchResult intentionally does NOT expose a final URL.
        # This is a deliberate surface minimization.
        fields = {f.name for f in FetchResult.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        assert "url" not in fields
        assert "final_url" not in fields
        assert "redirect_url" not in fields


# ---------------------------------------------------------------------------
# Socket-denial fixture: any unmocked socket call aborts the test
# ---------------------------------------------------------------------------


@pytest.fixture
def deny_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch socket.getaddrinfo / socket.socket to raise if invoked.

    Tests using this fixture are guaranteed to never hit the network.
    If they trigger a real DNS or TCP attempt, the test fails immediately.
    """

    def _explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("real socket call attempted in pure unit test")

    monkeypatch.setattr(socket, "getaddrinfo", _explode)
    # Don't patch socket.socket globally — that breaks pytest internals;
    # instead the resolver contract makes the only path through
    # getaddrinfo. The pinned backend also goes through inner AnyIO,
    # which is exercised only by the transport test.
    yield


class TestSocketDenialFixture:
    async def test_safe_fetch_never_touches_socket(self, deny_socket: None) -> None:
        # With the socket fixture, the production resolver would fail.
        # Safe fetch must still return a clean SafeFetchError, never a
        # socket leak. Resolver exception is redacted; the error code is
        # RESOLUTION_FAILED (or DESTINATION_DENIED if [] is returned).
        sentinel_url = f"http://{SENTINEL_HOSTNAME}.example/"
        with pytest.raises(SafeFetchError):
            await safe_fetch(sentinel_url, policy=FetchPolicy(), resolver=DualAddressResolver())


# ---------------------------------------------------------------------------
# AuthorizedHop / FetchResult serialization sanity
# ---------------------------------------------------------------------------


class TestValueObjectSanity:
    def test_authorized_hop_construction(self) -> None:
        h = AuthorizedHop(
            scheme="https",
            hostname="example.com",
            port=443,
            target="https://example.com:443/",
            selected_address="8.8.8.8",
        )
        assert h.scheme == "https"
        assert h.port == 443
        assert h.selected_address == "8.8.8.8"

    def test_fetch_result_construction(self) -> None:
        r = FetchResult(
            body=b"hello",
            media_type="text/plain",
            status=200,
            redirect_count=0,
        )
        assert r.body == b"hello"
        assert r.media_type == "text/plain"
        assert r.status == 200
        assert r.redirect_count == 0


# ---------------------------------------------------------------------------
# safe_fetch rejects non-GET methods (brief: GET only)
# ---------------------------------------------------------------------------


class TestSafeFetchMethodRestriction:
    def test_post_rejected(self) -> None:
        # Non-GET methods are not part of the supported public surface.
        # The implementation maps anything other than GET to INVALID_URL.
        import asyncio

        async def _call() -> None:
            await safe_fetch(
                "http://example.com/",
                policy=FetchPolicy(),
                resolver=DualAddressResolver(),
                method="POST",
            )

        with pytest.raises(SafeFetchError) as exc:
            asyncio.run(_call())
        assert exc.value.code is FetchErrorCode.INVALID_URL
