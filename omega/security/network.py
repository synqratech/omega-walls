"""Outbound URL validation with SSRF-oriented fail-closed defaults."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import socket
from typing import Iterable, Sequence
from urllib.parse import urlsplit


_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata",
    "metadata.google.internal",
}


@dataclass(frozen=True)
class OutboundURLPolicy:
    allowed_schemes: Sequence[str] = ("https",)
    allowed_hosts: Sequence[str] = ()
    allowed_ports: Sequence[int] = (443,)
    allow_ip_literals: bool = False
    resolve_dns: bool = True


def _normalise_host(value: str) -> str:
    return str(value or "").strip().rstrip(".").lower()


def _host_allowed(host: str, allowed_hosts: Iterable[str]) -> bool:
    rules = [_normalise_host(x) for x in allowed_hosts if _normalise_host(x)]
    if not rules:
        return False
    for rule in rules:
        if rule.startswith("*."):
            suffix = rule[1:]  # includes leading dot
            if host.endswith(suffix) and host != suffix.lstrip("."):
                return True
        elif host == rule:
            return True
    return False


def _ip_is_public(ip: ipaddress._BaseAddress) -> bool:  # type: ignore[attr-defined]
    return bool(
        ip.is_global
        and not ip.is_private
        and not ip.is_loopback
        and not ip.is_link_local
        and not ip.is_multicast
        and not ip.is_reserved
        and not ip.is_unspecified
    )


def validate_outbound_url(url: str, *, policy: OutboundURLPolicy) -> tuple[str, tuple[str, ...]]:
    """Validate an outbound URL and return ``(hostname, resolved_ips)``.

    The call is intentionally fail closed: an empty host allowlist, DNS failure,
    userinfo, fragments, private/reserved addresses, or a disallowed port all fail.
    Callers should re-run this immediately before connecting and must disable
    automatic redirects, validating every redirect target separately if redirects
    are ever enabled.
    """

    raw = str(url or "").strip()
    parsed = urlsplit(raw)
    scheme = str(parsed.scheme or "").lower()
    allowed_schemes = {str(x).strip().lower() for x in policy.allowed_schemes}
    if scheme not in allowed_schemes:
        raise ValueError("outbound URL scheme is not allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("outbound URL userinfo is forbidden")
    if parsed.fragment:
        raise ValueError("outbound URL fragments are forbidden")
    host = _normalise_host(parsed.hostname or "")
    if not host or host in _BLOCKED_HOSTNAMES or host.endswith(".localhost"):
        raise ValueError("outbound URL hostname is forbidden")
    if not _host_allowed(host, policy.allowed_hosts):
        raise ValueError("outbound URL host is not allowlisted")

    default_port = 443 if scheme == "https" else 80
    try:
        port = int(parsed.port or default_port)
    except ValueError as exc:
        raise ValueError("outbound URL port is invalid") from exc
    if port not in {int(x) for x in policy.allowed_ports}:
        raise ValueError("outbound URL port is not allowed")

    resolved: list[str] = []
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not policy.allow_ip_literals:
            raise ValueError("outbound URL IP literals are forbidden")
        if not _ip_is_public(literal):
            raise ValueError("outbound URL points to a non-public address")
        resolved.append(str(literal))
    elif policy.resolve_dns:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError("outbound URL DNS resolution failed") from exc
        for info in infos:
            address = str(info[4][0])
            ip = ipaddress.ip_address(address)
            if not _ip_is_public(ip):
                raise ValueError("outbound URL resolves to a non-public address")
            if address not in resolved:
                resolved.append(address)
        if not resolved:
            raise ValueError("outbound URL resolved to no addresses")
    return host, tuple(resolved)
