"""
CloudSleuth Website Health Monitor Module.
Performs external, passive health checks on publicly accessible websites.

Security Design:
- Only HTTP/HTTPS schemes allowed.
- SSRF protection: blocks private IPs, localhost, link-local, cloud metadata endpoints, CGNAT, and decimal IP representations.
- Every resolved IP (including after redirects) is validated before connection.
- Strict timeouts, response-size caps (1 MB), and per-session sliding-window rate limiting.
- Disallows non-HTTP ports to prevent port scanning or protocol abuse.
- No scanning, attacking, or stress-testing. Read-only single requests only.

Scope:
- Observable external symptoms only (HTTP status, response time, DNS, SSL, timeouts).
- Does NOT claim access to internal metrics, logs, or traces.
- OpenTelemetry integration is supported as an optional future enhancement.
"""

import re
import ssl
import time
import socket
import ipaddress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse, urljoin

import requests
import requests.adapters


# ─────────────────────────────────────────────────────────────
# Constants & Security Configuration
# ─────────────────────────────────────────────────────────────
MAX_RESPONSE_BYTES = 1_048_576  # 1 MB cap on response body
CONNECT_TIMEOUT = 8             # seconds
READ_TIMEOUT = 12               # seconds
MAX_REDIRECTS = 5
RATE_LIMIT_WINDOW = 60          # seconds
RATE_LIMIT_MAX_CHECKS = 10     # max checks per sliding window

# Cloud metadata endpoints to block (AWS, GCP, Azure, DigitalOcean, Alibaba, etc.)
BLOCKED_HOSTS = frozenset([
    "169.254.169.254",          # AWS, GCP, Azure, DigitalOcean metadata
    "169.254.169.253",          # AWS DNS
    "metadata.google.internal",  # GCP metadata DNS
    "metadata.google.com",
    "metadata.internal",
    "100.100.100.200",          # Alibaba Cloud metadata
    "instance-data",            # AWS legacy metadata host
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
])

# Reserved / Private networks not fully covered by ip.is_private alone
EXTRA_PRIVATE_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),    # RFC 6598 Carrier Grade NAT / Shared Address Space
    ipaddress.ip_network("198.18.0.0/15"),    # RFC 2544 Benchmarking
    ipaddress.ip_network("192.0.0.0/24"),     # RFC 6890 IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),     # RFC 5737 TEST-NET-1
    ipaddress.ip_network("198.51.100.0/24"),  # RFC 5737 TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),   # RFC 5737 TEST-NET-3
    ipaddress.ip_network("0.0.0.0/8"),        # RFC 1122 Current network ("this" host)
    ipaddress.ip_network("240.0.0.0/4"),      # RFC 1122 Reserved for future use
)

# Non-HTTP protocol ports blocked to prevent service probing or port scanning
DISALLOWED_PORTS = frozenset([
    20, 21,       # FTP
    22,           # SSH
    23,           # Telnet
    25, 465, 587, # SMTP
    53,           # DNS
    69,           # TFTP
    110, 995,     # POP3
    111,          # RPC
    135, 137, 138, 139, 445, # SMB/NetBIOS
    143, 993,     # IMAP
    1433, 1434,   # MS SQL
    1521,         # Oracle
    3306,         # MySQL
    3389,         # RDP
    5432,         # PostgreSQL
    5900, 5901,   # VNC
    6379,         # Redis
    11211,        # Memcached
    27017, 27018, # MongoDB
])


# ─────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────
@dataclass
class HealthCheckResult:
    """Structured result of a single website health check."""
    url: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Core measurements
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    content_length: Optional[int] = None

    # Classification
    is_healthy: bool = False
    error_category: Optional[str] = None   # "dns_failure", "ssl_error", "timeout", "connection_refused", "4xx_client_error", "5xx_server_error", "ssrf_blocked", "invalid_url", "redirect_loop", "network_error"
    error_detail: Optional[str] = None
    severity: str = "unknown"              # "healthy", "warning", "critical"

    # SSL info
    ssl_valid: Optional[bool] = None
    ssl_issuer: Optional[str] = None
    ssl_expiry: Optional[str] = None
    ssl_days_remaining: Optional[int] = None

    # Redirect info
    final_url: Optional[str] = None
    redirect_count: int = 0
    redirect_chain: List[str] = field(default_factory=list)

    # Observable symptoms vs confirmed issues
    observable_symptoms: List[str] = field(default_factory=list)
    confirmed_issues: List[str] = field(default_factory=list)

    # HTTP headers of interest
    server_header: Optional[str] = None
    content_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "timestamp": self.timestamp,
            "status_code": self.status_code,
            "response_time_ms": round(self.response_time_ms, 1) if self.response_time_ms else None,
            "is_healthy": self.is_healthy,
            "error_category": self.error_category,
            "error_detail": self.error_detail,
            "severity": self.severity,
            "ssl_valid": self.ssl_valid,
            "ssl_issuer": self.ssl_issuer,
            "ssl_expiry": self.ssl_expiry,
            "ssl_days_remaining": self.ssl_days_remaining,
            "redirect_count": self.redirect_count,
            "final_url": self.final_url,
            "content_length": self.content_length,
            "server_header": self.server_header,
            "content_type": self.content_type,
            "observable_symptoms": "; ".join(self.observable_symptoms),
            "confirmed_issues": "; ".join(self.confirmed_issues),
        }


# ─────────────────────────────────────────────────────────────
# SSRF / Security Validation
# ─────────────────────────────────────────────────────────────
def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is private, loopback, link-local, CGNAT, or reserved."""
    try:
        ip = ipaddress.ip_address(ip_str)
        if hasattr(ip, "ipv4_mapped") and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
        return any(ip in net for net in EXTRA_PRIVATE_NETWORKS)
    except ValueError:
        return True  # If unparseable, block for security


def _validate_url(url: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate and sanitize the URL. Returns (is_valid, cleaned_url, error_msg).
    Only http:// and https:// are allowed.
    Enforces SSRF prevention against private IPs, localhost, and cloud metadata.
    """
    url = url.strip()
    if not url:
        return False, url, "URL cannot be empty."

    # Scheme check
    if "://" in url:
        scheme = url.split("://", 1)[0].lower()
        if scheme not in ("http", "https"):
            return False, url, f"Scheme '{scheme}' is not allowed. Only HTTP and HTTPS are permitted."
    else:
        # Auto-add scheme if missing
        url = "https://" + url

    try:
        parsed = urlparse(url)
    except Exception:
        return False, url, "Could not parse URL."

    if parsed.scheme not in ("http", "https"):
        return False, url, f"Scheme '{parsed.scheme}' is not allowed. Only HTTP and HTTPS are permitted."

    hostname = parsed.hostname
    if not hostname:
        # Detect bare IPv6 like http://::1 or http://0:0:0:0
        netloc = parsed.netloc or parsed.path
        if "::" in netloc or netloc.strip("/").startswith(":"):
            return False, url, "Requests to localhost or private IP addresses are not allowed."
        return False, url, "URL has no valid hostname."

    # Block disallowed protocol ports (prevent port scanning / protocol abuse)
    if parsed.port and parsed.port in DISALLOWED_PORTS:
        return False, url, f"Port {parsed.port} is not an allowed web port for health monitoring."

    # Block cloud metadata and known dangerous hosts
    if hostname.lower() in BLOCKED_HOSTS:
        return False, url, "This hostname is blocked for security reasons (cloud metadata endpoint or localhost)."

    # Block common localhost aliases
    if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return False, url, "Requests to localhost are not allowed."

    # Block integer/decimal representations of IP addresses (e.g. 2130706433 -> 127.0.0.1)
    if hostname.isdigit():
        try:
            val = int(hostname)
            if 0 <= val <= 0xFFFFFFFF:
                ip_dec = ipaddress.IPv4Address(val)
                if _is_private_ip(str(ip_dec)):
                    return False, url, "Requests to private or reserved IP addresses are not allowed."
        except Exception:
            pass

    # Block IP addresses in private ranges
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_private_ip(str(ip)):
            return False, url, "Requests to private or reserved IP addresses are not allowed."
    except ValueError:
        # hostname is a domain name, not a raw IP — that's fine, we'll resolve it below
        pass

    # Resolve DNS and check all resolved IP addresses
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addr_infos = socket.getaddrinfo(hostname, port)
        for family, socktype, proto, canonname, sockaddr in addr_infos:
            resolved_ip = sockaddr[0]
            if _is_private_ip(resolved_ip):
                return False, url, f"Domain '{hostname}' resolves to a private IP address ({resolved_ip}). Blocked for security."
            if resolved_ip in BLOCKED_HOSTS:
                return False, url, f"Domain '{hostname}' resolves to a blocked IP ({resolved_ip})."
    except socket.gaierror:
        # DNS resolution failure will be reported naturally during connection check
        pass

    return True, url, None


def _validate_redirect_target(response_url: str) -> bool:
    """Validate that a redirect target is safe (not SSRF)."""
    is_valid, _, _ = _validate_url(response_url)
    return is_valid


# ─────────────────────────────────────────────────────────────
# Rate Limiter (per-session, in-memory sliding window)
# ─────────────────────────────────────────────────────────────
class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_requests: int = RATE_LIMIT_MAX_CHECKS, window_seconds: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: List[float] = []

    def is_allowed(self) -> bool:
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < self.window_seconds]
        return len(self._timestamps) < self.max_requests

    def record(self):
        self._timestamps.append(time.time())

    def remaining(self) -> int:
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < self.window_seconds]
        return max(0, self.max_requests - len(self._timestamps))


# ─────────────────────────────────────────────────────────────
# SSL Certificate Inspection
# ─────────────────────────────────────────────────────────────
def _check_ssl_certificate(hostname: str, port: int = 443) -> Dict[str, Any]:
    """Inspect the SSL certificate of a host. Returns dict with issuer, expiry, validity, days remaining."""
    result: Dict[str, Any] = {
        "valid": None,
        "issuer": None,
        "expiry": None,
        "days_remaining": None,
        "error": None,
    }

    # Security check: do not connect to private or blocked hosts
    if _is_private_ip(hostname) or hostname.lower() in BLOCKED_HOSTS:
        result["valid"] = False
        result["error"] = "SSL inspection blocked for private or restricted host."
        return result

    try:
        addr_infos = socket.getaddrinfo(hostname, port)
        for *_, sockaddr in addr_infos:
            if _is_private_ip(sockaddr[0]) or sockaddr[0] in BLOCKED_HOSTS:
                result["valid"] = False
                result["error"] = f"Domain resolves to private or blocked IP ({sockaddr[0]})."
                return result
    except socket.gaierror as e:
        result["valid"] = None
        result["error"] = f"DNS resolution failed: {e}"
        return result

    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as ssock:
            ssock.settimeout(CONNECT_TIMEOUT)
            ssock.connect((hostname, port))
            cert = ssock.getpeercert()
            if cert:
                result["valid"] = True
                issuer_parts = dict(x[0] for x in cert.get("issuer", []))
                result["issuer"] = issuer_parts.get("organizationName", issuer_parts.get("commonName", "Unknown Issuer"))
                not_after = cert.get("notAfter", "")
                if not_after:
                    result["expiry"] = not_after
                    try:
                        expiry_dt = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
                        days_left = (expiry_dt - datetime.now(timezone.utc).replace(tzinfo=None)).days
                        result["days_remaining"] = days_left
                        if days_left < 0:
                            result["valid"] = False
                            result["error"] = f"Certificate expired {abs(days_left)} days ago on {not_after}."
                    except Exception:
                        pass
    except ssl.SSLCertVerificationError as e:
        result["valid"] = False
        result["error"] = f"SSL certificate verification failed: {e.verify_message if hasattr(e, 'verify_message') else str(e)}"
    except ssl.SSLError as e:
        result["valid"] = False
        result["error"] = f"SSL error: {str(e)}"
    except socket.timeout:
        result["valid"] = None
        result["error"] = "SSL handshake timed out."
    except ConnectionRefusedError:
        result["valid"] = None
        result["error"] = "Connection refused on HTTPS port (443)."
    except Exception as e:
        result["valid"] = None
        result["error"] = f"Could not inspect SSL: {str(e)}"

    return result


# ─────────────────────────────────────────────────────────────
# Optional OpenTelemetry Integration (Future-Ready Hook)
# ─────────────────────────────────────────────────────────────
class WebsiteHealthTelemetryExporter:
    """
    Optional future integration hook for OpenTelemetry.
    When opentelemetry SDK is installed and configured, this exporter emits
    health check span metrics (http.status_code, http.response_time_ms, website.health).
    If opentelemetry is not installed, it acts as a zero-overhead safe no-op.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._tracer = None
        if enabled:
            try:
                from opentelemetry import trace  # type: ignore
                self._tracer = trace.get_tracer("cloudsleuth.website_monitor")
            except (ImportError, Exception):
                self.enabled = False

    def record_check(self, result: HealthCheckResult) -> None:
        """Export health check result as an OpenTelemetry span/metric if enabled."""
        if not self.enabled or not self._tracer:
            return
        try:
            with self._tracer.start_as_current_span("website_health_check") as span:
                span.set_attribute("http.url", result.url)
                if result.status_code is not None:
                    span.set_attribute("http.status_code", result.status_code)
                if result.response_time_ms is not None:
                    span.set_attribute("http.response_time_ms", result.response_time_ms)
                span.set_attribute("website.is_healthy", result.is_healthy)
                if result.error_category:
                    span.set_attribute("website.error_category", result.error_category)
                if result.severity:
                    span.set_attribute("website.severity", result.severity)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Core Health Check Engine
# ─────────────────────────────────────────────────────────────
def check_website_health(
    url: str,
    rate_limiter: Optional[RateLimiter] = None,
    telemetry_exporter: Optional[WebsiteHealthTelemetryExporter] = None,
) -> HealthCheckResult:
    """
    Perform a comprehensive external health check on a website URL.
    Returns a structured HealthCheckResult with symptoms and classifications.
    Safely validates every redirect hop and resolved IP address.
    """
    result = HealthCheckResult(url=url)

    # 1. Rate limit check
    if rate_limiter and not rate_limiter.is_allowed():
        result.error_category = "rate_limited"
        result.error_detail = f"Too many checks. Please wait before trying again (max {RATE_LIMIT_MAX_CHECKS} per {RATE_LIMIT_WINDOW}s)."
        result.severity = "warning"
        result.observable_symptoms.append("Rate limit reached for health checks.")
        return result

    # 2. Validate initial URL
    is_valid, cleaned_url, error_msg = _validate_url(url)
    if not is_valid:
        is_security_block = any(k in (error_msg or "").lower() for k in ("private", "blocked", "localhost", "port"))
        result.error_category = "ssrf_blocked" if is_security_block else "invalid_url"
        result.error_detail = error_msg
        result.severity = "critical"
        result.confirmed_issues.append(error_msg or "Invalid URL")
        result.observable_symptoms.append(f"Request blocked by security validator: {error_msg}")
        return result

    result.url = cleaned_url
    parsed = urlparse(cleaned_url)

    # 3. SSL certificate inspection (for HTTPS)
    if parsed.scheme == "https":
        ssl_info = _check_ssl_certificate(parsed.hostname, parsed.port or 443)
        result.ssl_issuer = ssl_info["issuer"]
        result.ssl_expiry = ssl_info["expiry"]
        result.ssl_days_remaining = ssl_info.get("days_remaining")
        if ssl_info["valid"] is False:
            result.ssl_valid = False
            result.observable_symptoms.append("SSL certificate verification issue detected.")
            result.confirmed_issues.append(ssl_info.get("error", "SSL verification failed"))
        elif ssl_info["valid"] is True:
            result.ssl_valid = True
            days = ssl_info.get("days_remaining")
            if days is not None and days <= 14:
                result.observable_symptoms.append(f"SSL certificate is expiring soon ({days} days remaining).")

    # 4. Record rate limit quota
    if rate_limiter:
        rate_limiter.record()

    # 5. Safe hop-by-hop HTTP request execution
    session = requests.Session()
    session.headers.update({
        "User-Agent": "CloudSleuth-HealthMonitor/1.0 (Passive Health Check; +https://github.com/Vaishnavi74-star/Cloudsleuth)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    current_url = cleaned_url
    redirect_count = 0
    redirect_chain = []
    hop_resp = None

    try:
        start_time = time.perf_counter()
        for hop in range(MAX_REDIRECTS + 1):
            # Validate every redirect hop before connection
            is_valid, validated_url, err_msg = _validate_url(current_url)
            if not is_valid:
                result.error_category = "ssrf_blocked" if any(k in (err_msg or "").lower() for k in ("private", "blocked", "port", "localhost")) else "invalid_url"
                result.error_detail = f"Redirect target blocked for security: {err_msg}"
                result.severity = "critical"
                result.confirmed_issues.append(f"Redirect target blocked by security validator: {err_msg}")
                result.observable_symptoms.append(f"Server attempted redirect to unsafe address ({current_url}).")
                if hop_resp:
                    hop_resp.close()
                return result

            hop_resp = session.get(
                validated_url,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                allow_redirects=False,
                stream=True,
                verify=True,
            )

            # Check for redirect status codes (301, 302, 303, 307, 308)
            if hop_resp.is_redirect or hop_resp.status_code in (301, 302, 303, 307, 308):
                redirect_count += 1
                loc = hop_resp.headers.get("Location")
                hop_resp.close()

                if not loc:
                    result.status_code = hop_resp.status_code
                    result.final_url = validated_url
                    result.redirect_count = redirect_count
                    break

                next_url = urljoin(validated_url, loc)
                redirect_chain.append(f"HTTP {hop_resp.status_code}: {validated_url} → {next_url}")

                if redirect_count > MAX_REDIRECTS:
                    result.error_category = "redirect_loop"
                    result.error_detail = f"Exceeded maximum allowed redirects ({MAX_REDIRECTS})."
                    result.severity = "critical"
                    result.confirmed_issues.append(f"Redirect loop detected: exceeded maximum of {MAX_REDIRECTS} redirects.")
                    result.observable_symptoms.append("Browser would show 'Too many redirects' error.")
                    result.redirect_count = redirect_count
                    result.final_url = next_url
                    result.redirect_chain = redirect_chain
                    return result

                current_url = next_url
                continue

            # Destination reached!
            break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        result.response_time_ms = elapsed_ms

        if hop_resp is not None:
            result.status_code = hop_resp.status_code
            result.final_url = current_url
            result.redirect_count = redirect_count
            result.redirect_chain = redirect_chain
            result.server_header = hop_resp.headers.get("Server")
            result.content_type = hop_resp.headers.get("Content-Type")

            # Read limited body chunk (up to 1 MB cap) to measure content length without memory blowup
            if hasattr(hop_resp, "raw") and hop_resp.raw:
                content_chunk = hop_resp.raw.read(MAX_RESPONSE_BYTES + 1)
                result.content_length = len(content_chunk)
            hop_resp.close()

        # Classify the result
        _classify_result(result)

    except requests.exceptions.SSLError as e:
        result.error_category = "ssl_error"
        result.error_detail = f"SSL/TLS connection failed: {_sanitize_error(str(e))}"
        result.severity = "critical"
        result.ssl_valid = False
        result.confirmed_issues.append("SSL/TLS handshake failure — the server's certificate may be expired, self-signed, or misconfigured.")
        result.observable_symptoms.append("Browser would show a security warning for this site.")

    except requests.exceptions.ConnectionError as e:
        err_str = _sanitize_error(str(e))
        if any(term in str(e).lower() for term in ("nameresolutionerror", "getaddrinfo", "name or service not known", "nodename nor servname provided")):
            result.error_category = "dns_failure"
            result.error_detail = f"DNS resolution failed for '{parsed.hostname}'."
            result.severity = "critical"
            result.confirmed_issues.append(f"Domain '{parsed.hostname}' could not be resolved. The domain may not exist, DNS may be down, or there is a typo.")
            result.observable_symptoms.append(f"DNS lookup failed for '{parsed.hostname}' — website is completely unreachable (no IP address resolved).")
        elif "connection refused" in str(e).lower():
            result.error_category = "connection_refused"
            result.error_detail = "Connection refused by the server."
            result.severity = "critical"
            result.confirmed_issues.append("The server actively refused the connection. The web server process may not be running.")
            result.observable_symptoms.append("Server is reachable at the network level but no service is listening on the expected port.")
        else:
            result.error_category = "network_error"
            result.error_detail = f"Network connection error: {err_str}"
            result.severity = "critical"
            result.confirmed_issues.append("Could not establish a network connection to the server.")
            result.observable_symptoms.append("The website may be down, or a firewall/network issue is blocking the connection.")

    except requests.exceptions.Timeout:
        result.error_category = "timeout"
        result.error_detail = f"Request timed out after {CONNECT_TIMEOUT + READ_TIMEOUT}s."
        result.severity = "critical"
        result.observable_symptoms.append("The server did not respond within the timeout period.")
        result.confirmed_issues.append("Connection or response timed out. The server may be overloaded, unresponsive, or the network path is congested.")

    except requests.exceptions.TooManyRedirects:
        result.error_category = "redirect_loop"
        result.error_detail = f"Exceeded maximum redirects ({MAX_REDIRECTS})."
        result.severity = "critical"
        result.confirmed_issues.append("The server is caught in a redirect loop, preventing the page from loading.")
        result.observable_symptoms.append("Browser would show 'too many redirects' error.")

    except requests.exceptions.RequestException as e:
        result.error_category = "network_error"
        result.error_detail = f"Request failed: {_sanitize_error(str(e))}"
        result.severity = "critical"
        result.observable_symptoms.append("An unexpected error occurred while connecting to the website.")

    finally:
        session.close()

    # Optional OpenTelemetry Telemetry Export
    if telemetry_exporter:
        telemetry_exporter.record_check(result)

    return result


def _classify_result(result: HealthCheckResult):
    """Classify the health check result based on status code and response time."""
    code = result.status_code
    if code is None:
        return

    if 200 <= code < 300:
        result.is_healthy = True
        result.severity = "healthy"
        result.observable_symptoms.append(f"HTTP {code} — Server responded successfully.")
        if result.response_time_ms and result.response_time_ms > 3000:
            result.severity = "warning"
            result.is_healthy = True  # Still reachable, but degraded latency
            result.observable_symptoms.append(f"Response time is slow ({result.response_time_ms:.0f} ms). Normal is under 1000 ms.")
        elif result.response_time_ms and result.response_time_ms > 1000:
            result.observable_symptoms.append(f"Response time is moderate ({result.response_time_ms:.0f} ms).")

    elif 300 <= code < 400:
        result.is_healthy = True
        result.severity = "healthy"
        result.observable_symptoms.append(f"HTTP {code} — Redirect response (final destination loaded successfully).")

    elif 400 <= code < 500:
        result.error_category = "4xx_client_error"
        result.severity = "warning" if code in (403, 429) else "critical"

        error_names = {
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            408: "Request Timeout",
            429: "Too Many Requests",
            451: "Unavailable for Legal Reasons",
        }
        name = error_names.get(code, "Client Error")
        result.error_detail = f"HTTP {code} {name}"
        result.observable_symptoms.append(f"Server returned HTTP {code} ({name}).")

        if code == 403:
            result.confirmed_issues.append("Access is forbidden. The server is running but denying this request (possibly IP-based blocking, WAF, or authentication required).")
        elif code == 404:
            result.confirmed_issues.append("The requested page was not found. The URL path may be wrong, or the page has been removed.")
        elif code == 429:
            result.confirmed_issues.append("Rate limited by the server. Too many requests have been sent to this endpoint.")
        elif code == 401:
            result.confirmed_issues.append("Authentication is required to access this resource.")
        else:
            result.confirmed_issues.append(f"Client-side HTTP error ({code}). The request may be malformed or the resource may have restrictions.")

    elif 500 <= code < 600:
        result.error_category = "5xx_server_error"
        result.severity = "critical"

        error_names = {
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Gateway Timeout",
        }
        name = error_names.get(code, "Server Error")
        result.error_detail = f"HTTP {code} {name}"
        result.observable_symptoms.append(f"Server returned HTTP {code} ({name}).")

        if code == 502:
            result.confirmed_issues.append("Bad Gateway — the reverse proxy or load balancer received an invalid response from the upstream server.")
        elif code == 503:
            result.confirmed_issues.append("Service Unavailable — the server is temporarily unable to handle requests (maintenance or overload).")
        elif code == 504:
            result.confirmed_issues.append("Gateway Timeout — the upstream server did not respond in time to the proxy/load balancer.")
        else:
            result.confirmed_issues.append(f"Server-side error ({code}). The application or backend service encountered an internal failure.")


def _sanitize_error(err: str) -> str:
    """Remove potentially sensitive internal system paths or long traces from error messages."""
    err = re.sub(r'[A-Za-z]:\\[^\s"\']+', '[path]', err)
    err = re.sub(r'/[^\s"\']*site-packages/[^\s"\']*', '[internal]', err)
    if len(err) > 300:
        err = err[:300] + "..."
    return err


# ─────────────────────────────────────────────────────────────
# AI Explanation Prompt Builder
# ─────────────────────────────────────────────────────────────
def build_website_health_prompt(result: HealthCheckResult) -> str:
    """
    Build a prompt for Gemma to explain the health check result.
    Strictly instructs the model to distinguish observable symptoms from confirmed root causes
    and emphasizes that external checks do NOT have access to internal telemetry.
    """
    symptoms_text = "\n".join(f"  - {s}" for s in result.observable_symptoms) if result.observable_symptoms else "  None detected."
    issues_text = "\n".join(f"  - {i}" for i in result.confirmed_issues) if result.confirmed_issues else "  None confirmed."

    ssl_line = ""
    if result.ssl_valid is not None:
        ssl_status = "Valid" if result.ssl_valid else "INVALID / Failed"
        ssl_line = f"SSL Certificate: {ssl_status}"
        if result.ssl_issuer:
            ssl_line += f" (Issuer: {result.ssl_issuer})"
        if result.ssl_expiry:
            ssl_line += f" (Expires: {result.ssl_expiry})"
        if result.ssl_days_remaining is not None:
            ssl_line += f" ({result.ssl_days_remaining} days remaining)"

    redirect_line = ""
    if result.redirect_count > 0:
        redirect_line = f"Redirects: {result.redirect_count} redirect(s) to {result.final_url}"

    prompt = f"""You are CloudSleuth's Website Health Monitoring assistant. A user has performed a passive, external health check on a public website. Your job is to explain the results in clear, simple English that a non-technical person can understand.

IMPORTANT RULES:
1. You are explaining EXTERNALLY OBSERVABLE symptoms only. You do NOT have access to the website's internal servers, logs, metrics, or code.
2. Clearly separate what you can CONFIRM from what you can only SUSPECT.
3. Provide practical troubleshooting suggestions the user can try.
4. Do not speculate about internal server architecture you cannot see.
5. Keep your response concise — around 150-250 words.

HEALTH CHECK RESULTS:
URL: {result.url}
Status: {"HEALTHY" if result.is_healthy else "UNHEALTHY"}
Severity: {result.severity.upper()}
HTTP Status Code: {result.status_code or "No response received"}
Response Time: {f"{result.response_time_ms:.0f} ms" if result.response_time_ms else "N/A (no response)"}
Error Category: {result.error_category or "None"}
Error Detail: {result.error_detail or "None"}
{ssl_line}
{redirect_line}
Server: {result.server_header or "Not disclosed"}
Content Type: {result.content_type or "Unknown"}

OBSERVABLE SYMPTOMS:
{symptoms_text}

CONFIRMED ISSUES:
{issues_text}

Please provide:
1. A brief, plain-English summary of the website's health status.
2. The distinction between what is an observable symptom vs. a confirmed root cause.
3. 2-3 practical troubleshooting steps the user can take.
"""
    return prompt


def generate_deterministic_explanation(result: HealthCheckResult) -> str:
    """Generate a structured explanation without AI, based purely on the check results."""
    lines = []

    # Header
    if result.is_healthy:
        lines.append("### ✅ Website is Healthy")
        lines.append(f"**{result.url}** responded successfully with HTTP {result.status_code}.")
    else:
        severity_emoji = "🔴" if result.severity == "critical" else "🟡"
        lines.append(f"### {severity_emoji} Website Health Issue Detected")
        lines.append(f"**{result.url}** — {result.error_detail or 'An issue was detected.'}")

    lines.append("")

    # Response time
    if result.response_time_ms:
        if result.response_time_ms < 500:
            lines.append(f"⚡ **Response Time**: {result.response_time_ms:.0f} ms (Fast)")
        elif result.response_time_ms < 1500:
            lines.append(f"⏱️ **Response Time**: {result.response_time_ms:.0f} ms (Normal)")
        elif result.response_time_ms < 3000:
            lines.append(f"🐢 **Response Time**: {result.response_time_ms:.0f} ms (Slow)")
        else:
            lines.append(f"🐌 **Response Time**: {result.response_time_ms:.0f} ms (Very Slow)")

    # SSL
    if result.ssl_valid is not None:
        if result.ssl_valid:
            days_str = f", {result.ssl_days_remaining} days remaining" if result.ssl_days_remaining is not None else ""
            lines.append(f"🔒 **SSL Certificate**: Valid (Issuer: {result.ssl_issuer or 'Unknown'}{days_str})")
        else:
            lines.append("🔓 **SSL Certificate**: INVALID — visitors will see browser security warnings.")

    # Redirects
    if result.redirect_count > 0:
        lines.append(f"↪️ **Redirects**: {result.redirect_count} redirect(s) → {result.final_url}")

    lines.append("")

    # Symptoms vs issues
    if result.observable_symptoms:
        lines.append("**What We Observed (External Symptoms):**")
        for s in result.observable_symptoms:
            lines.append(f"- {s}")
        lines.append("")

    if result.confirmed_issues:
        lines.append("**Confirmed Issues:**")
        for i in result.confirmed_issues:
            lines.append(f"- {i}")
        lines.append("")

    # Troubleshooting suggestions
    if not result.is_healthy:
        lines.append("**Suggested Next Steps:**")
        if result.error_category == "dns_failure":
            lines.append("1. Verify the domain name is spelled correctly.")
            lines.append("2. Try accessing the site from a different network or DNS resolver (e.g. 8.8.8.8 or 1.1.1.1).")
            lines.append("3. Check with domain registrar to confirm domain registration and DNS records are active.")
        elif result.error_category == "ssl_error":
            lines.append("1. Check if the SSL certificate has expired or was issued for a different domain.")
            lines.append("2. Verify your web server is serving the complete certificate chain (intermediate certificates).")
            lines.append("3. Renew certificate via Let's Encrypt / Certbot or your SSL certificate authority.")
        elif result.error_category == "timeout":
            lines.append("1. Wait a few moments and try again — the server or network route may be experiencing temporary traffic spikes.")
            lines.append("2. Check server host resources (CPU, RAM, network connections) if you own the server.")
            lines.append("3. Inspect whether a DDoS protection service or firewall is dropping probe packets.")
        elif result.error_category == "4xx_client_error":
            lines.append("1. Double-check the URL path for typos or outdated links.")
            lines.append("2. If HTTP 403 or 401, check authentication credentials, IP allowlists, or WAF rules.")
            lines.append("3. If HTTP 429, decrease request frequency or request a quota increase.")
        elif result.error_category == "5xx_server_error":
            lines.append("1. Wait a few minutes and recheck — upstream server restarts or deployments often trigger transient 502/503 errors.")
            lines.append("2. If you are the site owner, inspect backend application logs and reverse proxy (Nginx/Envoy) upstream health.")
            lines.append("3. Check status page and cloud provider health dashboards for infrastructure incidents.")
        elif result.error_category == "connection_refused":
            lines.append("1. Verify the web server daemon (Nginx, Apache, Caddy, Node.js) is running.")
            lines.append("2. Check firewall or cloud security group rules to ensure incoming traffic on ports 80/443 is permitted.")
            lines.append("3. Verify the service is bound to 0.0.0.0 rather than 127.0.0.1 on the host.")
        elif result.error_category == "ssrf_blocked":
            lines.append("1. Only publicly accessible websites on standard web ports can be monitored.")
            lines.append("2. Requests to private networks, localhost, cloud metadata, or restricted ports are rejected for security.")
        elif result.error_category == "redirect_loop":
            lines.append("1. Check web server redirect rules (e.g. HTTP to HTTPS loops or trailing slash redirects).")
            lines.append("2. Clear browser cookies and test the URL directly in an incognito window.")
        else:
            lines.append("1. Try checking the URL again in a few minutes.")
            lines.append("2. Verify network connectivity from multiple distinct networks.")
            lines.append("3. Contact the website administrator if the outage persists.")

    lines.append("")
    lines.append("*Note: This is an external, passive check only. We do not have access to the website's internal servers, logs, or infrastructure.*")

    return "\n".join(lines)
