"""
CloudSleuth Website Health Monitor Unit Tests.
Tests URL validation, SSRF protection, rate limiting, error classification,
symptoms vs. root-cause separation, and optional OpenTelemetry integration.
"""

from unittest.mock import patch, MagicMock
import pytest
import requests

from cloudsleuth.website_monitor import (
    _validate_url,
    _is_private_ip,
    RateLimiter,
    HealthCheckResult,
    _classify_result,
    build_website_health_prompt,
    generate_deterministic_explanation,
    check_website_health,
    WebsiteHealthTelemetryExporter,
    MAX_RESPONSE_BYTES,
)
from cloudsleuth.gemma_service import GemmaService


def test_allowed_schemes():
    # Only HTTP and HTTPS allowed
    valid, url, err = _validate_url("https://example.com")
    assert valid is True
    assert err is None

    valid, url, err = _validate_url("http://example.com")
    assert valid is True
    assert err is None

    valid, url, err = _validate_url("ftp://example.com")
    assert valid is False
    assert "Only HTTP and HTTPS" in err

    valid, url, err = _validate_url("file:///etc/passwd")
    assert valid is False


def test_ssrf_blocking_localhost():
    # Localhost variations
    for host in ["localhost", "127.0.0.1", "[::1]", "::1", "0.0.0.0"]:
        valid, url, err = _validate_url(f"http://{host}")
        assert valid is False
        assert "localhost" in err.lower() or "not allowed" in err.lower() or "blocked" in err.lower() or "private" in err.lower()


def test_ssrf_blocking_cloud_metadata():
    # AWS / GCP / Azure metadata IP
    valid, url, err = _validate_url("http://169.254.169.254/latest/meta-data/")
    assert valid is False
    assert "blocked" in err.lower() or "metadata" in err.lower()


def test_ssrf_blocking_private_ips():
    # Private RFC 1918 ranges
    for private_ip in ["10.0.0.1", "172.16.0.5", "192.168.1.1"]:
        valid, url, err = _validate_url(f"http://{private_ip}/status")
        assert valid is False
        assert "private" in err.lower() or "reserved" in err.lower()


def test_ssrf_blocking_cgnat_and_extra_ranges():
    # RFC 6598 CGNAT range and decimal IP representations
    assert _is_private_ip("100.64.0.1") is True
    assert _is_private_ip("198.18.0.1") is True
    assert _is_private_ip("0.0.0.1") is True

    valid, url, err = _validate_url("http://100.64.0.1/admin")
    assert valid is False
    assert "private" in err.lower() or "reserved" in err.lower()

    # Decimal representation of 127.0.0.1 (2130706433)
    valid, url, err = _validate_url("http://2130706433")
    assert valid is False


def test_disallowed_non_web_ports():
    # Disallowed service ports (e.g. SSH 22, MySQL 3306, Redis 6379)
    for port in [22, 25, 3306, 6379]:
        valid, url, err = _validate_url(f"http://example.com:{port}")
        assert valid is False
        assert "port" in err.lower() or "restricted" in err.lower() or "not an allowed" in err.lower()


def test_rate_limiter():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.is_allowed() is True
    assert limiter.remaining() == 3

    limiter.record()
    assert limiter.remaining() == 2
    limiter.record()
    assert limiter.remaining() == 1
    limiter.record()
    assert limiter.remaining() == 0
    assert limiter.is_allowed() is False


def test_error_classification_200():
    res = HealthCheckResult(url="https://example.com")
    res.status_code = 200
    res.response_time_ms = 180.0
    _classify_result(res)

    assert res.is_healthy is True
    assert res.severity == "healthy"
    assert any("200" in s for s in res.observable_symptoms)


def test_error_classification_4xx():
    res = HealthCheckResult(url="https://example.com/missing")
    res.status_code = 404
    _classify_result(res)

    assert res.is_healthy is False
    assert res.error_category == "4xx_client_error"
    assert "Not Found" in res.error_detail
    assert len(res.confirmed_issues) > 0


def test_error_classification_5xx():
    res = HealthCheckResult(url="https://example.com/broken")
    res.status_code = 503
    _classify_result(res)

    assert res.is_healthy is False
    assert res.severity == "critical"
    assert res.error_category == "5xx_server_error"
    assert "Service Unavailable" in res.error_detail
    assert len(res.confirmed_issues) > 0


def test_dns_failure_detection():
    # Domain that cannot be resolved
    res = check_website_health("https://nonexistent-domain-test-xyz-987123.org")
    assert res.is_healthy is False
    assert res.error_category == "dns_failure"
    assert res.severity == "critical"
    assert any("DNS" in s or "Host" in s for s in res.observable_symptoms)
    assert len(res.confirmed_issues) > 0


def test_timeout_detection():
    with patch("requests.Session.get", side_effect=requests.exceptions.Timeout("Connection timed out")):
        res = check_website_health("https://example.com")
        assert res.is_healthy is False
        assert res.error_category == "timeout"
        assert res.severity == "critical"
        assert any("timeout" in s.lower() or "timed out" in s.lower() for s in res.observable_symptoms)
        assert any("timed out" in c.lower() for c in res.confirmed_issues)


def test_ssl_failure_detection():
    with patch("requests.Session.get", side_effect=requests.exceptions.SSLError("certificate verify failed")):
        res = check_website_health("https://example.com")
        assert res.is_healthy is False
        assert res.error_category == "ssl_error"
        assert res.severity == "critical"
        assert res.ssl_valid is False
        assert any("ssl" in c.lower() or "certificate" in c.lower() for c in res.confirmed_issues)


def test_redirect_ssrf_blocking():
    # Mock a response that redirects to a private IP (169.254.169.254)
    mock_redirect = MagicMock()
    mock_redirect.is_redirect = True
    mock_redirect.status_code = 302
    mock_redirect.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}

    with patch("requests.Session.get", return_value=mock_redirect):
        res = check_website_health("https://example.com")
        assert res.is_healthy is False
        assert res.error_category == "ssrf_blocked"
        assert res.severity == "critical"
        assert any("blocked" in c.lower() or "redirect" in c.lower() for c in res.confirmed_issues)


def test_redirect_loop_detection():
    # Mock a response that endlessly redirects
    mock_hop = MagicMock()
    mock_hop.is_redirect = True
    mock_hop.status_code = 301
    mock_hop.headers = {"Location": "https://example.com/loop"}

    with patch("requests.Session.get", return_value=mock_hop):
        res = check_website_health("https://example.com")
        assert res.is_healthy is False
        assert res.error_category == "redirect_loop"
        assert res.severity == "critical"
        assert any("redirect loop" in c.lower() for c in res.confirmed_issues)


def test_symptoms_vs_confirmed_issues_distinction():
    res = HealthCheckResult(url="https://example.com")
    res.status_code = 502
    res.response_time_ms = 4500.0
    res.observable_symptoms.append("HTTP 502 response received from proxy.")
    res.confirmed_issues.append("Reverse proxy reported upstream invalid response.")

    prompt = build_website_health_prompt(res)
    assert "OBSERVABLE SYMPTOMS:" in prompt
    assert "CONFIRMED ISSUES:" in prompt
    assert "HTTP 502" in prompt

    det = generate_deterministic_explanation(res)
    assert "What We Observed (External Symptoms):" in det
    assert "Confirmed Issues:" in det
    assert "Suggested Next Steps:" in det


def test_gemma_service_explain_website_health():
    gemma = GemmaService()
    res = HealthCheckResult(url="https://example.com")
    res.status_code = 200
    res.response_time_ms = 150.0
    res.is_healthy = True
    res.severity = "healthy"

    explanation = gemma.explain_website_health(res)
    assert explanation is not None
    assert explanation.text != ""
    assert "example.com" in explanation.text


def test_result_serialization_to_dict():
    res = HealthCheckResult(url="https://example.com")
    res.status_code = 200
    res.response_time_ms = 120.4
    res.is_healthy = True
    res.severity = "healthy"
    res.observable_symptoms.append("HTTP 200 OK")
    res.confirmed_issues.append("Protocol handshake OK")

    d = res.to_dict()
    assert d["url"] == "https://example.com"
    assert d["status_code"] == 200
    assert d["is_healthy"] is True
    assert "HTTP 200 OK" in d["observable_symptoms"]


def test_opentelemetry_exporter_optional():
    exporter = WebsiteHealthTelemetryExporter(enabled=False)
    assert exporter.enabled is False

    res = HealthCheckResult(url="https://example.com")
    # Should not raise exception
    exporter.record_check(res)
