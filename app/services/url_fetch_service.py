import ipaddress
import logging
import re
import socket
import ssl
import unicodedata
from html.parser import HTMLParser
from typing import Callable, Iterable
from urllib.parse import urljoin, urlparse

import httpcore
import httpx

from app.core.config import settings
from app.services.scan_services import ScanValidationError

logger = logging.getLogger(__name__)

# Constants
CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")
ALLOWED_SCHEMES = frozenset({"http", "https"})
ALLOWED_PORTS = frozenset({80, 443})
ALLOWED_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})


class URLFetchError(ScanValidationError):
    """Base exception for URL fetching and validation failures."""
    pass


class SSRFError(URLFetchError):
    """Raised when a URL targets a private, loopback, link-local, or restricted address."""
    pass


class URLTimeoutError(URLFetchError):
    """Raised when URL fetching times out."""
    pass


class URLSizeLimitError(URLFetchError):
    """Raised when fetched content exceeds maximum byte limit."""
    pass


class URLContentTypeError(URLFetchError):
    """Raised when response content-type is unsupported."""
    pass


class URLExtractionError(URLFetchError):
    """Raised when the fetched page does not contain sufficient readable text."""
    pass


def is_safe_public_ip(ip_str: str) -> bool:
    """
    Checks whether an IP address string is a safe, routable public address.
    Rejects loopback, private, link-local, multicast, reserved, unspecified,
    CGNAT (100.64.0.0/10), and IPv4-mapped IPv6 representations of unsafe addresses.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    # Check IPv4 and standard IPv6 flags
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False

    # Check IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1)
    if ip.version == 6 and ip.ipv4_mapped is not None:
        mapped = ip.ipv4_mapped
        if (
            mapped.is_loopback
            or mapped.is_private
            or mapped.is_link_local
            or mapped.is_multicast
            or mapped.is_reserved
            or mapped.is_unspecified
            or mapped in CGNAT_NETWORK
        ):
            return False

    # Check Carrier-Grade NAT (RFC 6598)
    if ip.version == 4 and ip in CGNAT_NETWORK:
        return False

    return True


def validate_url_syntax(url_str: str) -> tuple[str, str, int, str]:
    """
    Validates URL scheme, credentials, hostname, and port.
    Normalizes 'www.' prefixes to 'https://www.'.
    Returns: (scheme, idna_hostname, port, path_and_query)
    """
    cleaned_url = url_str.strip()
    if cleaned_url.lower().startswith("www."):
        cleaned_url = "https://" + cleaned_url

    try:
        parsed = urlparse(cleaned_url)
    except Exception as exc:
        raise URLFetchError(f"Malformed URL: {exc}") from exc

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise URLFetchError(
            f"Unsupported URL scheme '{scheme}'. Only HTTP and HTTPS URLs are supported."
        )

    if parsed.username or parsed.password:
        raise URLFetchError("URL must not contain embedded user credentials.")

    hostname = parsed.hostname
    if not hostname:
        raise URLFetchError("URL is missing a valid hostname.")

    if any(c.isspace() or unicodedata.category(c) == "Cc" for c in hostname):
        raise URLFetchError("Malformed hostname in URL.")

    normalized_host = hostname.strip().rstrip(".").lower()
    if not normalized_host:
        raise URLFetchError("Malformed hostname in URL.")

    is_ip_literal = False
    try:
        ipaddress.ip_address(normalized_host)
        is_ip_literal = True
    except ValueError:
        pass

    if is_ip_literal and not is_safe_public_ip(normalized_host):
        raise SSRFError(
            f"Access to restricted or non-public address ({normalized_host}) is blocked."
        )

    try:
        idna_host = normalized_host.encode("idna").decode("ascii")
    except Exception as exc:
        raise URLFetchError("Malformed or unparseable hostname in URL.") from exc

    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80
    elif port not in ALLOWED_PORTS:
        raise URLFetchError(
            f"Port {port} is not permitted. Only standard web ports (80, 443) are supported."
        )

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    return scheme, idna_host, port, path


class JobHTMLTextExtractor(HTMLParser):
    """
    Extracts visible readable text from HTML markup without executing scripts.
    Discards script, style, noscript, svg, canvas, template, and head contents.
    """
    def __init__(self):
        super().__init__()
        self._ignore_tags = {
            "script", "style", "noscript", "svg", "canvas", "template", "head"
        }
        self._current_ignored = 0
        self._pieces: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        tag_lower = tag.lower()
        if tag_lower in self._ignore_tags:
            self._current_ignored += 1
        elif tag_lower == "body":
            # In HTML5, </head> is optional before <body>.
            # Reset ignored state when body tag begins.
            self._current_ignored = 0
            self._pieces.append("\n")
        elif tag_lower in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()
        if tag_lower in self._ignore_tags:
            self._current_ignored = max(0, self._current_ignored - 1)
        elif tag_lower in ("p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"):
            self._pieces.append("\n")

    def handle_data(self, data: str):
        if self._current_ignored == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw_text = "".join(self._pieces)
        lines = [line.strip() for line in raw_text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        return cleaned


def extract_readable_text(
    raw_bytes: bytes,
    content_type: str,
    charset: str | None = None,
) -> str:
    """
    Decodes response bytes and extracts readable text depending on Content-Type.
    Honors declared charset when valid, sanitizes null bytes and inappropriate
    control characters, and bounds output to URL_FETCH_MAX_TEXT_LENGTH.
    """
    text_content: str | None = None

    # 1. Try declared response charset if valid and available
    if charset:
        try:
            import codecs
            codecs.lookup(charset)
            text_content = raw_bytes.decode(charset)
        except (LookupError, UnicodeDecodeError):
            text_content = None

    # 2. Fall back to UTF-8 default
    if text_content is None:
        try:
            text_content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # 3. Fall back to latin-1
                text_content = raw_bytes.decode("latin-1")
            except Exception:
                text_content = raw_bytes.decode("utf-8", errors="replace")

    # M-2: Sanitize control characters (strip null bytes and non-whitespace Cc category characters)
    text_content = "".join(
        c for c in text_content
        if not (unicodedata.category(c) == "Cc" and c not in ("\n", "\r", "\t"))
    )

    content_type_lower = content_type.lower()
    if "text/plain" in content_type_lower:
        lines = [line.strip() for line in text_content.splitlines()]
        extracted = "\n".join(line for line in lines if line)
    else:
        extractor = JobHTMLTextExtractor()
        try:
            extractor.feed(text_content)
            extracted = extractor.get_text()
        except Exception as exc:
            raise URLExtractionError(f"Failed to parse HTML content: {exc}") from exc

    # Final sanitization pass on extracted text
    extracted = "".join(
        c for c in extracted
        if not (unicodedata.category(c) == "Cc" and c not in ("\n", "\r", "\t"))
    )

    # Enforce minimum readable text requirement
    non_ws_count = sum(1 for c in extracted if not c.isspace())
    if non_ws_count < settings.URL_FETCH_MIN_TEXT_LENGTH:
        raise URLExtractionError(
            "The webpage did not contain readable job posting text (it may be blank or require JavaScript to render). Please copy and paste the job description text directly."
        )

    # Cap text length
    if len(extracted) > settings.URL_FETCH_MAX_TEXT_LENGTH:
        extracted = extracted[:settings.URL_FETCH_MAX_TEXT_LENGTH]

    return extracted


class PinningSyncBackend(httpcore.SyncBackend):
    """
    Custom httpcore network backend that enforces connection-time DNS resolution,
    validates every resolved IP address, fails closed on any non-public address,
    and pins the TCP connection to a validated IP without a secondary DNS lookup.
    """
    def __init__(self, dns_resolver: Callable[[str, int], list[tuple[int, str, int]]] | None = None):
        super().__init__()
        self._dns_resolver = dns_resolver

    def _resolve(self, host: str, port: int) -> list[tuple[int, str, int]]:
        try:
            if self._dns_resolver is not None:
                records = self._dns_resolver(host, port)
            else:
                addrinfo = socket.getaddrinfo(
                    host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
                )
                records = [
                    (family, sockaddr[0], sockaddr[1])
                    for family, socktype, proto, canonname, sockaddr in addrinfo
                ]
        except (socket.gaierror, OSError) as exc:
            raise httpcore.ConnectError(f"DNS resolution failed for '{host}': {exc}") from exc

        if not records:
            raise httpcore.ConnectError(f"No DNS records found for '{host}'")
        return records

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore._backends.sync.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        # Check if host is already an IP address
        try:
            ipaddress.ip_address(host)
            is_ip_literal = True
        except ValueError:
            is_ip_literal = False

        if is_ip_literal:
            if not is_safe_public_ip(host):
                raise SSRFError(
                    f"Access to restricted or non-public address ({host}) is blocked."
                )
            target_ips = [(socket.AF_INET if ":" not in host else socket.AF_INET6, host, port)]
        else:
            # Connection-time DNS resolution
            records = self._resolve(host, port)
            # Validate EVERY resolved IP address. If ANY address is non-public, FAIL CLOSED!
            for family, ip_str, target_port in records:
                if not is_safe_public_ip(ip_str):
                    raise SSRFError(
                        f"Access to restricted or non-public address ({ip_str}) for host '{host}' is blocked."
                    )
            target_ips = records

        # Connect directly to the validated IP literal
        last_error = None
        for family, ip_str, target_port in target_ips:
            try:
                # Passing ip_str to super().connect_tcp establishes the socket directly to the IP literal
                return super().connect_tcp(
                    host=ip_str,
                    port=target_port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                last_error = exc
                continue

        if last_error:
            if isinstance(last_error, httpcore.ConnectTimeout):
                raise last_error
            raise httpcore.ConnectError(f"Failed to connect to any resolved address for '{host}'") from last_error

        raise httpcore.ConnectError(f"No connectable addresses found for '{host}'")


class HTTPCoreResponseStream(httpx.SyncByteStream):
    """
    Public httpx.SyncByteStream wrapper for httpcore response streams,
    avoiding reliance on private httpx transport classes.
    """
    def __init__(self, stream: Iterable[bytes]) -> None:
        self._stream = stream

    def __iter__(self):
        for part in self._stream:
            yield part

    def close(self) -> None:
        if hasattr(self._stream, "close"):
            self._stream.close()


class PinningTransport(httpx.BaseTransport):
    """
    HTTPX BaseTransport using an internal httpcore.ConnectionPool backed by
    PinningSyncBackend, with zero connection pooling across requests.
    """
    def __init__(
        self,
        backend: PinningSyncBackend | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ):
        super().__init__()
        self._backend = backend or PinningSyncBackend()
        self._ssl_context = ssl_context or ssl.create_default_context()
        self._pool = httpcore.ConnectionPool(
            ssl_context=self._ssl_context,
            network_backend=self._backend,
            max_connections=1,
            max_keepalive_connections=0,
            keepalive_expiry=0,
            retries=0,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        req = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        try:
            resp = self._pool.handle_request(req)
        except SSRFError:
            raise
        except (httpcore.ConnectTimeout, httpcore.ReadTimeout) as e:
            raise URLTimeoutError("Request timed out while connecting to server.") from e
        except (httpcore.ConnectError, httpcore.NetworkError) as e:
            raise URLFetchError(f"Network error while connecting to server: {e}") from e

        return httpx.Response(
            status_code=resp.status,
            headers=resp.headers,
            stream=HTTPCoreResponseStream(resp.stream),
            extensions=resp.extensions,
        )

    def close(self):
        self._pool.close()


def fetch_job_url(
    url_str: str,
    transport: httpx.BaseTransport | None = None,
    max_redirects: int = settings.URL_FETCH_MAX_REDIRECTS,
    max_bytes: int = settings.URL_FETCH_MAX_BYTES,
    timeout_total: float = settings.URL_FETCH_TIMEOUT_SECONDS,
    timeout_connect: float = settings.URL_FETCH_CONNECT_TIMEOUT_SECONDS,
    timeout_read: float = settings.URL_FETCH_READ_TIMEOUT_SECONDS,
) -> tuple[str, str]:
    """
    Fetches job posting content from a URL safely.
    Enforces scheme validation, SSRF checks, DNS pinning, manual redirects,
    streaming body limits, Content-Type verification, and HTML text extraction.

    Returns: (extracted_text, final_url)
    """
    current_url = url_str.strip()
    redirect_count = 0
    timeout_config = httpx.Timeout(timeout_total, connect=timeout_connect, read=timeout_read)
    active_transport = transport or PinningTransport()

    try:
        with httpx.Client(
            transport=active_transport,
            trust_env=False,
            timeout=timeout_config,
            follow_redirects=False,
            headers={"User-Agent": "AIJobScamDetector/1.0 (Job Verification Bot)"},
        ) as client:
            while True:
                scheme, host, port, path = validate_url_syntax(current_url)
                formatted_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
                canonical_target = f"{scheme}://{formatted_host}:{port}{path}" if (
                    (scheme == "http" and port != 80) or (scheme == "https" and port != 443)
                ) else f"{scheme}://{formatted_host}{path}"

                request = client.build_request("GET", canonical_target)
                response = client.send(request, stream=True)

                try:
                    if response.is_redirect:
                        redirect_count += 1
                        if redirect_count > max_redirects:
                            raise URLFetchError(
                                f"Too many redirects (exceeded limit of {max_redirects})."
                            )
                        location = response.headers.get("location")
                        if not location:
                            raise URLFetchError("Redirect response missing Location header.")
                        next_url = urljoin(current_url, location)
                        next_scheme = (urlparse(next_url).scheme or "").lower()
                        if scheme == "https" and next_scheme == "http":
                            raise URLFetchError(
                                "Insecure redirect downgrade from HTTPS to HTTP is not permitted."
                            )
                        current_url = next_url
                        continue

                    if response.status_code != 200:
                        raise URLFetchError(
                            f"Webpage returned unexpected HTTP status {response.status_code}."
                        )

                    raw_ct = response.headers.get("content-type", "")
                    content_type = raw_ct.lower().split(";")[0].strip()
                    if not content_type:
                        content_type = "text/html"

                    # Parse declared charset if present
                    declared_charset: str | None = None
                    if ";" in raw_ct:
                        for part in raw_ct.split(";")[1:]:
                            kv = part.strip().split("=", 1)
                            if len(kv) == 2 and kv[0].strip().lower() == "charset":
                                declared_charset = kv[1].strip().strip("'\"").strip()
                                break

                    if content_type not in ALLOWED_CONTENT_TYPES and not content_type.startswith("text/"):
                        raise URLContentTypeError(
                            f"Unsupported page content type '{content_type}'. Only HTML and plain text pages can be scanned."
                        )

                    total_bytes = 0
                    chunks: list[bytes] = []
                    for chunk in response.iter_bytes(chunk_size=8192):
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            raise URLSizeLimitError(
                                f"Webpage exceeds maximum allowed size of {max_bytes // (1024 * 1024)} MiB."
                            )
                        chunks.append(chunk)

                    raw_body = b"".join(chunks)

                finally:
                    response.close()

                extracted_text = extract_readable_text(raw_body, content_type, charset=declared_charset)
                return extracted_text, current_url

    except httpx.TimeoutException as exc:
        raise URLTimeoutError("Request timed out while retrieving webpage.") from exc
    except (SSRFError, URLFetchError):
        raise
    except Exception as exc:
        raise URLFetchError(f"Failed to fetch webpage: {exc}") from exc
