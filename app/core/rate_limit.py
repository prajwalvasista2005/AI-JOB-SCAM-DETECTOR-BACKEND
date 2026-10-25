from collections import defaultdict, deque
import math
import threading
import time
from typing import Callable

from fastapi import Depends, HTTPException, Request, status

from app.core.config import settings
from app.core.dependencies import get_current_user
from app.models.user import User


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding-window rate limiter using time.monotonic.
    Tracks timestamps per key and calculates Retry-After on limit violation.
    Periodically performs opportunistic cleanup of inactive keys to prevent
    unbounded memory growth.
    """

    def __init__(
        self,
        time_func: Callable[[], float] | None = None,
        cleanup_interval: int = 1000,
    ):
        self._lock = threading.Lock()
        self._records: dict[str, deque[float]] = defaultdict(deque)
        self._time_func = time_func
        self._cleanup_interval = max(1, cleanup_interval)
        self._check_count = 0

    def _get_time(self) -> float:
        if self._time_func is not None:
            return self._time_func()
        return time.monotonic()

    def _cleanup_inactive_keys(self, cutoff: float) -> None:
        """
        Removes keys whose timestamp deques are empty or whose latest
        timestamp has expired beyond the current window threshold.
        Must be called while holding self._lock.
        """
        keys_to_delete = [
            k for k, timestamps in self._records.items()
            if not timestamps or timestamps[-1] <= cutoff
        ]
        for k in keys_to_delete:
            self._records.pop(k, None)

    def check(self, key: str, limit: int, window_seconds: float) -> None:
        """
        Atomically checks if the limit has been reached for the given key in the
        sliding window. If exceeded, raises HTTP 429 with Retry-After header.
        Otherwise, records the current timestamp.
        """
        now = self._get_time()
        cutoff = now - window_seconds

        with self._lock:
            # Opportunistic cleanup to prune inactive keys and bound memory
            self._check_count += 1
            if self._check_count >= self._cleanup_interval:
                self._check_count = 0
                self._cleanup_inactive_keys(cutoff)

            timestamps = self._records[key]

            # Evict timestamps outside the sliding window
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            # Check limit
            if len(timestamps) >= limit:
                earliest = timestamps[0]
                # Seconds until the earliest request expires from the window
                retry_after = max(1, math.ceil((earliest + window_seconds) - now))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                    headers={"Retry-After": str(retry_after)},
                )

            # Record this request
            timestamps.append(now)

    def reset(self) -> None:
        """Clears all in-memory rate-limit state (useful for test isolation)."""
        with self._lock:
            self._records.clear()
            self._check_count = 0


# Global limiter instance for application routes
limiter = SlidingWindowRateLimiter()


def get_client_ip(request: Request) -> str:
    """
    Resolves client IP securely.
    Ignores X-Forwarded-For by default to prevent header spoofing.
    Only inspects X-Forwarded-For if the direct peer is in settings.TRUSTED_PROXIES.
    """
    direct_ip = request.client.host if request.client else "127.0.0.1"

    # Only trust proxy headers if trusted proxies are configured and direct peer matches
    if settings.TRUSTED_PROXIES and direct_ip in settings.TRUSTED_PROXIES:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For: client, proxy1, proxy2
            # Traverse right-to-left to find the first untrusted IP address
            parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
            for ip in reversed(parts):
                if ip not in settings.TRUSTED_PROXIES:
                    return ip
            if parts:
                return parts[0]

    return direct_ip


def login_rate_limiter(request: Request) -> None:
    """
    Dependency enforcing 5 login requests per 60 seconds per client IP.
    Runs before credential verification and database lookups.
    """
    client_ip = get_client_ip(request)
    limiter.check(f"login:ip:{client_ip}", limit=5, window_seconds=60.0)


def scan_rate_limiter(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency enforcing 10 scan requests per 60 seconds per authenticated user ID.
    Resolves user from JWT first (unauthenticated requests fail with 401 without
    consuming quota). Runs before scan creation, rule evaluation, or Gemini calls.
    """
    limiter.check(f"scan:user:{current_user.id}", limit=10, window_seconds=60.0)
    return current_user
