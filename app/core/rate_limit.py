import asyncio
from collections import defaultdict, deque
import enum
import ipaddress
import logging
import math
import re
import threading
import time
from typing import AsyncGenerator, Callable, Optional
import weakref

from fastapi import Depends, HTTPException, Request, status

from app.core.config import settings
from app.core.dependencies import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

# Threadsafe, weak-keyed dictionary for fallback semaphores outside lifespan.
_fallback_semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()
_fallback_semaphores_lock = threading.Lock()


def get_scan_semaphore(request: Optional[Request] = None) -> asyncio.Semaphore:
    """
    Retrieves the scan concurrency semaphore following strict precedence:
    1. Primary path: If request.app.state.scan_semaphore contains an active lifespan-owned
       semaphore, return it immediately.
    2. Fallback path: If no lifespan-managed semaphore exists (e.g. standalone test executions),
       retrieve or create an event-loop-scoped semaphore stored in _fallback_semaphores.
    3. The fallback never overrides or competes with an active lifespan-owned semaphore.
    4. Does not monkey-patch event loops or inspect private asyncio attributes.
    """
    if request is not None:
        lifespan_sem = getattr(request.app.state, "scan_semaphore", None)
        if lifespan_sem is not None:
            return lifespan_sem

    current_loop = asyncio.get_running_loop()
    with _fallback_semaphores_lock:
        sem = _fallback_semaphores.get(current_loop)
        if sem is None:
            sem = asyncio.Semaphore(settings.SCAN_MAX_CONCURRENCY)
            _fallback_semaphores[current_loop] = sem
        return sem


class SlotState(enum.Enum):
    DEPENDENCY_OWNED = 0
    WORKER_ACCEPTED = 1
    RELEASED = 2


class ScanAdmissionSlot:
    """
    Threadsafe, atomic admission slot token.
    Ensures that a scan slot is released back to the event loop's semaphore exactly once,
    guarantees race-safe ownership transfer between async dependency and synchronous worker,
    and forces the worker thread to abort immediately if cancelled prior to execution.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, sem: asyncio.Semaphore) -> None:
        self._loop = loop
        self._sem = sem
        self._state = SlotState.DEPENDENCY_OWNED
        self._lock = threading.Lock()

    def accept_worker(self) -> bool:
        """
        Called by the synchronous worker thread immediately upon entry.
        Atomically transitions state from DEPENDENCY_OWNED to WORKER_ACCEPTED.
        Returns True if ownership was transferred successfully.
        Returns False if the slot was already RELEASED (e.g. due to cancellation
        prior to worker execution), indicating the worker MUST ABORT immediately
        without executing the scan payload.
        """
        with self._lock:
            if self._state == SlotState.DEPENDENCY_OWNED:
                self._state = SlotState.WORKER_ACCEPTED
                return True
            return False

    def release_from_dependency(self) -> None:
        """
        Called by the async dependency's finally block on the event loop.
        Atomically releases the slot ONLY if still DEPENDENCY_OWNED.
        If the worker already accepted ownership (WORKER_ACCEPTED), this is a no-op.
        """
        self._release_internal(expected_state=SlotState.DEPENDENCY_OWNED)

    def release_from_worker(self) -> None:
        """
        Called by the synchronous worker thread in its finally block.
        Atomically releases the slot ONLY if WORKER_ACCEPTED.
        """
        self._release_internal(expected_state=SlotState.WORKER_ACCEPTED)

    def _release_internal(self, expected_state: SlotState) -> None:
        with self._lock:
            if self._state == expected_state:
                self._state = SlotState.RELEASED
                if not self._loop.is_closed():
                    try:
                        self._loop.call_soon_threadsafe(self._sem.release)
                    except RuntimeError:
                        logger.debug("Event loop closed before scan admission slot could be released")

    @property
    def state(self) -> SlotState:
        with self._lock:
            return self._state


async def scan_admission_control(request: Request) -> AsyncGenerator[ScanAdmissionSlot, None]:
    """
    Asynchronous admission control dependency for POST /scan/.
    Executes on the event loop before thread allocation.
    Rejects excess scans with HTTP 503 without consuming an AnyIO thread token.
    Propagates asyncio.CancelledError cleanly without converting cancellation to HTTP 503.
    """
    loop = asyncio.get_running_loop()
    sem = get_scan_semaphore(request)
    timeout = settings.SCAN_CONCURRENCY_TIMEOUT_SECONDS

    acquired = False
    slot: Optional[ScanAdmissionSlot] = None

    if timeout > 0:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=timeout)
            acquired = True
        except asyncio.TimeoutError:
            pass
        # asyncio.CancelledError is NOT caught; it propagates cleanly
    else:
        if not sem.locked():
            await sem.acquire()
            acquired = True

    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scan processing capacity is temporarily saturated. Please retry shortly.",
            headers={"Retry-After": "5"},
        )

    try:
        slot = ScanAdmissionSlot(loop, sem)
        request.state.scan_admission_slot = slot
        yield slot
    finally:
        if slot is not None:
            slot.release_from_dependency()
        elif acquired:
            try:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(sem.release)
            except RuntimeError:
                pass


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


IPV4_WITH_PORT_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3}):\d+$")
IPV6_WITH_PORT_RE = re.compile(r"^\[([a-fA-F0-9:]+)\]:\d+$")


def strip_port(ip_str: str) -> str:
    cleaned = ip_str.strip()
    m4 = IPV4_WITH_PORT_RE.match(cleaned)
    if m4:
        return m4.group(1)
    m6 = IPV6_WITH_PORT_RE.match(cleaned)
    if m6:
        return m6.group(1)
    return cleaned


def is_ip_trusted(ip_str: str, trusted_list: list[str]) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    for trusted in trusted_list:
        try:
            if "/" in trusted:
                net = ipaddress.ip_network(trusted, strict=False)
                if ip_obj in net:
                    return True
            else:
                target = ipaddress.ip_address(trusted)
                if ip_obj == target:
                    return True
        except ValueError:
            continue
    return False


def get_client_ip(request: Request) -> str:
    """
    Extracts client IP address adhering to reverse proxy rules:
    - Direct connection must be trusted before evaluating X-Forwarded-For.
    - Strips port numbers.
    - Limits proxy chain to 30 hops; raises HTTP 400 if exceeded.
    - Rejects malformed IP addresses or empty hop segments with HTTP 400.
    - Walks backwards right-to-left returning the first untrusted IP.
    - All-trusted proxy fallback: If every hop in the chain is in TRUSTED_PROXIES,
      falls back to direct_ip. (Note: multiple clients behind an all-trusted chain
      will share the proxy's rate-limit identity).
    """
    direct_ip = request.client.host if request.client else "127.0.0.1"
    direct_ip = strip_port(direct_ip)

    if not is_ip_trusted(direct_ip, settings.TRUSTED_PROXIES):
        return direct_ip

    forwarded_header = request.headers.get("x-forwarded-for")
    if not forwarded_header:
        return direct_ip

    raw_hops = [strip_port(h) for h in forwarded_header.split(",")]
    if len(raw_hops) > 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Forwarded-For chain length exceeds maximum permitted hops (30)",
        )

    # Walk backwards from right to left
    for hop in reversed(raw_hops):
        if not hop:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Malformed X-Forwarded-For header: empty hop segment",
            )
        try:
            ipaddress.ip_address(hop)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed IP address in proxy chain: '{hop}'",
            )

        if not is_ip_trusted(hop, settings.TRUSTED_PROXIES):
            return hop

    # All hops are trusted proxies: restore approved fallback to direct peer IP
    return direct_ip


def login_rate_limiter(request: Request) -> None:
    """
    Dependency enforcing 5 login requests per 60 seconds per client IP.
    Runs before credential verification and database lookups.
    """
    client_ip = get_client_ip(request)
    limiter.check(f"login:ip:{client_ip}", limit=5, window_seconds=60.0)


def register_rate_limiter(request: Request) -> None:
    """
    Dependency enforcing 5 user registration requests per 60 seconds per client IP.
    Runs before credential hashing (bcrypt) and database lookups.
    """
    client_ip = get_client_ip(request)
    limiter.check(
        f"register:ip:{client_ip}",
        limit=settings.REGISTER_RATE_LIMIT_MAX_REQUESTS,
        window_seconds=float(settings.REGISTER_RATE_LIMIT_WINDOW_SECONDS),
    )


def scan_rate_limiter(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency enforcing 10 scan requests per 60 seconds per authenticated user ID.
    Resolves user from JWT first (unauthenticated requests fail with 401 without
    consuming quota). Runs before scan creation, rule evaluation, or Gemini calls.
    """
    limiter.check(
        f"scan:user:{current_user.id}",
        limit=settings.SCAN_RATE_LIMIT_MAX_REQUESTS,
        window_seconds=float(settings.SCAN_RATE_LIMIT_WINDOW_SECONDS),
    )
    return current_user


def scan_history_rate_limiter(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency enforcing 30 scan history requests per 60 seconds per authenticated user ID.
    Resolves user from JWT first (unauthenticated requests fail with 401 without
    consuming quota). Runs before scan history queries.
    """
    limiter.check(
        f"scan_history:user:{current_user.id}",
        limit=settings.SCAN_HISTORY_RATE_LIMIT_MAX_REQUESTS,
        window_seconds=float(settings.SCAN_HISTORY_RATE_LIMIT_WINDOW_SECONDS),
    )
    return current_user
