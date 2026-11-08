import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import auth, scan, user


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize scan concurrency semaphore bound to this event loop
    app.state.scan_semaphore = asyncio.Semaphore(settings.SCAN_MAX_CONCURRENCY)
    try:
        yield
    finally:
        # Shutdown: Explicitly clear reference to prevent cross-loop reuse
        app.state.scan_semaphore = None


app = FastAPI(title="AI Job Scam Detector", lifespan=lifespan)

if settings.CORS_ORIGINS:
    # Wildcard origins with credentials enabled violate browser security standards
    allow_credentials = "*" not in settings.CORS_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """
    Asynchronous health check probe for container orchestrators and load balancers.
    Executes directly on the asyncio event loop without consuming an AnyIO threadpool token.
    """
    return {"status": "Healthy"}


app.include_router(auth.router)
app.include_router(user.router)
app.include_router(scan.router)