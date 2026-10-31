from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import auth, scan, user

app = FastAPI(title="AI Job Scam Detector")

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
def health_check() -> dict[str, str]:
    return {"status": "Healthy"}


app.include_router(auth.router)
app.include_router(user.router)
app.include_router(scan.router)