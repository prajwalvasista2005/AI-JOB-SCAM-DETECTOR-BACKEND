from fastapi import FastAPI

from app.routers import auth, scan, user

app = FastAPI(title="AI Job Scam Detector")


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "Healthy"}


app.include_router(auth.router)
app.include_router(user.router)
app.include_router(scan.router)