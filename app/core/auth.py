from datetime import datetime, timedelta, timezone
from jose import jwt

from app.core.config import settings
from app.core.dependencies import get_current_user, oauth2_scheme

__all__ = ["create_access_token", "get_current_user", "oauth2_scheme"]


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "exp": expire
    }

    return jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
