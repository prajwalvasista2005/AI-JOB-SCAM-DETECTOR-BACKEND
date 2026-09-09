from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import DUMMY_HASH, hash_password, normalize_email, verify_password
from app.models.user import User
from app.schemas.user import UserCreate


def create_user(user: UserCreate, db: Session) -> User | None:
    email = normalize_email(user.email)

    existing_user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if existing_user:
        return None

    hashed_password = hash_password(user.password)
    new_user = User(email=email, password_hash=hashed_password)

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except IntegrityError:
        db.rollback()
        return None
    except Exception:
        db.rollback()
        raise


def authenticate_user(email: str, password: str, db: Session) -> User | None:
    email = normalize_email(email)

    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    if not user:
        verify_password(password, DUMMY_HASH)
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user