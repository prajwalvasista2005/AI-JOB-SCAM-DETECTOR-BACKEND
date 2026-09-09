from passlib.context import CryptContext

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# Precomputed dummy hash used to neutralize timing side-channel attacks during authentication
DUMMY_HASH = "$2b$12$e86gUvS.qX9E1M7M4G1e6eJqMh1bW1Kz8YQhK0l9E4.gJzN6m9zKe"


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:
    return pwd_context.verify(
        plain_password,
        hashed_password
    )


def normalize_email(email: str) -> str:
    return email.strip().lower()