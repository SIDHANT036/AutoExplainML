from passlib.context import CryptContext

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12
)

# =========================
# PASSWORD UTILITIES
# =========================
def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not plain_password or not hashed_password:
        return False

    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    if not password:
        raise ValueError("Password cannot be empty")

    # prevent bcrypt 72-byte crash edge case
    password = password[:72]

    return pwd_context.hash(password)


# =========================
# FAKE DB (DEV ONLY)
# Replace with PostgreSQL later
# =========================
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$8DNDHR.N04QAAe1IvzBDXuf71fZ2thTnUJedp.ktuvzmzdiXSk8AG",
        "role": "admin"
    }
}


# =========================
# AUTH FUNCTION
# =========================
def authenticate(username: str, password: str):
    if not username or not password:
        return None

    user = fake_users_db.get(username)

    if not user:
        return None

    if not verify_password(password, user["hashed_password"]):
        return None

    return {
        "username": user["username"],
        "role": user.get("role", "user")
    }