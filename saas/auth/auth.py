from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


# Pre-hashed password (generate once and paste here)
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$8DNDHR.N04QAAe1IvzBDXuf71fZ2thTnUJedp.ktuvzmzdiXSk8AG"  # 🔁 replace with real hash
    }
}


def authenticate(username: str, password: str):
    user = fake_users_db.get(username)

    if not user:
        return False

    if not verify_password(password, user["hashed_password"]):
        return False

    return user