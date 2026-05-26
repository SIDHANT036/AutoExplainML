from jose import jwt, JWTError
from datetime import datetime, timedelta
import os

# =========================
# SECURITY CONFIG (PROD SAFE)
# =========================
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_IN_PROD_IMMEDIATELY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# =========================
# CREATE TOKEN
# =========================
def create_access_token(data: dict, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES):
    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# =========================
# VERIFY TOKEN
# =========================
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # basic validation guardrails
        if payload.get("type") != "access":
            return None

        return payload

    except JWTError:
        return None