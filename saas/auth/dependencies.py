from fastapi import Header, HTTPException
from .jwt_handler import verify_token

def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing token")

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid auth scheme")
    except:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    user = verify_token(token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user