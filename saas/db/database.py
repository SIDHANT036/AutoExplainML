from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# =========================
# IMPORTANT: MOVE TO ENV LATER
# =========================
DATABASE_URL = "postgresql+psycopg2://postgres:password@localhost:5432/autoexplainml"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()