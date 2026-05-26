from sqlalchemy import Column, Integer, String, Text
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)
    role = Column(String, default="user")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True)
    status = Column(String, default="pending")
    result = Column(Text)