from sqlmodel import SQLModel, Field, create_engine, Session
from datetime import datetime
from typing import Optional
from settings.settings import DATA_BASE_DIR
import os


class Logs(SQLModel, table=True):
    __tablename__ = "logs"
    id: Optional[int] = Field(default=None, primary_key=True)
    input_data: Optional[str]
    result: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    probability: float
    execution_time: float
    token_count: int


os.makedirs(DATA_BASE_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{DATA_BASE_DIR}/logs.db"
engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
