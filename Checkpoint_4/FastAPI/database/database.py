from sqlmodel import (
    SQLModel,
    Field,
    create_engine,
    Session,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
)
from sqlmodel.ext.asyncio.session import AsyncSession
from datetime import datetime
from typing import Optional, List
from settings.settings import DATA_BASE_PATH


class Logs(SQLModel, table=True):
    __tablename__ = 'logs'
    id: Optional[int] = Field(default=None, primary_key=True)
    input_data: Optional[str]
    result: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    probability: float
    execution_time: float
    token_count: int

DATABASE_URL = f'sqlite+aiosqlite:///{DATA_BASE_PATH}/logs.db'
engine = create_async_engine(DATABASE_URL, echo=True)

async_session_maker = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session():
    async with async_session_maker() as session:
        yield session

async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
