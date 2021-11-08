import contextlib

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, sessionmaker

SESSION_MAKER = None

Base = declarative_base()

# Base class of DB tables to add id/created_at/updated_at columns everywhere
class NiceBase:
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=sa.func.now())
    updated_at = Column(DateTime, default=sa.func.now(), onupdate=sa.func.now())

    def __init__(self, **kwargs):
        for [k, v] in kwargs.items():
            assert k in self.__table__.columns, k
            setattr(self, k, v)

# Gotta disambiguate with sqlalchemy Session I guess
class CubeSession(Base, NiceBase):
    __tablename__ = 'sessions'
    name = Column(String(128))
    scramble_type = Column(String(64))

class Solve(Base, NiceBase):
    __tablename__ = 'solves'
    session_id = Column(Integer, ForeignKey(CubeSession.id))
    session = relationship('CubeSession')
    scramble = Column(Text)
    reconstruction = Column(Text)
    smart_data = Column(JSON)
    notes = Column(Text)
    time_ms = Column(Integer)

class Settings(Base, NiceBase):
    __tablename__ = 'settings'
    current_session_id = Column(Integer, ForeignKey(CubeSession.id))
    current_session = relationship('CubeSession')

# Subclass of Session with some convenience functions
class NiceSession(Session):
    def query_first(self, table, *args, **kwargs):
        return self.query(table).filter_by(*args, **kwargs).first()

    def query_all(self, table, *args, **kwargs):
        return self.query(table).filter_by(*args, **kwargs).all()

    # Insert a new row in this table with the given column values
    def insert(self, table, **kwargs):
        row = table(**kwargs)
        self.add(row)
        # Flushing gets the row an ID from the db
        self.flush()
        return row

    # Update an existing row that matches match_args if one exists, otherwise
    # insert a new one
    def upsert(self, table, match_args, **kwargs):
        for row in self.query_all(table, **match_args):
            for [k, v] in kwargs.items():
                setattr(row, k, v)
            return row
        else:
            return self.insert(table, **match_args, **kwargs)

@contextlib.contextmanager
def get_session():
    session = SESSION_MAKER()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def init_db(db_url):
    global SESSION_MAKER
    engine = sa.create_engine(db_url, convert_unicode=True)
    SESSION_MAKER = sessionmaker(autocommit=False, autoflush=False, bind=engine,
            class_=NiceSession)
    Base.metadata.create_all(bind=engine)
