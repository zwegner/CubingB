# This file is part of CubingB.
#
# CubingB is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# CubingB is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with CubingB.  If not, see <https://www.gnu.org/licenses/>.

import contextlib
import threading

import sqlalchemy as sa
from sqlalchemy import (text, Column, Index, Boolean, DateTime, ForeignKey,
        Integer, String, Text, BLOB)
from sqlalchemy.dialects.sqlite import JSON as JSON_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import Mutable
from sqlalchemy.orm import relationship, Session as DBSession, sessionmaker

SESSION_MAKER = None

# Set up default naming convention for indices/constraints/etc. so migrations
# can rename them later 
SQL_NAMING_CONVENTION = {
    'ix': 'ix_%(column_0_label)s',
    'uq': 'uq_%(table_name)s_%(column_0_name)s',
    'ck': 'ck_%(table_name)s_%(constraint_name)s',
    'fk': 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s',
    'pk': 'pk_%(table_name)s',
}
metadata = sa.MetaData(naming_convention=SQL_NAMING_CONVENTION)

Base = declarative_base(metadata=metadata)

# Use SQLAlchemy magic (or "alchemy") to track changes to a JSON object
# See https://docs.sqlalchemy.org/en/14/orm/extensions/mutable.html
class MutableJSON(Mutable, dict):
    @classmethod
    def coerce(cls, attr, value):
        if value is None or isinstance(value, MutableJSON):
            return value
        if isinstance(value, dict):
            return MutableJSON(value)
        raise ValueError()

    def __setitem__(self, key, value):
        self.changed()
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        self.changed()
        dict.__delitem__(self, key)

JSON = MutableJSON.as_mutable(JSON_)

# Base class of DB tables to add id/created_at/updated_at columns everywhere
now = text('datetime("now", "localtime")')
class NiceBase:
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=now)
    updated_at = Column(DateTime, default=now, onupdate=now)

    def __init__(self, **kwargs):
        for [k, v] in kwargs.items():
            assert k in self.__table__.columns, k
            setattr(self, k, v)

class Session(Base, NiceBase):
    __tablename__ = 'sessions'
    sort_id = Column(Integer)
    name = Column(String(128))
    scramble_type = Column(String(64))
    notify_every_n_solves = Column(Integer)
    cached_stats_current = Column(JSON)
    cached_stats_best = Column(JSON)
    cached_stats_best_solve_id = Column(JSON)
    solves = relationship('Solve')

class Solve(Base, NiceBase):
    __tablename__ = 'solves'
    session_id = Column(Integer, ForeignKey(Session.id))
    session = relationship('Session', back_populates='solves')
    scramble = Column(Text)
    reconstruction = Column(Text)
    smart_data = Column(JSON)
    smart_data_raw = Column(BLOB)
    notes = Column(Text)
    time_ms = Column(Integer)
    dnf = Column(Boolean, default=False)
    plus_2 = Column(Boolean, default=False)
    segment_time_ms = Column(JSON)
    # Rolling stats for the containing session
    cached_stats = Column(JSON)

Index('solve_session_idx', Solve.session_id, Solve.created_at)

class AlgCase(Base, NiceBase):
    __tablename__ = 'alg_cases'
    alg_set = Column(String(32))
    alg_group = Column(String(32))
    alg_nb = Column(String(32))
    diagram = Column(String(32))
    diag_type = Column(String(32))
    algs = relationship('Algorithm')

class Algorithm(Base, NiceBase):
    __tablename__ = 'algorithms'
    alg_case_id = Column(Integer, ForeignKey(AlgCase.id))
    case = relationship('AlgCase', back_populates='algs')
    f2l_slot = Column(String(16))
    moves = Column(String(256))
    notes = Column(Text)
    known = Column(Boolean, default=False)
    ignore = Column(Boolean, default=False)

class AlgExecution(Base, NiceBase):
    __tablename__ = 'alg_execs'
    alg_id = Column(Integer, ForeignKey(Algorithm.id))
    alg = relationship('Algorithm')
    time_ms = Column(Integer)
    # Ehh that's probably not worth the space
    #smart_data_raw = Column(BLOB)

class Settings(Base, NiceBase):
    __tablename__ = 'settings'
    current_session_id = Column(Integer, ForeignKey(Session.id))
    current_session = relationship('Session')
    auto_calibrate = Column(Boolean, default=False)

# Make an object usable outside of a DB session
def make_transient(obj):
    return sa.orm.session.make_transient(obj)

# Subclass of DBSession with some convenience functions
class NiceSession(DBSession):
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

THREAD_LOCALS = threading.local()

@contextlib.contextmanager
def get_session():
    if getattr(THREAD_LOCALS, 'session', None):
        yield THREAD_LOCALS.session
    else:
        THREAD_LOCALS.session = session = SESSION_MAKER()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
            THREAD_LOCALS.session = None

def init_db(db_url):
    global SESSION_MAKER
    engine = sa.create_engine(db_url, convert_unicode=True)
    SESSION_MAKER = sessionmaker(autocommit=False, autoflush=False, bind=engine,
            class_=NiceSession)
    Base.metadata.create_all(bind=engine)
