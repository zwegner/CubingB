"""Add statistic cache to session/solve table

Revision ID: ec6b97a7c07d
Revises: 
Create Date: 2021-11-08 23:08:31.081080

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = 'ec6b97a7c07d'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('sessions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('cached_stats_current', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('cached_stats_best', sqlite.JSON(), nullable=True))

    with op.batch_alter_table('solves', schema=None) as batch_op:
        batch_op.add_column(sa.Column('cached_stats', sqlite.JSON(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('solves', schema=None) as batch_op:
        batch_op.drop_column('cached_stats')

    with op.batch_alter_table('sessions', schema=None) as batch_op:
        batch_op.drop_column('cached_stats_best')
        batch_op.drop_column('cached_stats_current')

    # ### end Alembic commands ###
