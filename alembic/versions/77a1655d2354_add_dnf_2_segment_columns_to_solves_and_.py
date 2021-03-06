"""Add dnf/+2/segment columns to solves, and an index

Revision ID: 77a1655d2354
Revises: ec6b97a7c07d
Create Date: 2021-11-09 16:01:55.734298

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = '77a1655d2354'
down_revision = 'ec6b97a7c07d'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('solves', schema=None) as batch_op:
        batch_op.add_column(sa.Column('dnf', sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('plus_2', sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('segment_time_ms', sqlite.JSON(), nullable=True))
        batch_op.create_index('solve_session_idx', ['session_id', 'created_at'], unique=False)

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('solves', schema=None) as batch_op:
        batch_op.drop_index('solve_session_idx')
        batch_op.drop_column('segment_time_ms')
        batch_op.drop_column('plus_2')
        batch_op.drop_column('dnf')

    # ### end Alembic commands ###
