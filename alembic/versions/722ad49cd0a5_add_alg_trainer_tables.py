"""Add alg trainer tables

Revision ID: 722ad49cd0a5
Revises: 52906ed112d0
Create Date: 2021-11-23 23:19:18.386555

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '722ad49cd0a5'
down_revision = '52906ed112d0'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('algorithms',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('alg_set', sa.String(length=32), nullable=True),
    sa.Column('alg_nb', sa.String(length=32), nullable=True),
    sa.Column('moves', sa.String(length=256), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_algorithms'))
    )
    op.create_table('alg_execs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('alg_id', sa.Integer(), nullable=True),
    sa.Column('time_ms', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['alg_id'], ['algorithms.id'], name=op.f('fk_alg_execs_alg_id_algorithms')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_alg_execs'))
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('alg_execs')
    op.drop_table('algorithms')
    # ### end Alembic commands ###
