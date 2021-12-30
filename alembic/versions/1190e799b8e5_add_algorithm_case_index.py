"""Add algorithm case index

Revision ID: 1190e799b8e5
Revises: 65829dfd52e1
Create Date: 2021-12-09 23:25:02.849259

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1190e799b8e5'
down_revision = '65829dfd52e1'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('algorithms', schema=None) as batch_op:
        batch_op.create_index('alg_case_idx', ['alg_case_id'], unique=False)

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('algorithms', schema=None) as batch_op:
        batch_op.drop_index('alg_case_idx')

    # ### end Alembic commands ###