"""add explanation and indicators to scans

Revision ID: d4e5f6a7b8c9
Revises: b1a5e08f361c
Create Date: 2026-10-24 19:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, Sequence[str], None] = 'b1a5e08f361c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('scans', sa.Column('explanation', sa.Text(), nullable=True))
    op.add_column('scans', sa.Column('indicators', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('scans', 'indicators')
    op.drop_column('scans', 'explanation')
