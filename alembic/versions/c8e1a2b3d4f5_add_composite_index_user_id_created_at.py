"""add composite index user_id created_at to scans

Revision ID: c8e1a2b3d4f5
Revises: f3a1b2c4d5e6
Create Date: 2026-10-31 15:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8e1a2b3d4f5'
down_revision: Union[str, Sequence[str], None] = 'f3a1b2c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create composite index on (user_id, created_at) for optimized history queries."""
    op.create_index(
        'ix_scans_user_id_created_at',
        'scans',
        ['user_id', 'created_at'],
        unique=False,
    )


def downgrade() -> None:
    """Drop composite index on (user_id, created_at)."""
    op.drop_index('ix_scans_user_id_created_at', table_name='scans')
