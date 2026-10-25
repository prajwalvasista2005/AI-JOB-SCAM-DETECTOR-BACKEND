"""add analysis_mode to scans

Revision ID: f3a1b2c4d5e6
Revises: e7c2a19b8f04
Create Date: 2026-10-25 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3a1b2c4d5e6'
down_revision: Union[str, Sequence[str], None] = 'e7c2a19b8f04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely: add nullable column, backfill existing rows, enforce non-null."""
    op.add_column('scans', sa.Column('analysis_mode', sa.String(length=50), nullable=True))
    op.execute(sa.text("UPDATE scans SET analysis_mode = 'legacy_unknown' WHERE analysis_mode IS NULL"))
    with op.batch_alter_table('scans') as batch_op:
        batch_op.alter_column('analysis_mode', existing_type=sa.String(length=50), nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('scans') as batch_op:
        batch_op.drop_column('analysis_mode')
