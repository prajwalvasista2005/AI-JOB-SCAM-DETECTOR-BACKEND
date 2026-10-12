import os

# Default to pure-Python psycopg implementation if C extensions are blocked by Windows policies
os.environ.setdefault("PSYCOPG_IMPL", "python")
