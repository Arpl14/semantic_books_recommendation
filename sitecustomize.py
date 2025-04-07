import sys
import importlib

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite"] = pysqlite3
except ImportError:
    pass
