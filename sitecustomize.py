import sys
import pysqlite3

# Override built-in sqlite3 with pysqlite3
sys.modules["sqlite3"] = pysqlite3
sys.modules["sqlite"] = pysqlite3.dbapi2
