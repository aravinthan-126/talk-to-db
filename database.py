"""
Database helpers for Talk-to-DB
Supports: SQLite, MySQL/MariaDB, PostgreSQL, MongoDB (+ GridFS for files)
"""

import os
import re
import tempfile
import json
from typing import Optional, Tuple, Any, List, Dict, Union
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# SQL engines (original)
# ---------------------------------------------------------------------------

def create_sample_database() -> str:
    """Create a small sample SQLite DB with employees + departments."""
    path = os.path.join(tempfile.gettempdir(), "talk_to_db_sample.db")
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department_id INTEGER,
                salary REAL,
                hire_date TEXT,
                FOREIGN KEY (department_id) REFERENCES departments(id)
            )
        """))
        # seed only if empty
        count = conn.execute(text("SELECT COUNT(*) FROM departments")).scalar()
        if count == 0:
            depts = [
                (1, "Engineering"), (2, "Sales"), (3, "HR"),
                (4, "Marketing"), (5, "Finance"), (6, "Support")
            ]
            conn.execute(
                text("INSERT INTO departments (id, name) VALUES (:id, :name)"),
                [{"id": d[0], "name": d[1]} for d in depts]
            )
            emps = [
                (1, "Alice Johnson", 1, 95000, "2020-03-15"),
                (2, "Bob Smith", 1, 87000, "2019-07-22"),
                (3, "Carol Lee", 2, 72000, "2021-01-10"),
                (4, "David Kim", 2, 78000, "2018-11-05"),
                (5, "Eva Martinez", 3, 65000, "2022-04-18"),
                (6, "Frank Wong", 4, 81000, "2020-09-30"),
                (7, "Grace Patel", 5, 92000, "2017-06-12"),
                (8, "Henry Brown", 1, 105000, "2016-02-28"),
                (9, "Ivy Chen", 6, 58000, "2023-01-05"),
                (10, "Jack Wilson", 3, 61000, "2021-08-14"),
                (11, "Karen Davis", 4, 75000, "2019-12-01"),
                (12, "Leo Garcia", 5, 88000, "2020-05-20"),
                (13, "Mia Rodriguez", 2, 69000, "2022-07-09"),
                (14, "Noah Thompson", 1, 98000, "2018-03-25"),
                (15, "Olivia White", 6, 55000, "2023-03-11"),
                (16, "Paul Harris", 4, 82000, "2019-10-17"),
                (17, "Quinn Clark", 5, 91000, "2017-09-08"),
                (18, "Rachel Lewis", 3, 67000, "2021-11-22"),
                (19, "Sam Walker", 1, 94000, "2020-01-30"),
                (20, "Tina Hall", 2, 76000, "2022-02-14"),
            ]
            conn.execute(
                text("""
                    INSERT INTO employees (id, name, department_id, salary, hire_date)
                    VALUES (:id, :name, :department_id, :salary, :hire_date)
                """),
                [
                    {
                        "id": e[0], "name": e[1], "department_id": e[2],
                        "salary": e[3], "hire_date": e[4]
                    }
                    for e in emps
                ]
            )
    return path


def create_sqlite_engine(path: str) -> Engine:
    return create_engine(f"sqlite:///{path}")


def create_mysql_engine(host: str, port: int, user: str, password: str, database: str) -> Engine:
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url)


def create_postgres_engine(host: str, port: int, user: str, password: str, database: str) -> Engine:
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url)


def test_connection(engine: Engine) -> Tuple[bool, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except Exception as e:
        return False, str(e)


def get_schema_from_engine(engine: Engine) -> str:
    insp = inspect(engine)
    lines = []
    for table in insp.get_table_names():
        cols = insp.get_columns(table)
        col_defs = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
        lines.append(f"TABLE {table}: {col_defs}")
        fks = insp.get_foreign_keys(table)
        for fk in fks:
            lines.append(f"  FK: {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
    return "\n".join(lines) if lines else "No tables found."


def list_tables_from_engine(engine: Engine) -> List[str]:
    try:
        return inspect(engine).get_table_names()
    except Exception:
        return []


def execute_sql(engine: Engine, sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            if result.returns_rows:
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df, None
            return pd.DataFrame(), None
    except Exception as e:
        return None, str(e)


def execute_write_sql(engine: Engine, sql: str) -> Tuple[bool, str]:
    try:
        with engine.begin() as conn:
            result = conn.execute(text(sql))
            return True, f"✅ Success. Rows affected: {result.rowcount}"
    except Exception as e:
        return False, f"❌ Error: {e}"


def save_uploaded_db(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".db"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

try:
    from pymongo import MongoClient
    from pymongo.database import Database
    from gridfs import GridFS
    from bson import ObjectId
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    MongoClient = None
    Database = None
    GridFS = None
    ObjectId = None


def create_mongo_client(
    uri: Optional[str] = None,
    host: str = "localhost",
    port: int = 27017,
    username: Optional[str] = None,
    password: Optional[str] = None,
    auth_source: str = "admin",
) -> "MongoClient":
    if not MONGO_AVAILABLE:
        raise RuntimeError("pymongo is not installed. Run: pip install pymongo")
    # Longer timeout helps Atlas cold starts / DNS
    timeout = 15000
    if uri:
        return MongoClient(
            uri,
            serverSelectionTimeoutMS=timeout,
            connectTimeoutMS=timeout,
        )
    if username and password:
        return MongoClient(
            host=host,
            port=port,
            username=username,
            password=password,
            authSource=auth_source,
            serverSelectionTimeoutMS=timeout,
            connectTimeoutMS=timeout,
        )
    return MongoClient(
        host=host,
        port=port,
        serverSelectionTimeoutMS=timeout,
        connectTimeoutMS=timeout,
    )


def test_mongo_connection(client: "MongoClient") -> Tuple[bool, str]:
    try:
        client.admin.command("ping")
        return True, "OK"
    except Exception as e:
        return False, str(e)


def get_mongo_schema(db: "Database", sample_size: int = 3) -> str:
    """Return a human-readable schema of all collections + sample docs."""
    lines = []
    for coll_name in sorted(db.list_collection_names()):
        if coll_name.startswith("system."):
            continue
        coll = db[coll_name]
        count = coll.estimated_document_count()
        lines.append(f"COLLECTION `{coll_name}` ({count} documents)")
        # sample a few docs to infer fields
        samples = list(coll.find().limit(sample_size))
        if samples:
            # collect field types across samples
            field_info: Dict[str, set] = {}
            for doc in samples:
                for k, v in doc.items():
                    if k == "_id":
                        continue
                    t = type(v).__name__
                    field_info.setdefault(k, set()).add(t)
            fields = ", ".join(f"{k}: {','.join(sorted(ts))}" for k, ts in sorted(field_info.items()))
            lines.append(f"  Fields (inferred): {fields or '(none)'}")
            # show one sample (pretty)
            sample = {k: (str(v) if not isinstance(v, (dict, list)) else v) for k, v in samples[0].items()}
            lines.append(f"  Sample: {json.dumps(sample, default=str)[:400]}...")
        else:
            lines.append("  (empty collection)")
        lines.append("")
    return "\n".join(lines) if lines else "No collections found."


def list_mongo_collections(db: "Database") -> List[str]:
    return [c for c in db.list_collection_names() if not c.startswith("system.")]


def format_mongo_write_result(result: Any) -> str:
    """Turn InsertOneResult / UpdateResult / DeleteResult into a readable message."""
    try:
        # InsertOneResult
        if hasattr(result, "inserted_id"):
            return f"Inserted document with _id = {result.inserted_id}"
        # InsertManyResult
        if hasattr(result, "inserted_ids"):
            ids = list(result.inserted_ids)
            return f"Inserted {len(ids)} document(s). _ids = {ids[:5]}{'...' if len(ids) > 5 else ''}"
        # UpdateResult
        if hasattr(result, "modified_count") and hasattr(result, "matched_count"):
            return (
                f"Matched {result.matched_count} document(s), "
                f"modified {result.modified_count} document(s)"
                + (f", upserted_id = {result.upserted_id}" if getattr(result, "upserted_id", None) else "")
            )
        # DeleteResult
        if hasattr(result, "deleted_count"):
            return f"Deleted {result.deleted_count} document(s)"
        # BulkWriteResult
        if hasattr(result, "bulk_api_result"):
            return f"Bulk write result: {result.bulk_api_result}"
    except Exception:
        pass
    return str(result)


def execute_mongo_query(db: "Database", query_code: str) -> Tuple[Any, Optional[str]]:
    """
    Execute a MongoDB operation generated by the LLM.
    The LLM is instructed to return Python code that uses the variable `db`
    (a pymongo Database) and assigns the result to `result`.
    """
    # Clean common LLM artifacts
    code = query_code.strip()
    code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\s*```$", "", code)
    code = code.strip()

    if not code:
        return None, "Empty code generated by LLM"

    try:
        safe_globals = {
            "db": db,
            "ObjectId": ObjectId,
            "datetime": datetime,
            "result": None,
            "__builtins__": {
                "True": True, "False": False, "None": None,
                "int": int, "float": float, "str": str, "list": list,
                "dict": dict, "set": set, "tuple": tuple,
                "len": len, "range": range, "print": print,
                "isinstance": isinstance, "type": type,
                "min": min, "max": max, "sum": sum, "sorted": sorted,
                "enumerate": enumerate, "zip": zip,
            },
        }
        local_ns = {}
        exec(code, safe_globals, local_ns)

        # Prefer explicit `result`, otherwise take the last assigned name
        result = local_ns.get("result", safe_globals.get("result"))
        if result is None and local_ns:
            # fallback: last value in locals that is not a builtin-like name
            for k in reversed(list(local_ns.keys())):
                if not k.startswith("_"):
                    result = local_ns[k]
                    break

        return result, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def mongo_result_to_dataframe(result: Any) -> Optional[pd.DataFrame]:
    """Convert common MongoDB result types to a DataFrame for display."""
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, list):
        if not result:
            return pd.DataFrame()
        if all(isinstance(x, dict) for x in result):
            cleaned = []
            for doc in result:
                cleaned.append({k: str(v) if isinstance(v, ObjectId) else v for k, v in doc.items()})
            return pd.DataFrame(cleaned)
        return pd.DataFrame({"value": [str(x) for x in result]})
    if isinstance(result, dict):
        cleaned = {k: str(v) if isinstance(v, ObjectId) else v for k, v in result.items()}
        return pd.DataFrame([cleaned])
    # Cursor
    try:
        from pymongo.cursor import Cursor
        if isinstance(result, Cursor):
            return mongo_result_to_dataframe(list(result))
    except Exception:
        pass
    # Write results → return None so caller shows a text message instead
    if any(hasattr(result, attr) for attr in ("inserted_id", "inserted_ids", "modified_count", "deleted_count")):
        return None
    try:
        docs = list(result)
        return mongo_result_to_dataframe(docs)
    except Exception:
        return None


try:
    from gridfs import GridFS
    from gridfs.errors import NoFile
    from bson import ObjectId
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False


def _safe_str(value, default=""):
    if value is None:
        return default
    return str(value)


def upload_file_to_gridfs(db, file_bytes: bytes, filename: str, content_type: str = None, metadata: dict = None) -> str:
    fs = GridFS(db)
    file_id = fs.put(
        file_bytes,
        filename=_safe_str(filename, "unnamed"),
        content_type=_safe_str(content_type, "application/octet-stream"),
        metadata=metadata or {},
        uploadDate=datetime.utcnow(),
    )
    return str(file_id)


def list_gridfs_files(db, limit: int = 100) -> List[dict]:
    """List GridFS files. All string fields are never None."""
    fs = GridFS(db)
    out = []
    for f in fs.find().sort("uploadDate", -1).limit(limit):
        out.append({
            "_id": str(f._id),
            "filename": _safe_str(getattr(f, "filename", None), "unnamed"),
            "length": getattr(f, "length", 0) or 0,
            "contentType": _safe_str(
                getattr(f, "content_type", None) or getattr(f, "contentType", None),
                "application/octet-stream",
            ),
            "uploadDate": getattr(f, "upload_date", None),
        })
    return out


def download_from_gridfs(db, file_id: str) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Returns (data, filename, content_type). Never raises on missing file."""
    try:
        fs = GridFS(db)
        grid_out = fs.get(ObjectId(file_id))
        return (
            grid_out.read(),
            _safe_str(getattr(grid_out, "filename", None), "unnamed"),
            _safe_str(
                getattr(grid_out, "content_type", None) or getattr(grid_out, "contentType", None),
                "application/octet-stream",
            ),
        )
    except Exception:
        return None, None, None


def delete_from_gridfs(db, file_id: str) -> Tuple[bool, str]:
    try:
        fs = GridFS(db)
        fs.delete(ObjectId(file_id))
        return True, "Deleted"
    except Exception as e:
        return False, str(e)
