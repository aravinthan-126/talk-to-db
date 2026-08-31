"""
Database helpers for Talk-to-DB.
Supports:
  1. SQLite (file upload or local path)
  2. MySQL / MariaDB (host, port, user, password, db name)
  3. PostgreSQL (host, port, user, password, db name)

All write operations go directly to the real database → changes reflect immediately.
"""

import sqlite3
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

SAMPLE_DB_PATH = Path(__file__).parent / "employees.db"


# ---------------------------------------------------------------------------
# Sample SQLite Database
# ---------------------------------------------------------------------------

def create_sample_database() -> str:
    """Create the sample employees SQLite database (if not exists)."""
    if SAMPLE_DB_PATH.exists():
        return str(SAMPLE_DB_PATH)

    conn = sqlite3.connect(SAMPLE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE departments (
        dept_id INTEGER PRIMARY KEY,
        dept_name TEXT NOT NULL UNIQUE,
        location TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE employees (
        emp_id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE,
        department TEXT,
        salary REAL,
        hire_date TEXT,
        manager_id INTEGER,
        city TEXT,
        FOREIGN KEY (manager_id) REFERENCES employees(emp_id)
    )
    """)

    departments = [
        (1, "HR", "Chennai"),
        (2, "Engineering", "Bangalore"),
        (3, "Sales", "Mumbai"),
        (4, "Finance", "Chennai"),
        (5, "Marketing", "Delhi"),
        (6, "Operations", "Hyderabad"),
    ]
    cursor.executemany("INSERT INTO departments VALUES (?, ?, ?)", departments)

    employees = [
        (1, "Arun", "Kumar", "arun.kumar@company.com", "HR", 55000, "2020-03-15", None, "Chennai"),
        (2, "Priya", "Sharma", "priya.sharma@company.com", "Engineering", 85000, "2019-07-22", None, "Bangalore"),
        (3, "Rajesh", "Patel", "rajesh.patel@company.com", "Sales", 62000, "2021-01-10", None, "Mumbai"),
        (4, "Lakshmi", "Iyer", "lakshmi.iyer@company.com", "Finance", 72000, "2018-11-05", None, "Chennai"),
        (5, "Vikram", "Singh", "vikram.singh@company.com", "Engineering", 95000, "2017-05-18", 2, "Bangalore"),
        (6, "Meena", "Ravi", "meena.ravi@company.com", "HR", 48000, "2022-02-28", 1, "Chennai"),
        (7, "Suresh", "Nair", "suresh.nair@company.com", "Marketing", 58000, "2020-09-12", None, "Delhi"),
        (8, "Anitha", "Krishnan", "anitha.krishnan@company.com", "Engineering", 78000, "2021-06-30", 2, "Bangalore"),
        (9, "Karthik", "Reddy", "karthik.reddy@company.com", "Sales", 67000, "2019-12-01", 3, "Hyderabad"),
        (10, "Divya", "Menon", "divya.menon@company.com", "Finance", 69000, "2020-04-20", 4, "Chennai"),
        (11, "Mohammed", "Ali", "mohammed.ali@company.com", "Operations", 52000, "2021-08-15", None, "Hyderabad"),
        (12, "Sowmya", "Bhat", "sowmya.bhat@company.com", "Engineering", 88000, "2018-03-25", 2, "Bangalore"),
        (13, "Ganesh", "Pillai", "ganesh.pillai@company.com", "HR", 51000, "2022-07-01", 1, "Chennai"),
        (14, "Fatima", "Begum", "fatima.begum@company.com", "Marketing", 61000, "2019-10-08", 7, "Delhi"),
        (15, "Ravi", "Shankar", "ravi.shankar@company.com", "Sales", 74000, "2017-12-14", 3, "Mumbai"),
        (16, "Kavitha", "Srinivasan", "kavitha.s@company.com", "Finance", 81000, "2016-08-30", 4, "Chennai"),
        (17, "Arjun", "Das", "arjun.das@company.com", "Engineering", 92000, "2020-01-19", 2, "Bangalore"),
        (18, "Nisha", "Joshi", "nisha.joshi@company.com", "Operations", 47000, "2023-01-05", 11, "Hyderabad"),
        (19, "Sathish", "Kumar", "sathish.kumar@company.com", "Sales", 59000, "2021-11-22", 3, "Chennai"),
        (20, "Deepa", "Rao", "deepa.rao@company.com", "Marketing", 64000, "2019-04-17", 7, "Bangalore"),
    ]
    cursor.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", employees
    )

    conn.commit()
    conn.close()
    return str(SAMPLE_DB_PATH)


# ---------------------------------------------------------------------------
# Engine / Connection builders
# ---------------------------------------------------------------------------

def create_sqlite_engine(db_path: str) -> Engine:
    return create_engine(f"sqlite:///{db_path}", future=True)


def create_mysql_engine(host: str, port: int, user: str, password: str, database: str) -> Engine:
    # quote password in case it contains special characters
    pwd = quote_plus(password)
    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{database}"
    return create_engine(url, future=True, pool_pre_ping=True)


def create_postgres_engine(host: str, port: int, user: str, password: str, database: str) -> Engine:
    pwd = quote_plus(password)
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{database}"
    return create_engine(url, future=True, pool_pre_ping=True)


def test_connection(engine: Engine) -> Tuple[bool, str]:
    """Try to connect and return (success, message)."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Schema & Tables
# ---------------------------------------------------------------------------

def get_schema_from_engine(engine: Engine) -> str:
    """Extract schema (tables + columns) from any SQLAlchemy engine."""
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            return "No tables found in the database."

        schema_parts = ["Database Schema:\n"]

        for table in tables:
            columns = inspector.get_columns(table)
            pk_constraint = inspector.get_pk_constraint(table)
            pk_cols = set(pk_constraint.get("constrained_columns") or [])

            schema_parts.append(f"TABLE: {table}")
            for col in columns:
                name = col["name"]
                col_type = str(col["type"])
                flags = []
                if name in pk_cols:
                    flags.append("Primary Key")
                if not col.get("nullable", True):
                    flags.append("NOT NULL")
                flag_str = f" ({', '.join(flags)})" if flags else ""
                schema_parts.append(f"  - {name} ({col_type}){flag_str}")
            schema_parts.append("")

            # Sample rows
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(f'SELECT * FROM "{table}" LIMIT 3'))
                    rows = result.fetchall()
                    if rows:
                        col_names = list(result.keys())
                        schema_parts.append(f"  Sample rows from {table}:")
                        schema_parts.append(f"  Columns: {', '.join(col_names)}")
                        for r in rows:
                            schema_parts.append(f"  {tuple(r)}")
                        schema_parts.append("")
            except Exception:
                pass

        return "\n".join(schema_parts)
    except Exception as e:
        return f"Error reading schema: {str(e)}"


def list_tables_from_engine(engine: Engine) -> List[str]:
    try:
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Execute queries (reads go to real DB, writes also go to real DB immediately)
# ---------------------------------------------------------------------------

def execute_sql(engine: Engine, sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Execute a read query. Returns (dataframe, error)."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn)
        return df, None
    except Exception as e:
        return None, str(e)


def execute_write_sql(engine: Engine, sql: str) -> Tuple[bool, str]:
    """
    Execute INSERT / UPDATE / DELETE / etc.
    Changes are committed immediately → they reflect in the real database.
    """
    try:
        with engine.begin() as conn:  # begin() auto-commits on success
            result = conn.execute(text(sql))
            affected = result.rowcount
        return True, f"✅ Query executed successfully. Rows affected: {affected}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


# ---------------------------------------------------------------------------
# Helpers for Streamlit upload
# ---------------------------------------------------------------------------

def save_uploaded_db(uploaded_file) -> str:
    """Save uploaded SQLite file to a temp location and return path."""
    suffix = Path(uploaded_file.name).suffix or ".db"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name
