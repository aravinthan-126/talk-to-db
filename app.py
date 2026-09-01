"""
Talk-to-DB : Conversational Natural Language Interface to Database

Flow:
  1. Login (admin / admin@123)
  2. Connect Database:
       - SQLite file (upload or path)
       - MySQL / MariaDB (host, port, user, password, database)
       - PostgreSQL (host, port, user, password, database)
       - MongoDB (URI or host/port/user/password/database) + GridFS for files
       - Or skip → Sample SQLite database
  3. Chat with the connected DB
     (All updates go directly to the real database → changes reflect immediately)
  4. (MongoDB only) Upload / list / download images, videos & documents via GridFS
"""

import os
import re
from typing import Optional, Any

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from deep_translator import GoogleTranslator
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy.engine import Engine

from database import (
    # SQL
    create_sample_database,
    create_sqlite_engine,
    create_mysql_engine,
    create_postgres_engine,
    test_connection,
    get_schema_from_engine,
    list_tables_from_engine,
    execute_sql,
    execute_write_sql,
    save_uploaded_db,
    # Mongo
    MONGO_AVAILABLE,
    create_mongo_client,
    test_mongo_connection,
    get_mongo_schema,
    list_mongo_collections,
    execute_mongo_query,
    mongo_result_to_dataframe,
    format_mongo_write_result,
    # GridFS
    upload_file_to_gridfs,
    list_gridfs_files,
    download_from_gridfs,
    delete_from_gridfs,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Talk-to-DB",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_USERNAME = "admin"
VALID_PASSWORD = "admin@123"

# API key is inside the program (not entered in the application UI)
GROQ_API_KEY = "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"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TAMIL_UNICODE_RANGE = re.compile(r"[\u0B80-\u0BFF]")


def contains_tamil(text: str) -> bool:
    return bool(TAMIL_UNICODE_RANGE.search(text))


def translate_to_english(text: str) -> str:
    if not contains_tamil(text):
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        st.warning(f"Translation failed ({e}). Using original text.")
        return text


def is_write_query(sql: str) -> bool:
    sql_upper = sql.strip().upper()
    return any(
        sql_upper.startswith(kw)
        for kw in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE")
    )


def is_mongo_write(code: str) -> bool:
    """Heuristic: does the generated Python code modify data?"""
    lower = code.lower()
    write_keywords = [
        "insert_one", "insert_many", "update_one", "update_many",
        "delete_one", "delete_many", "replace_one", "drop(",
        "drop_collection", "create_index", "create_collection",
        "bulk_write", "find_one_and_update", "find_one_and_delete",
        "find_one_and_replace", "rename_collection",
    ]
    return any(kw in lower for kw in write_keywords)


def get_mongo_db():
    """Safe access to the connected Mongo database (never raises AttributeError)."""
    db = st.session_state.get("mongo_db", None)
    if db is None:
        st.session_state.db_connected = False
        st.session_state.db_type = None
        st.warning("MongoDB is not connected. Please connect again.")
        st.rerun()
    return db


def get_llm(api_key: str, model: str = "llama-3.3-70b-versatile"):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=0.1,
    )


def _extract_code_from_llm(text: str) -> str:
    """Pull clean code out of LLM output (handles thinking models + markdown)."""
    if not text:
        return ""
    raw = str(text).strip()

    # Strip Qwen/DeepSeek style thinking blocks
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"<thinking>[\s\S]*?</thinking>", "", raw, flags=re.IGNORECASE).strip()

    # Prefer fenced code block if present
    m = re.search(r"```(?:python|py|sql)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if m:
        raw = m.group(1).strip()

    # Drop leading language labels
    raw = re.sub(r"^(python|py|sql)\s*\n", "", raw, flags=re.IGNORECASE).strip()

    lines = raw.splitlines()
    code_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            if code_lines:
                code_lines.append("")
            continue
        if s.lower().startswith(("here", "this", "the following", "sure", "okay", "note:")):
            continue
        code_lines.append(line)
    cleaned = "\n".join(code_lines).strip()
    return cleaned or raw.strip()


def _sql_system_message(schema: str) -> str:
    return (
        "You are an expert SQL assistant.\n"
        "Convert the user's natural language question into a valid SQL query for the connected database.\n\n"
        "DATABASE SCHEMA:\n"
        f"{schema}\n\n"
        "RULES:\n"
        "1. Return ONLY the SQL query. No explanations, no markdown, no ```sql blocks.\n"
        "2. Use only the tables and columns that exist in the schema above.\n"
        "3. Prefer SELECT statements. Generate INSERT/UPDATE/DELETE only when user clearly asks to modify data.\n"
        "4. Be case-insensitive for string comparisons when reasonable.\n"
        "5. Handle follow-up questions using conversation history.\n"
        "6. Never generate DROP, ALTER, or CREATE unless explicitly asked.\n"
        "7. Generate SQL compatible with the database type (SQLite / MySQL / PostgreSQL).\n"
    )


def _mongo_system_message(schema: str) -> str:
    examples = (
        "result = list(db.students.find({}))\n"
        'result = list(db.students.find({"age": {"$gt": 18}}).limit(20))\n'
        "result = db.students.count_documents({})\n"
        'result = list(db.students.aggregate([{"$group": {"_id": "$department", "count": {"$sum": 1}}}]))\n'
        'result = db.students.insert_one({"name": "Alice", "age": 20, "course": "CS"})\n'
        'result = db.students.insert_many([{"name": "Bob", "age": 21}, {"name": "Carol", "age": 22}])\n'
        'result = db.students.update_one({"name": "Alice"}, {"$set": {"age": 21}})\n'
        'result = db.students.update_many({"course": "CS"}, {"$set": {"active": True}})\n'
        'result = db.students.delete_one({"name": "Alice"})\n'
        'result = db.students.delete_many({"age": {"$lt": 18}})'
    )
    return (
        "You are an expert MongoDB assistant.\n"
        "Convert the user's natural language request into Python code that uses the pymongo Database object named `db`.\n\n"
        "DATABASE SCHEMA (collections + sample fields):\n"
        f"{schema}\n\n"
        "RULES:\n"
        "1. Return ONLY valid Python code. No explanations, no markdown, no ```python or ``` blocks.\n"
        "2. The code MUST assign the final result to a variable named `result`.\n"
        "3. Use only the collections that exist in the schema above.\n"
        "4. Support full CRUD:\n"
        "   - READ: find, find_one, count_documents, aggregate\n"
        "   - CREATE: insert_one, insert_many\n"
        "   - UPDATE: update_one, update_many, replace_one\n"
        "   - DELETE: delete_one, delete_many\n"
        "5. For find results always wrap with list(...), e.g. result = list(db.students.find({}))\n"
        "6. You may use ObjectId and datetime (they are already available).\n"
        "7. Never drop collections or databases unless the user explicitly asks.\n"
        "8. Keep the code short (1-5 lines). Do not add comments in the final code.\n"
        "9. When the user asks to show / list / get data, use find or aggregate.\n"
        "10. When the user asks to add / insert / create, use insert_one or insert_many.\n"
        "11. When the user asks to change / update / set, use update_one or update_many.\n"
        "12. When the user asks to remove / delete, use delete_one or delete_many.\n\n"
        "Examples of good output:\n"
        f"{examples}"
    )


def generate_sql(llm, question: str, schema: str, chat_history: list) -> str:
    # Build messages directly — avoids ChatPromptTemplate brace-formatting bugs
    messages = [SystemMessage(content=_sql_system_message(schema))]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    if not content or not str(content).strip():
        # some models put text in additional_kwargs
        content = str(getattr(response, "additional_kwargs", {}) or content)
    return _extract_code_from_llm(str(content))


def generate_mongo_code(llm, question: str, schema: str, chat_history: list) -> str:
    # Build messages directly — avoids ChatPromptTemplate brace-formatting bugs
    messages = [SystemMessage(content=_mongo_system_message(schema))]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    if not content or not str(content).strip():
        content = str(getattr(response, "additional_kwargs", {}) or content)
    return _extract_code_from_llm(str(content))


# ---------------------------------------------------------------------------
# 1. Login
# ---------------------------------------------------------------------------

def login_page():
    st.title("🔐 Talk-to-DB Login")
    st.markdown("Please login to continue")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

        if submitted:
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")


# ---------------------------------------------------------------------------
# 2. Database Connection Page
# ---------------------------------------------------------------------------

def db_connection_page():
    st.title("📂 Connect Your Database")
    st.markdown("Choose how you want to connect. All changes made through the app will be written **directly** to the database.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 SQLite File",
        "🐬 MySQL / MariaDB",
        "🐘 PostgreSQL",
        "🍃 MongoDB",
        "🧪 Sample Database"
    ])

    # ----- Tab 1: SQLite -----
    with tab1:
        st.subheader("Connect SQLite Database")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Upload file**")
            uploaded = st.file_uploader(
                "Upload .db / .sqlite file",
                type=["db", "sqlite", "sqlite3"],
                key="sqlite_uploader",
            )
            if uploaded is not None:
                path = save_uploaded_db(uploaded)
                engine = create_sqlite_engine(path)
                ok, msg = test_connection(engine)
                if ok:
                    st.session_state.engine = engine
                    st.session_state.db_type = "sql"
                    st.session_state.db_source = f"SQLite (Uploaded): {uploaded.name}"
                    st.session_state.db_connected = True
                    st.success(f"Connected to **{uploaded.name}**")
                    st.rerun()
                else:
                    st.error(f"Connection failed: {msg}")

        with col_b:
            st.markdown("**Local file path**")
            local_path = st.text_input(
                "Full path",
                placeholder=r"C:\data\mydata.db  or  /home/user/data.db",
                key="sqlite_path",
            )
            if st.button("Connect SQLite Path", key="btn_sqlite_path"):
                if local_path and os.path.isfile(local_path.strip()):
                    engine = create_sqlite_engine(local_path.strip())
                    ok, msg = test_connection(engine)
                    if ok:
                        st.session_state.engine = engine
                        st.session_state.db_type = "sql"
                        st.session_state.db_source = f"SQLite: {os.path.basename(local_path)}"
                        st.session_state.db_connected = True
                        st.success(f"Connected to **{os.path.basename(local_path)}**")
                        st.rerun()
                    else:
                        st.error(f"Connection failed: {msg}")
                else:
                    st.error("File not found. Check the path.")

    # ----- Tab 2: MySQL -----
    with tab2:
        st.subheader("Connect MySQL / MariaDB")
        st.markdown("Enter connection details. Changes will be written **immediately** to this database.")

        c1, c2 = st.columns(2)
        with c1:
            mysql_host = st.text_input("Host / Server Address", value="localhost", key="mysql_host")
            mysql_port = st.number_input("Port Number", value=3306, min_value=1, max_value=65535, key="mysql_port")
            mysql_user = st.text_input("Username", key="mysql_user")
        with c2:
            mysql_password = st.text_input("Password", type="password", key="mysql_pass")
            mysql_db = st.text_input("Database Name", key="mysql_db")

        if st.button("Connect to MySQL", type="primary", key="btn_mysql"):
            if not all([mysql_host, mysql_user, mysql_db]):
                st.error("Host, Username and Database Name are required.")
            else:
                try:
                    engine = create_mysql_engine(
                        host=mysql_host,
                        port=int(mysql_port),
                        user=mysql_user,
                        password=mysql_password or "",
                        database=mysql_db,
                    )
                    ok, msg = test_connection(engine)
                    if ok:
                        st.session_state.engine = engine
                        st.session_state.db_type = "sql"
                        st.session_state.db_source = f"MySQL: {mysql_user}@{mysql_host}:{mysql_port}/{mysql_db}"
                        st.session_state.db_connected = True
                        st.success("MySQL connection successful!")
                        st.rerun()
                    else:
                        st.error(f"Connection failed: {msg}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ----- Tab 3: PostgreSQL -----
    with tab3:
        st.subheader("Connect PostgreSQL")
        st.markdown("Enter connection details. Changes will be written **immediately** to this database.")

        c1, c2 = st.columns(2)
        with c1:
            pg_host = st.text_input("Host / Server Address", value="localhost", key="pg_host")
            pg_port = st.number_input("Port Number", value=5432, min_value=1, max_value=65535, key="pg_port")
            pg_user = st.text_input("Username", key="pg_user")
        with c2:
            pg_password = st.text_input("Password", type="password", key="pg_pass")
            pg_db = st.text_input("Database Name", key="pg_db")

        if st.button("Connect to PostgreSQL", type="primary", key="btn_pg"):
            if not all([pg_host, pg_user, pg_db]):
                st.error("Host, Username and Database Name are required.")
            else:
                try:
                    engine = create_postgres_engine(
                        host=pg_host,
                        port=int(pg_port),
                        user=pg_user,
                        password=pg_password or "",
                        database=pg_db,
                    )
                    ok, msg = test_connection(engine)
                    if ok:
                        st.session_state.engine = engine
                        st.session_state.db_type = "sql"
                        st.session_state.db_source = f"PostgreSQL: {pg_user}@{pg_host}:{pg_port}/{pg_db}"
                        st.session_state.db_connected = True
                        st.success("PostgreSQL connection successful!")
                        st.rerun()
                    else:
                        st.error(f"Connection failed: {msg}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ----- Tab 4: MongoDB -----
    with tab4:
        st.subheader("Connect MongoDB / Atlas")
        if not MONGO_AVAILABLE:
            st.error("`pymongo` is not installed. Run: `pip install pymongo`")
        else:
            st.markdown(
                "Connect to **MongoDB Atlas** (or local MongoDB). "
                "After connecting you can store **images, videos and documents** with GridFS."
            )
            st.info(
                "Atlas tip: In Atlas → Database → Connect → Drivers, copy the URI. "
                "Replace `<password>` with your real password. "
                "If the password has special characters (@ # %), URL-encode them."
            )

            use_uri = st.checkbox("Use connection URI / Atlas connection string", value=True)

            if use_uri:
                mongo_uri = st.text_input(
                    "MongoDB URI",
                    placeholder="mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/",
                    key="mongo_uri_input",
                )
                mongo_db_name = st.text_input(
                    "Database Name",
                    placeholder="e.g. sample1",
                    key="mongo_db_name_uri_input",
                    help="The database name inside Atlas (not the cluster name).",
                )
            else:
                c1, c2 = st.columns(2)
                with c1:
                    mongo_host = st.text_input("Host", value="localhost", key="mongo_host_input")
                    mongo_port = st.number_input("Port", value=27017, min_value=1, max_value=65535, key="mongo_port_input")
                    mongo_user = st.text_input("Username (optional)", key="mongo_user_input")
                with c2:
                    mongo_password = st.text_input("Password (optional)", type="password", key="mongo_pass_input")
                    mongo_db_name = st.text_input("Database Name", key="mongo_db_name_input")
                    mongo_auth_source = st.text_input("Auth Source", value="admin", key="mongo_auth_input")

            if st.button("Connect to MongoDB", type="primary", key="btn_mongo"):
                if not mongo_db_name:
                    st.error("Database Name is required.")
                else:
                    try:
                        if use_uri:
                            if not mongo_uri:
                                st.error("URI is required for Atlas / URI mode.")
                                st.stop()
                            uri = mongo_uri.strip()
                            # common Atlas fix: ensure we don't rely on path DB only
                            client = create_mongo_client(uri=uri)
                        else:
                            client = create_mongo_client(
                                host=mongo_host,
                                port=int(mongo_port),
                                username=mongo_user or None,
                                password=mongo_password or None,
                                auth_source=mongo_auth_source or "admin",
                            )
                        ok, msg = test_mongo_connection(client)
                        if ok:
                            db = client[mongo_db_name.strip()]
                            # touch the db so empty DBs are still usable
                            _ = db.list_collection_names()
                            st.session_state.mongo_client = client
                            st.session_state.mongo_db = db
                            st.session_state.mongo_db_name = mongo_db_name.strip()
                            st.session_state.db_type = "mongo"
                            st.session_state.db_source = f"MongoDB: {mongo_db_name.strip()}"
                            st.session_state.db_connected = True
                            # clear SQL leftovers
                            st.session_state.pop("engine", None)
                            st.success(f"MongoDB connection successful! Database: **{mongo_db_name.strip()}**")
                            st.rerun()
                        else:
                            st.error(f"Connection failed: {msg}")
                    except Exception as e:
                        st.error(f"Error connecting to MongoDB: {e}")
                        st.caption(
                            "Atlas checklist: (1) Network Access allows your IP (or 0.0.0.0/0 for testing), "
                            "(2) Database User password is correct and URL-encoded, "
                            "(3) URI uses mongodb+srv:// for Atlas."
                        )

    # ----- Tab 5: Sample -----
    with tab5:
        st.subheader("Use Sample Employees Database")
        st.info("A ready-made SQLite database with 20 employees and 6 departments. Perfect for testing.")
        st.markdown("If you cancel / don't want to connect your own database, use this option.")

        if st.button("➡️ Continue with Sample Database", type="secondary", key="btn_sample"):
            path = create_sample_database()
            engine = create_sqlite_engine(path)
            st.session_state.engine = engine
            st.session_state.db_type = "sql"
            st.session_state.db_source = "Sample Employees Database (SQLite)"
            st.session_state.db_connected = True
            st.success("Sample database loaded")
            st.rerun()


# ---------------------------------------------------------------------------
# GridFS File Manager (MongoDB only)
# ---------------------------------------------------------------------------

def render_gridfs_panel(db):
    st.subheader("📎 Store Images / Videos / Documents (GridFS)")
    st.caption("Files are stored inside MongoDB using GridFS. You can also refer to them from chat.")

    uploaded = st.file_uploader(
        "Upload image, video or document",
        type=["png", "jpg", "jpeg", "gif", "webp", "mp4", "mov", "avi", "pdf", "doc", "docx", "txt", "csv", "xlsx", "zip"],
        accept_multiple_files=True,
        key="gridfs_uploader",
    )

    if uploaded:
        for f in uploaded:
            if st.button(f"⬆️ Upload **{f.name}**", key=f"up_{f.name}_{f.size}"):
                try:
                    file_id = upload_file_to_gridfs(
                        db,
                        file_bytes=f.getvalue(),
                        filename=f.name or "unnamed",
                        content_type=f.type or "application/octet-stream",
                        metadata={"original_name": f.name, "size": f.size},
                    )
                    st.success(f"Uploaded **{f.name}** → GridFS `_id` = `{file_id}`")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.markdown("---")
    st.markdown("**Stored files**")
    try:
        files = list_gridfs_files(db, limit=100)
        if not files:
            st.info("No files stored yet.")
        else:
            for f in files:
                # ---- SAFE values (never None) ----
                fid = str(f.get("_id", ""))
                fname = f.get("filename") or "unnamed"
                length = f.get("length") or 0
                ctype = f.get("contentType") or f.get("content_type") or "application/octet-stream"
                if not isinstance(ctype, str):
                    ctype = "application/octet-stream"

                cols = st.columns([3, 1, 1, 1])
                cols[0].write(f"**{fname}**  \n`{fid}` · {length:,} bytes · {ctype}")

                # download once per file
                data, dl_name, dl_ctype = download_from_gridfs(db, fid)
                dl_name = dl_name or fname
                dl_ctype = dl_ctype or ctype

                with cols[1]:
                    if data is not None:
                        st.download_button(
                            "⬇️",
                            data=data,
                            file_name=dl_name,
                            mime=dl_ctype,
                            key=f"dl_{fid}",
                        )

                with cols[2]:
                    # SAFE startswith — ctype is always a string
                    if ctype.startswith("image/") and data is not None:
                        try:
                            st.image(data, width=80)
                        except Exception:
                            st.caption("preview n/a")

                with cols[3]:
                    if st.button("🗑️", key=f"del_{fid}"):
                        ok, msg = delete_from_gridfs(db, fid)
                        if ok:
                            st.success("Deleted")
                            st.rerun()
                        else:
                            st.error(msg)
    except Exception as e:
        st.error(f"Could not list files: {e}")
# ---------------------------------------------------------------------------
# 3. Main Chat Application
# ---------------------------------------------------------------------------

def chat_app():
    db_type = st.session_state.get("db_type", "sql")

    # Guard: required connection objects must exist
    if db_type == "sql" and st.session_state.get("engine", None) is None:
        st.session_state.db_connected = False
        st.warning("SQL connection lost. Please reconnect.")
        st.rerun()
    if db_type == "mongo" and st.session_state.get("mongo_db", None) is None:
        st.session_state.db_connected = False
        st.warning("MongoDB connection lost. Please reconnect.")
        st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("💬 Talk-to-DB")
        st.success(f"**{st.session_state.get('db_source', 'Database')}**")
        st.caption("✅ Writes go directly to the real database")

        if db_type == "sql":
            engine: Engine = st.session_state.engine
            tables = list_tables_from_engine(engine)
            st.markdown("**Tables:**")
            if tables:
                st.write(", ".join(f"`{t}`" for t in tables))
            else:
                st.warning("No tables found")
            with st.expander("View Schema"):
                st.code(get_schema_from_engine(engine), language="text")
        else:
            db = get_mongo_db()
            collections = list_mongo_collections(db)
            st.markdown("**Collections:**")
            if collections:
                st.write(", ".join(f"`{c}`" for c in collections))
            else:
                st.warning("No collections found")
            with st.expander("View Schema / Samples"):
                st.code(get_mongo_schema(db), language="text")

        st.divider()
        st.subheader("LLM Settings")
        st.success("✅ Groq API Key loaded from program")
        model = st.selectbox(
            "Model",
            [
                
                "qwen/qwen3.8-27b",
                "qwen/qwen3.6-27b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
            ],
            index=0,
        )

        st.divider()
        if st.button("🔄 Change Database"):
            for k in ["db_connected", "engine", "mongo_client", "mongo_db", "db_type",
                      "messages", "pending_sql", "pending_mongo", "schema", "last_engine_id"]:
                st.session_state.pop(k, None)
            st.rerun()

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.pending_sql = None
            st.session_state.pending_mongo = None
            st.rerun()

        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main area
    st.title("💬 Talk-to-DB")
    st.caption(f"Connected to: **{st.session_state.get('db_source', 'Database')}** · Changes reflect immediately in the DB")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_sql" not in st.session_state:
        st.session_state.pending_sql = None
    if "pending_mongo" not in st.session_state:
        st.session_state.pending_mongo = None

    # ---- MongoDB: GridFS panel ----
    if db_type == "mongo":
        with st.expander("📎 Upload / Manage Images, Videos & Documents (GridFS)", expanded=False):
            render_gridfs_panel(get_mongo_db())

    # Cache schema
    if db_type == "sql":
        engine = st.session_state.engine
        engine_id = id(engine)
        if "schema" not in st.session_state or st.session_state.get("last_engine_id") != engine_id:
            st.session_state.schema = get_schema_from_engine(engine)
            st.session_state.last_engine_id = engine_id
    else:
        db = get_mongo_db()
        schema_key = id(db)
        if "schema" not in st.session_state or st.session_state.get("last_engine_id") != schema_key:
            st.session_state.schema = get_mongo_schema(db)
            st.session_state.last_engine_id = schema_key

    schema = st.session_state.schema
    api_key = GROQ_API_KEY

    # Chat history display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("dataframe") is not None:
                st.dataframe(msg["dataframe"], use_container_width=True)
            if msg.get("sql"):
                with st.expander("Generated SQL"):
                    st.code(msg["sql"], language="sql")
            if msg.get("mongo_code"):
                with st.expander("Generated MongoDB code"):
                    st.code(msg["mongo_code"], language="python")

    # Pending write confirmation (SQL)
    if st.session_state.pending_sql:
        st.warning("⚠️ This query will **modify data in the real database**. Changes will be permanent. Please confirm.")
        st.code(st.session_state.pending_sql, language="sql")

        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Confirm & Execute", type="primary", key="confirm_sql"):
                result = execute_write_sql(st.session_state.engine, st.session_state.pending_sql)
                if isinstance(result, tuple) and len(result) >= 2:
                    success, message = result[0], result[1]
                else:
                    success, message = False, str(result)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": message,
                        "sql": st.session_state.pending_sql,
                    }
                )
                st.session_state.pending_sql = None
                st.session_state.schema = get_schema_from_engine(st.session_state.engine)
                st.rerun()
        with col2:
            if st.button("❌ Cancel", key="cancel_sql"):
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Query cancelled by user. No changes made."}
                )
                st.session_state.pending_sql = None
                st.rerun()
        return

    # Pending write confirmation (Mongo)
    if st.session_state.pending_mongo:
        st.warning("⚠️ This operation will **modify data in the real MongoDB database**. Changes will be permanent. Please confirm.")
        st.code(st.session_state.pending_mongo, language="python")

        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Confirm & Execute", type="primary", key="confirm_mongo"):
                result, error = execute_mongo_query(get_mongo_db(), st.session_state.pending_mongo)
                if error:
                    message = f"❌ Error: {error}"
                else:
                    message = f"✅ {format_mongo_write_result(result)}"
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": message,
                        "mongo_code": st.session_state.pending_mongo,
                    }
                )
                st.session_state.pending_mongo = None
                # refresh schema after write
                st.session_state.schema = get_mongo_schema(get_mongo_db())
                st.session_state.last_engine_id = id(get_mongo_db())
                st.rerun()
        with col2:
            if st.button("❌ Cancel", key="cancel_mongo"):
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Operation cancelled by user. No changes made."}
                )
                st.session_state.pending_mongo = None
                st.rerun()
        return

    # Chat input
    if prompt := st.chat_input("Ask anything about your database... (Tamil + English OK)"):
        if not api_key:
            st.error("Groq API Key is missing in the program.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        english_question = translate_to_english(prompt)
        if english_question != prompt:
            st.info(f"🌐 Translated: *{english_question}*")

        history = []
        for m in st.session_state.messages[-12:]:
            if m["role"] == "user":
                history.append(HumanMessage(content=m["content"]))
            else:
                history.append(AIMessage(content=m.get("content", "")))

        generated = None
        with st.spinner("Generating query..."):
            try:
                llm = get_llm(api_key, model)
                if db_type == "sql":
                    generated = generate_sql(llm, english_question, schema, history)
                else:
                    generated = generate_mongo_code(llm, english_question, schema, history)
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ LLM error: {e}"}
                )
                st.rerun()

        if not generated or not str(generated).strip():
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "❌ The model returned empty output. Try another model (e.g. llama-3.3-70b-versatile) or rephrase the question.",
                }
            )
            st.rerun()

        # ---- SQL path ----
        if db_type == "sql":
            if is_write_query(generated):
                st.session_state.pending_sql = generated
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "This is a **data-modifying** query. It will change the real database. Please confirm below.",
                        "sql": generated,
                    }
                )
                st.rerun()
            else:
                df, error = execute_sql(st.session_state.engine, generated)
                if error:
                    reply = f"❌ SQL Error: {error}"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply, "sql": generated}
                    )
                else:
                    if df is not None and not df.empty:
                        reply = f"Found **{len(df)}** row(s)."
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": reply,
                                "dataframe": df,
                                "sql": generated,
                            }
                        )
                    else:
                        reply = "No matching records found."
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply, "sql": generated}
                        )
                st.rerun()

        # ---- Mongo path ----
        else:
            if is_mongo_write(generated):
                st.session_state.pending_mongo = generated
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "This is a **data-modifying** operation. It will change the real MongoDB database. Please confirm below.",
                        "mongo_code": generated,
                    }
                )
                st.rerun()
            else:
                result, error = execute_mongo_query(get_mongo_db(), generated)
                if error:
                    reply = f"❌ MongoDB Error: {error}"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply, "mongo_code": generated}
                    )
                else:
                    df = mongo_result_to_dataframe(result)
                    if df is not None and not df.empty:
                        reply = f"Found **{len(df)}** document(s)."
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": reply,
                                "dataframe": df,
                                "mongo_code": generated,
                            }
                        )
                    elif df is not None and df.empty:
                        reply = "No matching documents found."
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply, "mongo_code": generated}
                        )
                    else:
                        # write-like result that slipped through, or scalar (count, etc.)
                        if any(hasattr(result, a) for a in ("inserted_id", "inserted_ids", "modified_count", "deleted_count")):
                            reply = f"✅ {format_mongo_write_result(result)}"
                        else:
                            reply = f"Result: `{result}`" if result is not None else "No matching documents found."
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply, "mongo_code": generated}
                        )
                st.rerun()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    try:
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "db_connected" not in st.session_state:
            st.session_state.db_connected = False

        # Step 1: Login
        if not st.session_state.authenticated:
            login_page()
            return

        # Step 2: Connect Database
        if not st.session_state.db_connected:
            db_connection_page()
            return

        # Step 3: Chat
        chat_app()
    except Exception as e:
        st.error("App error — see details below")
        st.exception(e)


if __name__ == "__main__":
    main()
