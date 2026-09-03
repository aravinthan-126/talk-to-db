"""
Talk-to-DB : Conversational Natural Language Interface to Database
SQL + MongoDB + GridFS
"""

import os
import re
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from deep_translator import GoogleTranslator
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy.engine import Engine

from database import (
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
    MONGO_AVAILABLE,
    create_mongo_client,
    test_mongo_connection,
    get_mongo_schema,
    list_mongo_collections,
    execute_mongo_query,
    mongo_result_to_dataframe,
    format_mongo_write_result,
    upload_file_to_gridfs,
    list_gridfs_files,
    download_from_gridfs,
    delete_from_gridfs,
    # Excel
    load_excel_workbook,
    save_uploaded_excel,
    get_excel_schema,
    list_excel_sheets,
    execute_excel_code,
    excel_result_to_dataframe,
    is_excel_write,
)

st.set_page_config(
    page_title="Talk-to-DB",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

VALID_USERNAME = "admin"
VALID_PASSWORD = "admin@123"
GROQ_API_KEY = "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"

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
    sql_upper = (sql or "").strip().upper()
    return any(
        sql_upper.startswith(kw)
        for kw in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE")
    )


def is_mongo_write(code: str) -> bool:
    lower = (code or "").lower()
    write_keywords = [
        "insert_one", "insert_many", "update_one", "update_many",
        "delete_one", "delete_many", "replace_one", "drop(",
        "drop_collection", "create_index", "create_collection",
        "bulk_write", "find_one_and_update", "find_one_and_delete",
        "find_one_and_replace", "rename_collection",
    ]
    return any(kw in lower for kw in write_keywords)


def get_mongo_db():
    db = st.session_state.get("mongo_db", None)
    if db is None:
        st.session_state.db_connected = False
        st.session_state.db_type = None
        st.warning("MongoDB is not connected. Please connect again.")
        st.rerun()
    return db


def get_llm(api_key: str, model: str = "qwen/qwen3.8-27b"):
    return ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.1)


def _extract_code_from_llm(text: str) -> str:
    if not text:
        return ""
    raw = str(text).strip()
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"<thinking>[\s\S]*?</thinking>", "", raw, flags=re.IGNORECASE).strip()
    m = re.search(r"```(?:python|py|sql)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
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
        "Convert the user's natural language question into a valid SQL query.\n\n"
        f"DATABASE SCHEMA:\n{schema}\n\n"
        "RULES:\n"
        "1. Return ONLY the SQL query. No explanations, no markdown.\n"
        "2. Use only tables/columns in the schema.\n"
        "3. Prefer SELECT. Use INSERT/UPDATE/DELETE only when asked to modify data.\n"
        "4. Never DROP/ALTER/CREATE unless explicitly asked.\n"
    )


def _mongo_system_message(schema: str) -> str:
    return (
        "You are an expert MongoDB assistant.\n"
        "Convert the user request into Python code using pymongo Database object `db`.\n\n"
        f"DATABASE SCHEMA:\n{schema}\n\n"
        "RULES:\n"
        "1. Return ONLY Python code. No markdown.\n"
        "2. Assign final result to variable `result`.\n"
        "3. For find: result = list(db.collection.find({}))\n"
        "4. Support insert_one, insert_many, update_one, update_many, delete_one, delete_many.\n"
        "5. ObjectId and datetime are available.\n"
        "6. Never drop collections unless explicitly asked.\n"
    )


def generate_sql(llm, question: str, schema: str, chat_history: list) -> str:
    messages = [SystemMessage(content=_sql_system_message(schema))]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    return _extract_code_from_llm(str(content or ""))


def generate_mongo_code(llm, question: str, schema: str, chat_history: list) -> str:
    messages = [SystemMessage(content=_mongo_system_message(schema))]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    return _extract_code_from_llm(str(content or ""))


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def _excel_system_message(schema: str) -> str:
    return f"""You are an expert Excel assistant using pandas + openpyxl.
Convert the user request into short Python code.

WORKBOOK SCHEMA:
{schema}

OBJECTS:
- dfs: dict of DataFrames, e.g. dfs['Sheet1']
- wb: openpyxl Workbook
- path: str file path
- pd: pandas

RULES (MUST FOLLOW):
1. Return ONLY Python code. No markdown.
2. ALWAYS end by setting result = ...
3. For SHOW / LIST / READ: result must be a DataFrame.
   Example: result = dfs['Sheet1']
   Example: result = dfs['Sheet1'][dfs['Sheet1']['Age'] > 20]
4. CREATE (add row):
   dfs['Sheet1'] = pd.concat([dfs['Sheet1'], pd.DataFrame([{{'Col': 'Val'}}])], ignore_index=True)
   with pd.ExcelWriter(path, engine='openpyxl', mode='w') as w:
       for n,d in dfs.items(): d.to_excel(w, sheet_name=n, index=False)
   result = dfs['Sheet1']
5. UPDATE:
   dfs['Sheet1'].loc[dfs['Sheet1']['Name']=='A', 'Salary'] = 90000
   with pd.ExcelWriter(path, engine='openpyxl', mode='w') as w:
       for n,d in dfs.items(): d.to_excel(w, sheet_name=n, index=False)
   result = dfs['Sheet1']
6. DELETE rows:
   dfs['Sheet1'] = dfs['Sheet1'][dfs['Sheet1']['Name']!='A']
   with pd.ExcelWriter(path, engine='openpyxl', mode='w') as w:
       for n,d in dfs.items(): d.to_excel(w, sheet_name=n, index=False)
   result = dfs['Sheet1']
7. FORMULAS:
   ws = wb.active
   ws['C2'] = '=B2*10'
   wb.save(path)
   result = dfs[ws.title]
8. Aggregates as DataFrame:
   result = dfs['Sheet1'].groupby('Dept')['Salary'].mean().reset_index()
"""


def generate_excel_code(llm, question: str, schema: str, chat_history: list) -> str:
    messages = [SystemMessage(content=_excel_system_message(schema))]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    return _extract_code_from_llm(str(content or ""))



def login_page():
    st.title("🔐 Talk-to-DB Login")
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
# Connect DB
# ---------------------------------------------------------------------------

def db_connection_page():
    st.title("📂 Connect Your Database")
    st.markdown("All confirmed writes go **directly** to the real database.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 SQLite File",
        "🐬 MySQL / MariaDB",
        "🐘 PostgreSQL",
        "🍃 MongoDB",
        "📊 Excel File",
        "🧪 Sample Database",
    ])

    with tab1:
        st.subheader("Connect SQLite Database")
        col_a, col_b = st.columns(2)
        with col_a:
            uploaded = st.file_uploader("Upload .db / .sqlite", type=["db", "sqlite", "sqlite3"], key="sqlite_uploader")
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
            local_path = st.text_input("Full path", placeholder=r"C:\data\mydata.db", key="sqlite_path")
            if st.button("Connect SQLite Path", key="btn_sqlite_path"):
                if local_path and os.path.isfile(local_path.strip()):
                    engine = create_sqlite_engine(local_path.strip())
                    ok, msg = test_connection(engine)
                    if ok:
                        st.session_state.engine = engine
                        st.session_state.db_type = "sql"
                        st.session_state.db_source = f"SQLite: {os.path.basename(local_path)}"
                        st.session_state.db_connected = True
                        st.success("Connected")
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("File not found.")

    with tab2:
        st.subheader("Connect MySQL / MariaDB")
        c1, c2 = st.columns(2)
        with c1:
            mysql_host = st.text_input("Host", value="localhost", key="mysql_host")
            mysql_port = st.number_input("Port", value=3306, key="mysql_port")
            mysql_user = st.text_input("Username", key="mysql_user")
        with c2:
            mysql_password = st.text_input("Password", type="password", key="mysql_pass")
            mysql_db = st.text_input("Database Name", key="mysql_db")
        if st.button("Connect to MySQL", type="primary", key="btn_mysql"):
            if not all([mysql_host, mysql_user, mysql_db]):
                st.error("Host, Username and Database Name are required.")
            else:
                try:
                    engine = create_mysql_engine(mysql_host, int(mysql_port), mysql_user, mysql_password or "", mysql_db)
                    ok, msg = test_connection(engine)
                    if ok:
                        st.session_state.engine = engine
                        st.session_state.db_type = "sql"
                        st.session_state.db_source = f"MySQL: {mysql_user}@{mysql_host}:{mysql_port}/{mysql_db}"
                        st.session_state.db_connected = True
                        st.success("MySQL connection successful!")
                        st.rerun()
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(str(e))

    with tab3:
        st.subheader("Connect PostgreSQL")
        c1, c2 = st.columns(2)
        with c1:
            pg_host = st.text_input("Host", value="localhost", key="pg_host")
            pg_port = st.number_input("Port", value=5432, key="pg_port")
            pg_user = st.text_input("Username", key="pg_user")
        with c2:
            pg_password = st.text_input("Password", type="password", key="pg_pass")
            pg_db = st.text_input("Database Name", key="pg_db")
        if st.button("Connect to PostgreSQL", type="primary", key="btn_pg"):
            if not all([pg_host, pg_user, pg_db]):
                st.error("Host, Username and Database Name are required.")
            else:
                try:
                    engine = create_postgres_engine(pg_host, int(pg_port), pg_user, pg_password or "", pg_db)
                    ok, msg = test_connection(engine)
                    if ok:
                        st.session_state.engine = engine
                        st.session_state.db_type = "sql"
                        st.session_state.db_source = f"PostgreSQL: {pg_user}@{pg_host}:{pg_port}/{pg_db}"
                        st.session_state.db_connected = True
                        st.success("PostgreSQL connection successful!")
                        st.rerun()
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(str(e))

    with tab4:
        st.subheader("Connect MongoDB / Atlas")
        if not MONGO_AVAILABLE:
            st.error("`pymongo` is not installed. Run: pip install pymongo")
        else:
            use_uri = st.checkbox("Use connection URI / Atlas string", value=True)
            if use_uri:
                mongo_uri = st.text_input("MongoDB URI", placeholder="mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net/", key="mongo_uri_input")
                mongo_db_name = st.text_input("Database Name", key="mongo_db_name_uri_input")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    mongo_host = st.text_input("Host", value="localhost", key="mongo_host_input")
                    mongo_port = st.number_input("Port", value=27017, key="mongo_port_input")
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
                                st.error("URI is required.")
                                st.stop()
                            client = create_mongo_client(uri=mongo_uri.strip())
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
                            _ = db.list_collection_names()
                            st.session_state.mongo_client = client
                            st.session_state.mongo_db = db
                            st.session_state.mongo_db_name = mongo_db_name.strip()
                            st.session_state.db_type = "mongo"
                            st.session_state.db_source = f"MongoDB: {mongo_db_name.strip()}"
                            st.session_state.db_connected = True
                            st.session_state.pop("engine", None)
                            st.success(f"MongoDB connected: **{mongo_db_name.strip()}**")
                            st.rerun()
                        else:
                            st.error(f"Connection failed: {msg}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab5:
        st.subheader("Connect Excel File (.xlsx / .xls / .csv)")
        st.markdown(
            "Upload a spreadsheet to query data, run calculations, "
            "and add **Excel formulas** (SUM, AVERAGE, IF, VLOOKUP, etc.)."
        )
        up_x = st.file_uploader(
            "Upload Excel / CSV",
            type=["xlsx", "xls", "xlsm", "csv"],
            key="excel_uploader",
        )
        local_x = st.text_input(
            "Or local file path",
            placeholder=r"C:\data\report.xlsx",
            key="excel_path_input",
        )
        if up_x is not None:
            try:
                path = save_uploaded_excel(up_x)
                # CSV → convert to xlsx for formula support
                if path.lower().endswith(".csv"):
                    import pandas as _pd
                    from openpyxl import Workbook
                    df = _pd.read_csv(path)
                    xlsx_path = path.rsplit(".", 1)[0] + ".xlsx"
                    df.to_excel(xlsx_path, index=False, engine="openpyxl")
                    path = xlsx_path
                state = load_excel_workbook(path)
                state["original_name"] = up_x.name
                st.session_state.excel_state = state
                st.session_state.db_type = "excel"
                st.session_state.db_source = f"Excel: {up_x.name}"
                st.session_state.db_connected = True
                st.session_state.pop("engine", None)
                st.session_state.pop("mongo_db", None)
                st.success(
                    f"Loaded **{up_x.name}** · sheets: {', '.join(state['sheets'])}\n\n"
                    f"Working copy saved at: `{path}`\n"
                    "Edits are saved to this working copy. Use **Download Excel** to get the updated file."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load Excel: {e}")
        if st.button("Connect Excel Path", key="btn_excel_path"):
            if local_x and os.path.isfile(local_x.strip()):
                try:
                    path = local_x.strip()
                    if path.lower().endswith(".csv"):
                        import pandas as _pd
                        df = _pd.read_csv(path)
                        xlsx_path = path.rsplit(".", 1)[0] + "_converted.xlsx"
                        df.to_excel(xlsx_path, index=False, engine="openpyxl")
                        path = xlsx_path
                    state = load_excel_workbook(path)
                    state["original_name"] = os.path.basename(path)
                    st.session_state.excel_state = state
                    st.session_state.db_type = "excel"
                    st.session_state.db_source = f"Excel: {os.path.basename(path)}"
                    st.session_state.db_connected = True
                    st.session_state.pop("engine", None)
                    st.session_state.pop("mongo_db", None)
                    st.success(
                        f"Loaded sheets: {', '.join(state['sheets'])}\n"
                        f"File path: `{path}` (edits save here directly)."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
            else:
                st.error("File not found.")

    with tab6:
        st.subheader("Sample Employees Database")
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
# GridFS panel (FIXED — no None.startswith)
# ---------------------------------------------------------------------------

def render_gridfs_panel(db):
    st.subheader("📎 Store Images / Videos / Documents (GridFS)")
    st.caption("Files are stored inside MongoDB using GridFS.")

    uploaded = st.file_uploader(
        "Upload image, video or document",
        type=["png", "jpg", "jpeg", "gif", "webp", "mp4", "mov", "avi", "pdf", "doc", "docx", "txt", "csv", "xlsx", "zip"],
        accept_multiple_files=True,
        key="gridfs_uploader",
    )

    if uploaded:
        for f in uploaded:
            btn_key = f"up_{f.name}_{getattr(f, 'size', 0)}"
            if st.button(f"⬆️ Upload **{f.name}**", key=btn_key):
                try:
                    file_id = upload_file_to_gridfs(
                        db,
                        file_bytes=f.getvalue(),
                        filename=f.name or "unnamed",
                        content_type=f.type or "application/octet-stream",
                        metadata={"original_name": f.name, "size": getattr(f, "size", 0)},
                    )
                    st.success(f"Uploaded **{f.name}** → `_id` = `{file_id}`")
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
                fid = str(f.get("_id", ""))
                fname = f.get("filename") or "unnamed"
                length = f.get("length") or 0
                # CRITICAL FIX: contentType must always be a string
                ctype = f.get("contentType") or f.get("content_type") or "application/octet-stream"
                if not isinstance(ctype, str):
                    ctype = "application/octet-stream"

                cols = st.columns([3, 1, 1, 1])
                cols[0].write(f"**{fname}**  \n`{fid}` · {length:,} bytes · {ctype}")

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
# Chat
# ---------------------------------------------------------------------------

def chat_app():
    db_type = st.session_state.get("db_type", "sql")

    if db_type == "sql" and st.session_state.get("engine") is None:
        st.session_state.db_connected = False
        st.warning("SQL connection lost. Please reconnect.")
        st.rerun()
    if db_type == "mongo" and st.session_state.get("mongo_db") is None:
        st.session_state.db_connected = False
        st.warning("MongoDB connection lost. Please reconnect.")
        st.rerun()
    if db_type == "excel" and st.session_state.get("excel_state") is None:
        st.session_state.db_connected = False
        st.warning("Excel file not loaded. Please reconnect.")
        st.rerun()

    with st.sidebar:
        st.header("💬 Talk-to-DB")
        st.success(f"**{st.session_state.get('db_source', 'Database')}**")
        st.caption("✅ Writes go directly to the real database")

        if db_type == "sql":
            engine: Engine = st.session_state.engine
            tables = list_tables_from_engine(engine)
            st.markdown("**Tables:**")
            st.write(", ".join(f"`{t}`" for t in tables) if tables else "None")
            with st.expander("View Schema"):
                st.code(get_schema_from_engine(engine), language="text")
        elif db_type == "mongo":
            db = get_mongo_db()
            collections = list_mongo_collections(db)
            st.markdown("**Collections:**")
            st.write(", ".join(f"`{c}`" for c in collections) if collections else "None")
            with st.expander("View Schema / Samples"):
                st.code(get_mongo_schema(db), language="text")
        else:
            # Excel
            xs = st.session_state.excel_state
            sheets = list_excel_sheets(xs)
            st.markdown("**Sheets:**")
            st.write(", ".join(f"`{s}`" for s in sheets) if sheets else "None")
            with st.expander("View Schema / Samples"):
                st.code(get_excel_schema(xs), language="text")
            st.caption(f"Working file:\n`{xs.get('path','')}`")
            st.info("Browser uploads cannot overwrite your original file. Download the updated Excel after edits.")
            try:
                with open(xs["path"], "rb") as fh:
                    data = fh.read()
                fname = xs.get("original_name") or os.path.basename(xs["path"])
                if not str(fname).lower().endswith((".xlsx", ".xlsm", ".xls")):
                    fname = str(Path(fname).stem) + ".xlsx"
                st.download_button(
                    "⬇️ Download updated Excel",
                    data=data,
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_excel_sidebar",
                    type="primary",
                )
            except Exception as e:
                st.warning(f"Download unavailable: {e}")

        st.divider()
        st.subheader("LLM Settings")
        st.success("✅ Groq API Key loaded from program")
        model = st.selectbox(
            "Model",
            ["qwen/qwen3.8-27b", "groq/compound-mini", "groq/compound", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"],
            index=0,
        )

        st.divider()
        if st.button("🔄 Change Database"):
            for k in list(st.session_state.keys()):
                if k != "authenticated":
                    st.session_state.pop(k, None)
            st.session_state.db_connected = False
            st.rerun()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.pending_sql = None
            st.session_state.pending_mongo = None
            st.session_state.pending_excel = None
            st.rerun()
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.title("💬 Talk-to-DB")
    st.caption(f"Connected to: **{st.session_state.get('db_source', 'Database')}**")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_sql" not in st.session_state:
        st.session_state.pending_sql = None
    if "pending_mongo" not in st.session_state:
        st.session_state.pending_mongo = None
    if "pending_excel" not in st.session_state:
        st.session_state.pending_excel = None

    if db_type == "mongo":
        with st.expander("📎 Upload / Manage Files (GridFS)", expanded=False):
            render_gridfs_panel(get_mongo_db())

    if db_type == "sql":
        engine = st.session_state.engine
        engine_id = id(engine)
        if "schema" not in st.session_state or st.session_state.get("last_engine_id") != engine_id:
            st.session_state.schema = get_schema_from_engine(engine)
            st.session_state.last_engine_id = engine_id
    elif db_type == "mongo":
        db = get_mongo_db()
        schema_key = id(db)
        if "schema" not in st.session_state or st.session_state.get("last_engine_id") != schema_key:
            st.session_state.schema = get_mongo_schema(db)
            st.session_state.last_engine_id = schema_key
    else:
        xs = st.session_state.excel_state
        schema_key = (xs.get("path"), tuple(xs.get("sheets") or []))
        if "schema" not in st.session_state or st.session_state.get("last_engine_id") != schema_key:
            st.session_state.schema = get_excel_schema(xs)
            st.session_state.last_engine_id = schema_key

    schema = st.session_state.schema
    api_key = GROQ_API_KEY

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
            if msg.get("excel_code"):
                with st.expander("Generated Excel code"):
                    st.code(msg["excel_code"], language="python")

    if st.session_state.pending_sql:
        st.warning("⚠️ This will **modify** the real database. Confirm?")
        st.code(st.session_state.pending_sql, language="sql")
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Confirm & Execute", type="primary", key="confirm_sql"):
                result = execute_write_sql(st.session_state.engine, st.session_state.pending_sql)
                message = result[1] if isinstance(result, tuple) and len(result) >= 2 else str(result)
                st.session_state.messages.append({"role": "assistant", "content": message, "sql": st.session_state.pending_sql})
                st.session_state.pending_sql = None
                st.session_state.schema = get_schema_from_engine(st.session_state.engine)
                st.rerun()
        with col2:
            if st.button("❌ Cancel", key="cancel_sql"):
                st.session_state.messages.append({"role": "assistant", "content": "Cancelled."})
                st.session_state.pending_sql = None
                st.rerun()
        return

    if st.session_state.pending_mongo:
        st.warning("⚠️ This will **modify** MongoDB. Confirm?")
        st.code(st.session_state.pending_mongo, language="python")
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Confirm & Execute", type="primary", key="confirm_mongo"):
                result, error = execute_mongo_query(get_mongo_db(), st.session_state.pending_mongo)
                message = f"❌ Error: {error}" if error else f"✅ {format_mongo_write_result(result)}"
                st.session_state.messages.append({"role": "assistant", "content": message, "mongo_code": st.session_state.pending_mongo})
                st.session_state.pending_mongo = None
                st.session_state.schema = get_mongo_schema(get_mongo_db())
                st.rerun()
        with col2:
            if st.button("❌ Cancel", key="cancel_mongo"):
                st.session_state.messages.append({"role": "assistant", "content": "Cancelled."})
                st.session_state.pending_mongo = None
                st.rerun()
        return

    # Pending Excel write
    if st.session_state.pending_excel:
        st.warning("⚠️ This will **modify the Excel file**. Confirm?")
        st.code(st.session_state.pending_excel, language="python")
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Confirm & Execute", type="primary", key="confirm_excel"):
                result, error, new_state = execute_excel_code(
                    st.session_state.excel_state, st.session_state.pending_excel
                )
                msg = {
                    "role": "assistant",
                    "excel_code": st.session_state.pending_excel,
                }
                if error:
                    msg["content"] = f"❌ Error: {error}"
                else:
                    st.session_state.excel_state = new_state
                    st.session_state.schema = get_excel_schema(new_state)
                    df = excel_result_to_dataframe(result)
                    path = new_state.get("path", "")
                    note = "\n\n💾 Changes are saved to the working file. Click **Download updated Excel** in the sidebar."
                    if df is not None and not df.empty:
                        msg["content"] = f"✅ Updated & saved. Showing **{len(df)}** row(s)." + note
                        msg["dataframe"] = df
                    elif df is not None and df.empty:
                        msg["content"] = "✅ Updated & saved. (empty table)" + note
                        msg["dataframe"] = df
                    else:
                        msg["content"] = f"✅ Excel updated & saved. Result: `{result}`" + note
                st.session_state.messages.append(msg)
                st.session_state.pending_excel = None
                st.rerun()
        with col2:
            if st.button("❌ Cancel", key="cancel_excel"):
                st.session_state.messages.append({"role": "assistant", "content": "Cancelled."})
                st.session_state.pending_excel = None
                st.rerun()
        return

    if prompt := st.chat_input("Ask anything about your database... (Tamil + English OK)"):
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

        with st.spinner("Generating query..."):
            try:
                llm = get_llm(api_key, model)
                if db_type == "sql":
                    generated = generate_sql(llm, english_question, schema, history)
                elif db_type == "mongo":
                    generated = generate_mongo_code(llm, english_question, schema, history)
                else:
                    generated = generate_excel_code(llm, english_question, schema, history)
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ LLM error: {e}"})
                st.rerun()

        if not generated or not str(generated).strip():
            st.session_state.messages.append({"role": "assistant", "content": "❌ Empty model output. Try another model or rephrase."})
            st.rerun()

        if db_type == "sql":
            if is_write_query(generated):
                st.session_state.pending_sql = generated
                st.session_state.messages.append({"role": "assistant", "content": "Data-modifying query — confirm below.", "sql": generated})
            else:
                df, error = execute_sql(st.session_state.engine, generated)
                if error:
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ SQL Error: {error}", "sql": generated})
                elif df is not None and not df.empty:
                    st.session_state.messages.append({"role": "assistant", "content": f"Found **{len(df)}** row(s).", "dataframe": df, "sql": generated})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "No matching records found.", "sql": generated})
            st.rerun()
        elif db_type == "mongo":
            if is_mongo_write(generated):
                st.session_state.pending_mongo = generated
                st.session_state.messages.append({"role": "assistant", "content": "Data-modifying operation — confirm below.", "mongo_code": generated})
            else:
                result, error = execute_mongo_query(get_mongo_db(), generated)
                if error:
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ MongoDB Error: {error}", "mongo_code": generated})
                else:
                    df = mongo_result_to_dataframe(result)
                    if df is not None and not df.empty:
                        st.session_state.messages.append({"role": "assistant", "content": f"Found **{len(df)}** document(s).", "dataframe": df, "mongo_code": generated})
                    elif df is not None and df.empty:
                        st.session_state.messages.append({"role": "assistant", "content": "No matching documents found.", "mongo_code": generated})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": f"Result: `{result}`", "mongo_code": generated})
            st.rerun()
        else:
            # Excel
            if is_excel_write(generated):
                st.session_state.pending_excel = generated
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "This will **modify the Excel file** (data or formulas). Confirm below.",
                    "excel_code": generated,
                })
            else:
                result, error, new_state = execute_excel_code(st.session_state.excel_state, generated)
                if error:
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ Excel Error: {error}", "excel_code": generated})
                else:
                    st.session_state.excel_state = new_state
                    df = excel_result_to_dataframe(result)
                    # Fallback: show first sheet as table if still no df
                    if df is None and new_state.get("dataframes"):
                        first_name = next(iter(new_state["dataframes"]))
                        df = new_state["dataframes"][first_name]
                    if df is not None and not df.empty:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Found **{len(df)}** row(s).",
                            "dataframe": df,
                            "excel_code": generated,
                        })
                    elif df is not None and df.empty:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "No matching rows.",
                            "dataframe": df,
                            "excel_code": generated,
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Result: `{result}`",
                            "excel_code": generated,
                        })
            st.rerun()


def main():
    try:
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "db_connected" not in st.session_state:
            st.session_state.db_connected = False
        if not st.session_state.authenticated:
            login_page()
            return
        if not st.session_state.db_connected:
            db_connection_page()
            return
        chat_app()
    except Exception as e:
        st.error("App error")
        st.exception(e)


if __name__ == "__main__":
    main()
