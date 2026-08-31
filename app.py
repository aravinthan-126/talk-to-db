"""
Talk-to-DB : Conversational Natural Language Interface to Database

Flow:
  1. Login (admin / admin@123)
  2. Connect Database:
       - SQLite file (upload or path)
       - MySQL / MariaDB (host, port, user, password, database)
       - PostgreSQL (host, port, user, password, database)
       - Or skip → Sample SQLite database
  3. Chat with the connected DB
     (All updates go directly to the real database → changes reflect immediately)
"""

import os
import re
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from deep_translator import GoogleTranslator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
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


def get_llm(api_key: str, model: str = "qwen/qwen3.8-27b"):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=0.1,
    )


def build_sql_prompt(schema: str) -> ChatPromptTemplate:
    system_msg = f"""You are an expert SQL assistant.
Convert the user's natural language question into a valid SQL query for the connected database.

DATABASE SCHEMA:
{schema}

RULES:
1. Return ONLY the SQL query. No explanations, no markdown, no ```sql blocks.
2. Use only the tables and columns that exist in the schema above.
3. Prefer SELECT statements. Generate INSERT/UPDATE/DELETE only when user clearly asks to modify data.
4. Be case-insensitive for string comparisons when reasonable.
5. Handle follow-up questions using conversation history.
6. Never generate DROP, ALTER, or CREATE unless explicitly asked.
7. Generate SQL compatible with the database type (SQLite / MySQL / PostgreSQL).
"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


def generate_sql(llm, question: str, schema: str, chat_history: list) -> str:
    prompt = build_sql_prompt(schema)
    chain = prompt | llm
    response = chain.invoke(
        {
            "question": question,
            "chat_history": chat_history,
        }
    )
    sql = response.content.strip()
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^```\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


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

    st.caption("Default credentials → Username: `admin`  |  Password: `admin@123`")


# ---------------------------------------------------------------------------
# 2. Database Connection Page
# ---------------------------------------------------------------------------

def db_connection_page():
    st.title("📂 Connect Your Database")
    st.markdown("Choose how you want to connect. All changes made through the app will be written **directly** to the database.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 SQLite File",
        "🐬 MySQL / MariaDB",
        "🐘 PostgreSQL",
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
                        st.session_state.db_source = f"PostgreSQL: {pg_user}@{pg_host}:{pg_port}/{pg_db}"
                        st.session_state.db_connected = True
                        st.success("PostgreSQL connection successful!")
                        st.rerun()
                    else:
                        st.error(f"Connection failed: {msg}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ----- Tab 4: Sample -----
    with tab4:
        st.subheader("Use Sample Employees Database")
        st.info("A ready-made SQLite database with 20 employees and 6 departments. Perfect for testing.")
        st.markdown("If you cancel / don't want to connect your own database, use this option.")

        if st.button("➡️ Continue with Sample Database", type="secondary", key="btn_sample"):
            path = create_sample_database()
            engine = create_sqlite_engine(path)
            st.session_state.engine = engine
            st.session_state.db_source = "Sample Employees Database (SQLite)"
            st.session_state.db_connected = True
            st.success("Sample database loaded")
            st.rerun()


# ---------------------------------------------------------------------------
# 3. Main Chat Application
# ---------------------------------------------------------------------------

def chat_app():
    engine: Engine = st.session_state.engine

    # Sidebar
    with st.sidebar:
        st.header("💬 Talk-to-DB")
        st.success(f"**{st.session_state.get('db_source', 'Database')}**")
        st.caption("✅ Writes go directly to the real database")

        tables = list_tables_from_engine(engine)
        st.markdown("**Tables:**")
        if tables:
            st.write(", ".join(f"`{t}`" for t in tables))
        else:
            st.warning("No tables found")

        with st.expander("View Schema"):
            st.code(get_schema_from_engine(engine), language="text")

        st.divider()
        st.subheader("LLM Settings")
        # API key is in the program – no manual input
        st.success("✅ Groq API Key loaded from program")
        model = st.selectbox(
            "Model",
            [
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
                "groq/compound",
                "allam-2-7b",
            ],
            index=0,
        )

        st.divider()
        if st.button("🔄 Change Database"):
            st.session_state.db_connected = False
            st.session_state.engine = None
            st.session_state.messages = []
            st.session_state.pending_sql = None
            st.rerun()

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.pending_sql = None
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

    # Cache schema
    engine_id = id(engine)
    if "schema" not in st.session_state or st.session_state.get("last_engine_id") != engine_id:
        st.session_state.schema = get_schema_from_engine(engine)
        st.session_state.last_engine_id = engine_id

    schema = st.session_state.schema
    api_key = GROQ_API_KEY  # from program

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("dataframe") is not None:
                st.dataframe(msg["dataframe"], use_container_width=True)
            if msg.get("sql"):
                with st.expander("Generated SQL"):
                    st.code(msg["sql"], language="sql")

    # Pending write confirmation
    if st.session_state.pending_sql:
        st.warning("⚠️ This query will **modify data in the real database**. Changes will be permanent. Please confirm.")
        st.code(st.session_state.pending_sql, language="sql")

        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Confirm & Execute", type="primary"):
                result = execute_write_sql(engine, st.session_state.pending_sql)
                # works with (success, message) or (success, message, rows)
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
                st.session_state.schema = get_schema_from_engine(engine)
                st.rerun()
        with col2:
            if st.button("❌ Cancel"):
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Query cancelled by user. No changes made."}
                )
                st.session_state.pending_sql = None
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

        with st.spinner("Generating SQL..."):
            try:
                llm = get_llm(api_key, model)
                sql = generate_sql(llm, english_question, schema, history)
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.stop()

        if is_write_query(sql):
            st.session_state.pending_sql = sql
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "This is a **data-modifying** query. It will change the real database. Please confirm below.",
                    "sql": sql,
                }
            )
            st.rerun()
        else:
            df, error = execute_sql(engine, sql)
            if error:
                reply = f"❌ SQL Error: {error}"
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply, "sql": sql}
                )
            else:
                if df is not None and not df.empty:
                    reply = f"Found **{len(df)}** row(s)."
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "dataframe": df,
                            "sql": sql,
                        }
                    )
                else:
                    reply = "No matching records found."
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply, "sql": sql}
                    )
            st.rerun()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
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


if __name__ == "__main__":
    main()