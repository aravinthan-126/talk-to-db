# 💬 Talk-to-DB

**Conversational Natural Language Interface to Database**  
Academic Mini Project

---

## 🔐 Login

| Username | Password    |
|----------|-------------|
| `admin`  | `admin@123` |

---

## 📂 Database Connection Options

After login you can connect in any of these ways:

| Option | Details |
|--------|---------|
| **SQLite File** | Upload `.db` / `.sqlite` file **or** give full local path |
| **MySQL / MariaDB** | Host, Port, Username, Password, Database Name |
| **PostgreSQL** | Host, Port, Username, Password, Database Name |
| **Sample Database** | Ready-made Employees DB (if you cancel / skip) |

---

## ✅ Important: Immediate Reflection of Changes

All `INSERT` / `UPDATE` / `DELETE` operations are executed with an **immediate commit** on the real database.

→ Any change you confirm in the app is written directly to the connected database and is visible immediately (in MySQL Workbench, pgAdmin, DB Browser for SQLite, etc.).

---

## 🎯 Features

- Login authentication  
- Multiple connection methods (SQLite file + MySQL + PostgreSQL)  
- Falls back to Sample DB if user cancels  
- Dynamic schema extraction  
- English variations  
- Mixed Tamil + English queries  
- Context-aware conversations  
- Safe SQL (confirmation required before any write)

---

## 📦 Installation

```bash
cd talk_to_db
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## ▶️ Usage Flow

1. Login → `admin` / `admin@123`
2. Choose connection method (SQLite / MySQL / PostgreSQL / Sample)
3. Chat with the database
4. For any write query the app will ask confirmation → after confirm the change is saved in the real DB immediately

---

## 📁 Project Structure

```
talk_to_db/
├── app.py
├── database.py
├── requirements.txt
├── .env
├── README.md
└── employees.db          # Sample SQLite DB
```

---

**Made for Academic Mini Project**
