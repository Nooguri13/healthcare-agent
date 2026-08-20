"""SQLite persistence for extracted structured records."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from . import config
from .schema import DischargeRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS discharge_records (
    doc_id              TEXT PRIMARY KEY,
    patient_id          TEXT NOT NULL,
    age                 INTEGER,
    sex                 TEXT,
    admission_date      TEXT,
    discharge_date      TEXT,
    attending_physician TEXT,
    principal_diagnosis TEXT,
    allergies           TEXT,
    medications         TEXT
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = Path(db_path or config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def upsert(conn: sqlite3.Connection, doc_id: str, record: DischargeRecord) -> None:
    d = record.dict_compat()
    conn.execute(
        """
        INSERT INTO discharge_records
            (doc_id, patient_id, age, sex, admission_date, discharge_date,
             attending_physician, principal_diagnosis, allergies, medications)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(doc_id) DO UPDATE SET
            patient_id=excluded.patient_id, age=excluded.age, sex=excluded.sex,
            admission_date=excluded.admission_date, discharge_date=excluded.discharge_date,
            attending_physician=excluded.attending_physician,
            principal_diagnosis=excluded.principal_diagnosis,
            allergies=excluded.allergies, medications=excluded.medications
        """,
        (
            doc_id,
            d["patient_id"],
            d["age"],
            d["sex"],
            d["admission_date"],
            d["discharge_date"],
            d["attending_physician"],
            d["principal_diagnosis"],
            d["allergies"],
            json.dumps(d["medications"]),
        ),
    )
    conn.commit()


def query(conn: sqlite3.Connection, sql: str, params: Iterable = ()) -> list[dict]:
    cur = conn.execute(sql, tuple(params))
    return [dict(r) for r in cur.fetchall()]
