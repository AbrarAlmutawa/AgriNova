# -*- coding: utf-8 -*-
"""
plant_store.py — Plant registry + prediction history + GIS location

Adds three things on top of the existing Day 5 pipeline, without touching
the CNN or RAG logic in day5_api.py:

  1. Each physical plant gets a short unique plant_id (registered once,
     e.g. when a sensor/mic is installed in the field).
  2. Every /predict call is logged against that plant_id, building a
     history of predictions, confidence, and recommendations over time.
  3. Each plant carries a latitude/longitude, so Streamlit can render
     all plants on a map (st.map) — a lightweight GIS layer.

SQLite is used deliberately instead of Postgres/PostGIS: this is a
hackathon demo, not a production GIS system, and SQLite needs zero setup.

USAGE (inside day5_api.py):
    from plant_store import PlantStore
    store = PlantStore()
    ...
    store.log_reading(plant_id, prediction, confidence, recommendation,
                       technical_sources, policy_sources)
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime, timezone

DB_PATH = os.environ.get("PLANT_DB_PATH", "./agrinova_plants.db")
PHOTO_DIR = os.environ.get("PLANT_PHOTO_DIR", "./plant_photos")


class PlantStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()
        os.makedirs(PHOTO_DIR, exist_ok=True)

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS plants (
                plant_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                field_zone TEXT,
                created_at TEXT NOT NULL
            )
        """)
        # Migration: add photo_filename to existing DBs that predate this column.
        existing_cols = [row[1] for row in self._conn.execute("PRAGMA table_info(plants)")]
        if "photo_filename" not in existing_cols:
            self._conn.execute("ALTER TABLE plants ADD COLUMN photo_filename TEXT")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                reading_id TEXT PRIMARY KEY,
                plant_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                recommendation TEXT,
                technical_sources TEXT,
                policy_sources TEXT,
                llm_called INTEGER,
                FOREIGN KEY (plant_id) REFERENCES plants(plant_id)
            )
        """)
        self._conn.commit()

    # ---------------- Plants ----------------

    def create_plant(self, name: str, latitude: float, longitude: float,
                      field_zone: str = None) -> str:
        plant_id = uuid.uuid4().hex[:8]
        self._conn.execute(
            "INSERT INTO plants (plant_id, name, latitude, longitude, field_zone, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (plant_id, name, latitude, longitude, field_zone,
             datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        return plant_id

    def get_plant(self, plant_id: str):
        row = self._conn.execute(
            "SELECT plant_id, name, latitude, longitude, field_zone, created_at, photo_filename "
            "FROM plants WHERE plant_id = ?", (plant_id,)
        ).fetchone()
        if not row:
            return None
        keys = ["plant_id", "name", "latitude", "longitude", "field_zone", "created_at", "photo_filename"]
        return dict(zip(keys, row))

    def list_plants(self):
        rows = self._conn.execute(
            "SELECT plant_id, name, latitude, longitude, field_zone, created_at, photo_filename "
            "FROM plants ORDER BY created_at DESC"
        ).fetchall()
        keys = ["plant_id", "name", "latitude", "longitude", "field_zone", "created_at", "photo_filename"]
        return [dict(zip(keys, r)) for r in rows]

    def plant_exists(self, plant_id: str) -> bool:
        return self.get_plant(plant_id) is not None

    def set_plant_photo(self, plant_id: str, photo_bytes: bytes, original_filename: str) -> str:
        """Save a photo to disk for this plant and record its filename.
        Overwrites any previous photo for the same plant."""
        ext = os.path.splitext(original_filename)[1].lower() or ".jpg"
        stored_filename = f"{plant_id}{ext}"
        with open(os.path.join(PHOTO_DIR, stored_filename), "wb") as f:
            f.write(photo_bytes)
        self._conn.execute(
            "UPDATE plants SET photo_filename = ? WHERE plant_id = ?",
            (stored_filename, plant_id),
        )
        self._conn.commit()
        return stored_filename

    def get_photo_path(self, plant_id: str):
        plant = self.get_plant(plant_id)
        if not plant or not plant["photo_filename"]:
            return None
        path = os.path.join(PHOTO_DIR, plant["photo_filename"])
        return path if os.path.exists(path) else None

    # ---------------- Readings ----------------

    def log_reading(self, plant_id: str, prediction: str, confidence: float,
                     recommendation: str, technical_sources: list,
                     policy_sources: list, llm_called: bool) -> str:
        reading_id = uuid.uuid4().hex[:8]
        self._conn.execute(
            "INSERT INTO readings VALUES (?,?,?,?,?,?,?,?,?)",
            (
                reading_id, plant_id, datetime.now(timezone.utc).isoformat(),
                prediction, confidence, recommendation,
                json.dumps(technical_sources), json.dumps(policy_sources),
                int(llm_called),
            ),
        )
        self._conn.commit()
        return reading_id

    def get_history(self, plant_id: str, limit: int = 50):
        rows = self._conn.execute(
            "SELECT reading_id, timestamp, prediction, confidence, recommendation, "
            "technical_sources, policy_sources, llm_called FROM readings "
            "WHERE plant_id = ? ORDER BY timestamp DESC LIMIT ?",
            (plant_id, limit),
        ).fetchall()
        history = []
        for r in rows:
            history.append({
                "reading_id": r[0], "timestamp": r[1], "prediction": r[2],
                "confidence": r[3], "recommendation": r[4],
                "technical_sources": json.loads(r[5]) if r[5] else [],
                "policy_sources": json.loads(r[6]) if r[6] else [],
                "llm_called": bool(r[7]),
            })
        return history

    def recent_predictions(self, plant_id: str, n: int = 3):
        """Just the prediction labels from the last n readings, most recent last.
        Useful for trend-aware RAG queries (e.g. 3 'Dry' readings in a row)."""
        rows = self._conn.execute(
            "SELECT prediction FROM readings WHERE plant_id = ? "
            "ORDER BY timestamp DESC LIMIT ?", (plant_id, n),
        ).fetchall()
        return [r[0] for r in reversed(rows)]