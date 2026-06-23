import sqlite3

import numpy as np

from config import CATEGORIES
from src.data.data_versioning import scan_npy, scan_stroke_db


def test_scan_npy(tmp_path):
    cat = CATEGORIES[0]
    arr = np.zeros((5, 784), dtype="uint8")
    np.save(tmp_path / f"{cat}.npy", arr)
    out = scan_npy(tmp_path, do_hash=False)
    assert out["classes"][cat]["present"] is True
    assert out["classes"][cat]["samples"] == 5
    assert out["total_samples"] == 5
    assert out["num_classes_present"] == 1


def test_scan_npy_with_hash(tmp_path):
    cat = CATEGORIES[0]
    np.save(tmp_path / f"{cat}.npy", np.zeros((2, 784), dtype="uint8"))
    out = scan_npy(tmp_path, do_hash=True)
    assert "sha1" in out["classes"][cat]


def test_scan_stroke_db(tmp_path):
    db = tmp_path / "test.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE stroke_samples (id INTEGER PRIMARY KEY, target TEXT, "
            "predicted TEXT, confidence REAL, correct INTEGER, mode TEXT, "
            "strokes_json TEXT, created_at TEXT)"
        )
        conn.executemany(
            "INSERT INTO stroke_samples (target, predicted, confidence, correct, mode, strokes_json, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                ("cat", "cat", 0.9, 1, "mouse", "[]", "2026-01-01"),
                ("cat", "dog", 0.4, 0, "mouse", "[]", "2026-01-02"),
                ("dog", "dog", 0.8, 1, "hand", "[]", "2026-01-03"),
            ],
        )
    out = scan_stroke_db(db)
    assert out["present"] is True
    assert out["total_stroke_samples"] == 3
    assert out["per_target"]["cat"] == 2


def test_scan_stroke_db_missing(tmp_path):
    out = scan_stroke_db(tmp_path / "nope.sqlite3")
    assert out["present"] is False
