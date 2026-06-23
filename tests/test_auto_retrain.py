import sqlite3

from src.training.auto_retrain import (
    decide_retrain, read_stroke_count, read_state, write_state,
)


def test_decide_retrain_enough():
    should, new = decide_retrain(current_count=100, last_count=50, min_new=20)
    assert should is True
    assert new == 50


def test_decide_retrain_not_enough():
    should, new = decide_retrain(current_count=60, last_count=50, min_new=20)
    assert should is False
    assert new == 10


def test_decide_retrain_zero_threshold():
    should, _ = decide_retrain(current_count=100, last_count=0, min_new=0)
    assert should is False


def test_read_stroke_count(tmp_path):
    db = tmp_path / "t.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE stroke_samples (id INTEGER PRIMARY KEY, target TEXT)")
        conn.executemany("INSERT INTO stroke_samples (target) VALUES (?)", [("a",), ("b",)])
    assert read_stroke_count(db) == 2


def test_read_stroke_count_missing(tmp_path):
    assert read_stroke_count(tmp_path / "nope.sqlite3") == 0


def test_state_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    write_state(p, {"last_count_stroke": 42})
    assert read_state(p)["last_count_stroke"] == 42


def test_read_state_missing(tmp_path):
    assert read_state(tmp_path / "nope.json") == {}
