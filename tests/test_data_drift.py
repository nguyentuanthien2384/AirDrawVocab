from src.evaluation.data_drift import (
    population_stability_index, interpret,
)


def test_identical_distributions():
    a = {"cat": 10, "dog": 10, "car": 10}
    psi, per_key = population_stability_index(a, a)
    assert abs(psi) < 1e-6
    assert interpret(psi) == "ổn định"


def test_drift_detected():
    expected = {"cat": 100, "dog": 100, "car": 100}
    actual = {"cat": 280, "dog": 10, "car": 10}
    psi, per_key = population_stability_index(expected, actual)
    assert psi > 0.25
    assert interpret(psi) == "TRÔI MẠNH"


def test_new_class_handled():
    expected = {"cat": 50}
    actual = {"cat": 25, "dog": 25}
    psi, per_key = population_stability_index(expected, actual)
    assert psi > 0
    assert "dog" in per_key


def test_interpret_thresholds():
    assert interpret(0.05) == "ổn định"
    assert interpret(0.15) == "thay đổi vừa"
    assert interpret(0.5) == "TRÔI MẠNH"
