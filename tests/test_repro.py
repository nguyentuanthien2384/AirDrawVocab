import random

from src.utils.repro import set_global_seed, collect_environment


def test_set_global_seed_returns_seed():
    assert set_global_seed(123) == 123


def test_set_global_seed_reproducible():
    set_global_seed(7)
    a = [random.random() for _ in range(5)]
    set_global_seed(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_collect_environment_keys():
    env = collect_environment()
    assert "python_version" in env
    assert "platform" in env
    assert "gpu_count" in env
