from src.utils.model_versioning import save_versioned_model
from src.utils import model_registry as mr


class FakeModel:
    def save(self, path):
        with open(path, "wb") as f:
            f.write(b"fake")


def _seed_versions(tmp_path):
    save_versioned_model(FakeModel(), base_name="alpha", metrics={"val_accuracy": 0.80}, versions_dir=tmp_path)
    save_versioned_model(FakeModel(), base_name="beta", metrics={"val_accuracy": 0.92}, versions_dir=tmp_path)
    save_versioned_model(FakeModel(), base_name="gamma", metrics={"val_accuracy": 0.55}, versions_dir=tmp_path)


def test_build_registry(tmp_path):
    _seed_versions(tmp_path)
    reg = mr.build_registry(versions_dir=tmp_path)
    assert reg["count"] == 3
    assert len(reg["models"]) == 3


def test_save_registry(tmp_path):
    _seed_versions(tmp_path)
    reg = mr.build_registry(versions_dir=tmp_path)
    out = mr.save_registry(reg, path=tmp_path / "registry.json")
    assert out.exists()


def test_find_best(tmp_path):
    _seed_versions(tmp_path)
    best = mr.find_best("val_accuracy", versions_dir=tmp_path)
    assert best is not None
    assert best["base_name"] == "beta"
    assert best["metrics"]["val_accuracy"] == 0.92


def test_find_best_missing_metric(tmp_path):
    _seed_versions(tmp_path)
    assert mr.find_best("nonexistent_metric", versions_dir=tmp_path) is None


def test_promote(tmp_path):
    info = save_versioned_model(FakeModel(), base_name="alpha", metrics={}, versions_dir=tmp_path)
    src = tmp_path / info["metadata"]["model_file"]
    deploy = tmp_path / "deploy" / "best.keras"
    out = mr.promote(src, deploy_path=deploy)
    assert out.exists()
    assert out.read_bytes() == b"fake"
