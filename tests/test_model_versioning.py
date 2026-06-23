import json

from src.utils.model_versioning import (
    make_version_tag, save_versioned_model, list_versions,
)


class FakeModel:
    """Giả lập Keras model: chỉ cần .save(path) ghi ra file."""
    def save(self, path):
        with open(path, "wb") as f:
            f.write(b"fake-keras-model")


def test_version_tag_format():
    tag = make_version_tag()
    assert tag.startswith("v")
    assert len(tag) == len("v20260622_141530")


def test_save_versioned_model(tmp_path):
    info = save_versioned_model(
        FakeModel(),
        base_name="unit_test_model",
        metrics={"val_accuracy": 0.91},
        params={"epochs": 3},
        versions_dir=tmp_path,
    )
    assert info  # không rỗng
    model_file = info["metadata"]["model_file"]
    stem = model_file.rsplit(".", 1)[0]
    # file model + metadata tồn tại
    assert (tmp_path / model_file).exists()
    meta = json.loads((tmp_path / f"{stem}.json").read_text(encoding="utf-8"))
    assert meta["base_name"] == "unit_test_model"
    assert meta["metrics"]["val_accuracy"] == 0.91


def test_list_versions(tmp_path):
    save_versioned_model(FakeModel(), base_name="a", metrics={"acc": 0.5}, versions_dir=tmp_path)
    save_versioned_model(FakeModel(), base_name="b", metrics={"acc": 0.7}, versions_dir=tmp_path)
    all_v = list_versions(versions_dir=tmp_path)
    assert len(all_v) == 2
    only_a = list_versions(base_name="a", versions_dir=tmp_path)
    assert len(only_a) == 1
    assert only_a[0]["base_name"] == "a"
