"""특징 JSON과 Jev state."""

import json
from pathlib import Path

import pytest

from jev_vehicle.features import VehicleFeatures
from jev_vehicle.vision_mock import list_sample_names, load_sample, samples_dir

ROOT = Path(__file__).resolve().parents[1]


def test_round_trip_and_state_has_no_image():
    features = load_sample("sedan")
    restored = VehicleFeatures.from_dict(features.to_dict())
    assert restored == features

    state = features.to_state()
    assert state["schema"] == "jev-vehicle.features.v1"
    assert state["pipeline"]["image_included"] is False
    assert state["features"]["axle_count"] == 2
    assert state["features"]["length_m"] == pytest.approx(4.65)
    assert state["features"]["body_type"] == "sedan"
    blob = json.dumps(state)
    assert "image_bytes" not in blob
    assert "base64" not in blob
    # 정답 코드를 state에 싣지 않습니다. Jev가 고르게 둡니다.
    assert "class_01" not in blob


def test_samples_match_package_and_examples():
    packaged = samples_dir()
    examples = ROOT / "examples" / "sample_features"
    names = sorted(path.name for path in packaged.glob("*.json"))
    assert names == sorted(path.name for path in examples.glob("*.json"))
    assert len(names) >= 12
    for name in names:
        assert (packaged / name).read_text(encoding="utf-8") == (
            examples / name
        ).read_text(encoding="utf-8")
        payload = json.loads((examples / name).read_text(encoding="utf-8"))
        blob = json.dumps(payload)
        # 정답 코드는 샘플에 넣지 않습니다. 필드 이름 gross_weight_class_hint와는 구분합니다.
        assert not any(f"class_{number:02d}" in blob for number in range(1, 13))


def test_every_sample_loads():
    assert "sedan" in list_sample_names()
    assert "truck3axle" in list_sample_names()
    assert "bus" in list_sample_names()
    for name in list_sample_names():
        features = load_sample(name)
        assert features.source_id
        assert features.to_state()["features"]["source_id"] == features.source_id


def test_rejects_unknown_field_and_bad_axle():
    with pytest.raises(ValueError, match="알 수 없는 특징 필드"):
        VehicleFeatures.from_dict({"source_id": "x", "plate": "secret"})
    with pytest.raises(ValueError, match="axle_count"):
        VehicleFeatures.from_dict({"source_id": "x", "axle_count": 0})
    with pytest.raises(ValueError, match="JSON"):
        from jev_vehicle.vision_mock import load_features

        load_features(ROOT / "README.md")
