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
    assert state["schema"] == "jev-vehicle.features.v2"
    assert state["pipeline"]["image_included"] is False
    assert state["features"]["axle_count"] == 2
    assert state["features"]["unit_count"] == 1
    assert state["features"]["trailer_type"] == "none"
    assert state["features"]["load_tonnage_t"] is None
    blob = json.dumps(state)
    assert "image_bytes" not in blob
    assert "class_01" not in blob


def test_combination_sample_is_one_state():
    features = load_sample("semi5")
    assert features.unit_count == 2
    assert features.trailer_type == "semi"
    assert features.axle_count == 5
    state = features.to_state()
    assert state["features"]["unit_count"] == 2
    assert "one tractor-plus-trailer" in state["note"]


def test_samples_match_package_and_examples():
    packaged = samples_dir()
    examples = ROOT / "examples" / "sample_features"
    names = sorted(path.name for path in packaged.glob("*.json"))
    assert names == sorted(path.name for path in examples.glob("*.json"))
    assert "bus_city.json" not in names
    assert "tractor.json" not in names
    assert {"sedan.json", "bus.json", "truck_2_5t.json", "truck3axle.json", "semi5.json", "full4.json"} <= set(names)
    for name in names:
        assert (packaged / name).read_text(encoding="utf-8") == (
            examples / name
        ).read_text(encoding="utf-8")
        payload = json.loads((examples / name).read_text(encoding="utf-8"))
        blob = json.dumps(payload)
        assert not any(f"class_{number:02d}" in blob for number in range(1, 13))


def test_every_sample_loads():
    assert "sedan" in list_sample_names()
    assert "truck3axle" in list_sample_names()
    assert "bus" in list_sample_names()
    for name in list_sample_names():
        features = load_sample(name)
        assert features.source_id
        assert features.to_state()["features"]["source_id"] == features.source_id


def test_rejects_unknown_field_and_bad_values():
    with pytest.raises(ValueError, match="알 수 없는 특징 필드"):
        VehicleFeatures.from_dict({"source_id": "x", "is_towed": True})
    with pytest.raises(ValueError, match="axle_count"):
        VehicleFeatures.from_dict({"source_id": "x", "axle_count": 0})
    with pytest.raises(ValueError, match="trailer_type"):
        VehicleFeatures.from_dict({"source_id": "x", "trailer_type": "pony"})
    with pytest.raises(ValueError, match="JSON"):
        from jev_vehicle.vision_mock import load_features

        load_features(ROOT / "README.md")
