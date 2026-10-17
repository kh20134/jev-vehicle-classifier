"""mock Choice. 네트워크와 API 키를 쓰지 않습니다."""

import pytest

from jev_vehicle.classes import CLASS_CODES
from jev_vehicle.classifier import (
    DEFAULT_REVIEW_CONFIDENCE,
    ClassificationError,
    choice_confidence,
    classify,
)
from jev_vehicle.features import VehicleFeatures
from jev_vehicle.vision_mock import load_sample

# 단서가 분명한 샘플 → 데모가 고르는 코드.
CLEAR_SAMPLES = {
    "sedan": "class_01",
    "bus": "class_02",
    "bus_city": "class_03",
    "truck_small": "class_04",
    "truck_medium": "class_05",
    "truck2axle": "class_06",
    "truck3axle": "class_07",
    "truck4axle": "class_08",
    "truck5axle": "class_09",
    "tractor": "class_10",
    "trailer": "class_11",
    "motorcycle": "class_12",
}


@pytest.mark.parametrize("sample,code", CLEAR_SAMPLES.items())
def test_clear_samples(sample, code):
    result = classify(load_sample(sample), mock=True)
    assert result.class_code == code
    assert result.name_ko
    assert result.mode == "mock"
    assert result.model is None
    assert result.needs_review is False
    assert set(result.probabilities) == set(CLASS_CODES)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.confidence == pytest.approx(choice_confidence(result.probabilities))
    assert result.confidence >= DEFAULT_REVIEW_CONFIDENCE
    assert max(result.probabilities, key=result.probabilities.get) == code


def test_tractor_is_not_three_axle_cargo_and_trailer_is_separate():
    tractor = classify(load_sample("tractor"), mock=True)
    trailer = classify(load_sample("trailer"), mock=True)
    assert tractor.class_code == "class_10"
    assert tractor.name_ko == "특수(견인)"
    assert trailer.class_code == "class_11"
    assert trailer.name_ko == "특수(피견인)"


def test_ambiguous_observation_needs_review_without_a_thirteenth_class():
    result = classify(load_sample("ambiguous"), mock=True)
    assert result.class_code in CLASS_CODES
    assert result.needs_review is True
    assert result.confidence < DEFAULT_REVIEW_CONFIDENCE
    assert "needs_review" not in result.probabilities


def test_review_threshold_is_application_policy():
    features = load_sample("sedan")
    automatic = classify(features, mock=True, threshold=0.0)
    review = classify(features, mock=True, threshold=0.99)
    assert automatic.class_code == review.class_code == "class_01"
    assert automatic.needs_review is False
    assert review.needs_review is True


def test_missing_key_stays_on_mock(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    result = classify(load_sample("truck3axle"))
    assert result.mode == "mock"
    assert result.class_code == "class_07"
    assert result.name_ko == "대형화물(3축)"


def test_live_without_key_does_not_call_the_network(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ClassificationError, match="TYPESAFE_API_KEY"):
        classify(load_sample("sedan"), live=True)


def test_six_axle_rigid_truck_is_leaned_to_class_09_and_reviewed():
    features = VehicleFeatures.from_dict(
        {
            "source_id": "rigid-6axle",
            "extractor": "mock",
            "axle_count": 6,
            "length_m": 13.0,
            "body_type": "box_truck",
            "is_towing": False,
            "is_towed": False,
        }
    )
    result = classify(features, mock=True)
    assert result.class_code == "class_09"
    assert result.needs_review is True


def test_alias_bus_seated():
    assert classify(load_sample("bus_seated"), mock=True).class_code == "class_02"
