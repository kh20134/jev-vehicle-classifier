"""mock Choice. 네트워크와 API 키를 쓰지 않습니다."""

import pytest

from jev_vehicle.classes import CLASS_CODES, UNCLASSIFIED_COLUMN
from jev_vehicle.classifier import (
    DEFAULT_REVIEW_CONFIDENCE,
    ClassificationError,
    choice_confidence,
    classify,
)
from jev_vehicle.features import VehicleFeatures
from jev_vehicle.vision_mock import load_sample

# 단서가 분명한 샘플 → 공식 12종.
CLEAR_SAMPLES = {
    "sedan": "class_01",
    "bus": "class_02",
    "cargo_a": "class_03",
    "truck_2_5t": "class_04",
    "truck3axle": "class_05",
    "truck4axle": "class_06",
    "truck5axle": "class_07",
    "semi4": "class_08",
    "full4": "class_09",
    "semi5": "class_10",
    "full5": "class_11",
    "semi6": "class_12",
}


def _base(**overrides):
    payload = {
        "source_id": "unit",
        "extractor": "mock",
        "axle_count": 2,
        "unit_count": 1,
        "trailer_type": "none",
    }
    payload.update(overrides)
    return VehicleFeatures.from_dict(payload)


@pytest.mark.parametrize("sample,code", CLEAR_SAMPLES.items())
def test_clear_samples(sample, code):
    result = classify(load_sample(sample), mock=True)
    assert result.class_code == code
    assert result.name_ko
    assert result.tra_column == f"TRA_{int(code[-2:]):02d}"
    assert result.counted_as == result.tra_column
    assert result.mode == "mock"
    assert result.model is None
    assert result.needs_review is False
    assert result.out_of_scope is False
    assert set(result.probabilities) == set(CLASS_CODES)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.confidence == pytest.approx(choice_confidence(result.probabilities))
    assert result.confidence >= DEFAULT_REVIEW_CONFIDENCE
    assert max(result.probabilities, key=result.probabilities.get) == code


def test_combination_is_one_vehicle():
    semi = classify(load_sample("semi5"), mock=True)
    full = classify(load_sample("full4"), mock=True)
    assert semi.class_code == "class_10"
    assert semi.name_ko == "대형화물 C"
    assert semi.counted_as == "TRA_10"
    assert full.class_code == "class_09"
    assert full.name_ko == "대형화물 B"
    assert full.counted_as == "TRA_09"


def test_tonnage_boundary_on_two_axle_one_unit():
    under = classify(
        _base(body_type="cargo", load_tonnage_t=2.49, source_id="under"),
        mock=True,
    )
    at = classify(
        _base(body_type="cargo", load_tonnage_t=2.5, source_id="at"),
        mock=True,
    )
    mini = classify(
        _base(body_type="mini_truck", load_tonnage_t=0.8, source_id="mini"),
        mock=True,
    )
    assert under.class_code == "class_03"
    assert at.class_code == "class_04"
    assert mini.class_code == "class_01"
    assert mini.name_ko == "승용차·미니트럭"


def test_motorcycle_is_unclassified_even_when_the_choice_is_confident():
    features = _base(
        body_type="motorcycle",
        wheel_count=2,
        seat_count=2,
        length_m=2.1,
        source_id="moto",
    )
    result = classify(features, mock=True, threshold=0.0)
    assert result.class_code in CLASS_CODES
    assert result.confidence >= DEFAULT_REVIEW_CONFIDENCE
    assert result.out_of_scope is True
    assert result.needs_review is True
    assert result.counted_as == UNCLASSIFIED_COLUMN
    assert "이륜차" not in result.name_ko


def test_sample_motorcycle_and_ambiguous_go_to_unclassified():
    motorcycle = classify(load_sample("motorcycle"), mock=True)
    ambiguous = classify(load_sample("ambiguous"), mock=True)
    assert motorcycle.out_of_scope is True
    assert motorcycle.counted_as == UNCLASSIFIED_COLUMN
    assert ambiguous.out_of_scope is False
    assert ambiguous.needs_review is True
    assert ambiguous.counted_as == UNCLASSIFIED_COLUMN
    assert "TRA_미분류" not in ambiguous.probabilities


def test_review_threshold_is_application_policy():
    features = load_sample("sedan")
    automatic = classify(features, mock=True, threshold=0.0)
    review = classify(features, mock=True, threshold=0.99)
    assert automatic.class_code == review.class_code == "class_01"
    assert automatic.counted_as == "TRA_01"
    assert review.counted_as == UNCLASSIFIED_COLUMN
    assert review.needs_review is True


def test_missing_key_stays_on_mock(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    result = classify(load_sample("truck3axle"))
    assert result.mode == "mock"
    assert result.class_code == "class_05"
    assert result.name_ko == "중형화물 A"


def test_live_without_key_does_not_call_the_network(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ClassificationError, match="TYPESAFE_API_KEY"):
        classify(load_sample("sedan"), live=True)


def test_six_axle_full_trailer_is_outside_the_table():
    features = _base(
        axle_count=6,
        unit_count=2,
        trailer_type="full",
        body_type="full",
        source_id="full6",
    )
    result = classify(features, mock=True, threshold=0.0)
    assert result.out_of_scope is True
    assert result.counted_as == UNCLASSIFIED_COLUMN


def test_answers_map_is_what_the_mock_returns():
    from jev_vehicle.classifier import MockJevClient

    features = load_sample("bus")
    response = MockJevClient().system_one(
        state=features.to_state(),
        questions={"vehicle_class": {"instructions": "x", "criteria": {}}},
    )
    assert response.answers["vehicle_class"].choice == "class_02"
