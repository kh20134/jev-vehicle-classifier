"""12종 코드와 Choice criteria."""

from jev_vehicle.classes import CLASS_CODES, VEHICLE_CLASSES, get_class
from jev_vehicle.classifier import vehicle_class_question


EXPECTED_NAMES = [
    "승용차",
    "버스(좌석)",
    "버스(입석)",
    "소형화물(1톤 이하)",
    "중형화물(1~2.5톤)",
    "대형화물(2.5톤 초과, 2축)",
    "대형화물(3축)",
    "대형화물(4축)",
    "대형화물(5축)",
    "특수(견인)",
    "특수(피견인)",
    "이륜차",
]


def test_twelve_codes_and_korean_names():
    assert [item.number for item in VEHICLE_CLASSES] == list(range(1, 13))
    assert CLASS_CODES == tuple(f"class_{number:02d}" for number in range(1, 13))
    assert [item.name_ko for item in VEHICLE_CLASSES] == EXPECTED_NAMES


def test_choice_criteria_use_codes_and_korean_names():
    question = vehicle_class_question()
    assert set(question["criteria"]) == set(CLASS_CODES)
    for item in VEHICLE_CLASSES:
        criteria = question["criteria"][item.code]
        assert criteria["name_ko"] == item.name_ko
        assert item.name_ko in criteria["what"]
    assert "needs_review" not in question["criteria"]
    assert "미분류" not in question["criteria"]


def test_unknown_code_is_rejected():
    try:
        get_class("class_13")
    except KeyError as exc:
        assert "class_13" in str(exc)
    else:
        raise AssertionError("class_13은 차종이 아닙니다.")
