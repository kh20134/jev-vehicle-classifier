"""공식 12종 코드, TRA 열, Choice criteria."""

from jev_vehicle.classes import CLASS_CODES, UNCLASSIFIED_COLUMN, VEHICLE_CLASSES, get_class
from jev_vehicle.classifier import vehicle_class_question


EXPECTED_NAMES = [
    "승용차·미니트럭",
    "버스",
    "소형화물 A",
    "소형화물 B",
    "중형화물 A",
    "중형화물 B",
    "중형화물 C",
    "대형화물 A",
    "대형화물 B",
    "대형화물 C",
    "대형화물 D",
    "대형화물 E",
]


def test_twelve_codes_names_and_tra_columns():
    assert [item.number for item in VEHICLE_CLASSES] == list(range(1, 13))
    assert CLASS_CODES == tuple(f"class_{number:02d}" for number in range(1, 13))
    assert [item.name_ko for item in VEHICLE_CLASSES] == EXPECTED_NAMES
    assert [item.tra_column for item in VEHICLE_CLASSES] == [
        f"TRA_{number:02d}" for number in range(1, 13)
    ]
    assert [item.tra_column_ko for item in VEHICLE_CLASSES] == [
        f"TRA_{number:02d}종" for number in range(1, 13)
    ]
    assert UNCLASSIFIED_COLUMN == "TRA_미분류"
    assert all(name not in EXPECTED_NAMES for name in ("이륜차", "버스(좌석)", "버스(입석)"))


def test_choice_criteria_follow_axles_and_units():
    question = vehicle_class_question()
    assert set(question["criteria"]) == set(CLASS_CODES)
    assert "TRA_미분류" not in question["criteria"]
    assert "이륜차" not in question["criteria"]
    by_code = {item.code: item for item in VEHICLE_CLASSES}
    assert by_code["class_01"].units == 1 and by_code["class_01"].axles == "2"
    assert by_code["class_03"].name_ko == "소형화물 A"
    assert by_code["class_08"].units == 2
    assert "semi" in by_code["class_10"].signals
    assert "full" in by_code["class_09"].signals
    assert "6" in by_code["class_12"].axles
    for item in VEHICLE_CLASSES:
        criteria = question["criteria"][item.code]
        assert criteria["name_ko"] == item.name_ko
        assert item.name_ko in criteria["what"]
        assert criteria["tra_column"] == item.tra_column_ko


def test_unknown_code_is_rejected():
    try:
        get_class("class_13")
    except KeyError as exc:
        assert "class_13" in str(exc)
    else:
        raise AssertionError("class_13은 차종이 아닙니다.")
