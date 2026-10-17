"""도로교통량조사 12종 차종분류.

기준은 한국건설기술연구원 교통량정보제공시스템의 12종 차종분류표입니다.
https://www.road.re.kr/intro/intro_04.asp

축수와 단위(1단위 단일 차체, 2단위 견인차+피견인차를 한 대로 센 것)로
나눕니다. 버스는 16인승 이상이면 한 종이고, 이륜차 칸은 없습니다.
트랙터와 트레일러를 두 차종으로 쪼개지 않습니다.

코드 ``class_01`` … ``class_12``는 한국도로공사 AVC 원시자료의
``TRA_01종`` … ``TRA_12종``과 같은 순서입니다.
https://www.data.go.kr/data/15045525/fileData.do

``TRA_미분류``는 Choice 선택지가 아닙니다. 12종 밖이거나 confidence가
낮으면 애플리케이션이 그 열로 보냅니다.
"""

from __future__ import annotations

from dataclasses import dataclass

# AVC 원시자료에서 12종 밖에 해당하는 열 이름.
UNCLASSIFIED_COLUMN = "TRA_미분류"


@dataclass(frozen=True)
class VehicleClass:
    """분류표의 한 행.

    ``code``가 Choice 선택지입니다. ``tra_column``은 ``TRA_01``처럼
    짧은 이름이고, ``tra_column_ko``는 원시자료 CSV의 열 이름입니다.
    """

    code: str
    number: int
    name_ko: str
    name_en: str
    tra_column: str
    axles: str
    units: int
    what: str
    signals: str
    not_for: str

    @property
    def tra_column_ko(self) -> str:
        return f"{self.tra_column}종"

    def criteria(self) -> dict[str, str]:
        """Jev Choice criteria. 한글 이름이 설명 안에 있습니다."""

        return {
            "name_ko": self.name_ko,
            "tra_column": self.tra_column_ko,
            "axles": self.axles,
            "units": str(self.units),
            "what": f"{self.name_ko}. {self.what}",
            "signals": self.signals,
            "not_for": self.not_for,
        }


# 적재 톤 경계는 분류표 문장 그대로입니다.
# 3종은 1톤 이상 2.5톤 미만, 4종은 2.5톤 이상.
VEHICLE_CLASSES: tuple[VehicleClass, ...] = (
    VehicleClass(
        code="class_01",
        number=1,
        name_ko="승용차·미니트럭",
        name_en="passenger car or mini truck",
        tra_column="TRA_01",
        axles="2",
        units=1,
        what=(
            "One rigid unit with 2 axles: a passenger vehicle under 16 seats, "
            "or a mini truck whose maximum load is under 1 tonne."
        ),
        signals=(
            "unit_count 1; axle_count 2; trailer_type none; "
            "body sedan, suv, hatchback, passenger_van, or mini_truck; "
            "seat_count under 16; load_tonnage_t null or under 1."
        ),
        not_for="Buses of 16 or more seats (class_02) and cargo trucks of 1 tonne or more (class_03, class_04).",
    ),
    VehicleClass(
        code="class_02",
        number=2,
        name_ko="버스",
        name_en="bus, 16 or more seats",
        tra_column="TRA_02",
        axles="2",
        units=1,
        what=(
            "A bus with 16 or more seats, 2 axles, one rigid unit. "
            "City, express, seated, and charter buses are the same class."
        ),
        signals="unit_count 1; axle_count 2; body bus; seat_count 16 or more, or body bus when seats are unobserved.",
        not_for="Passenger vans under 16 seats (class_01). Do not split standing and seated buses.",
    ),
    VehicleClass(
        code="class_03",
        number=3,
        name_ko="소형화물 A",
        name_en="small cargo A, 2 axles, 1 tonne to under 2.5 tonnes",
        tra_column="TRA_03",
        axles="2",
        units=1,
        what=(
            "A cargo truck, 2 axles, one unit, maximum load from 1 tonne "
            "up to but not including 2.5 tonnes."
        ),
        signals="unit_count 1; axle_count 2; trailer_type none; load_tonnage_t >= 1 and < 2.5; cargo body.",
        not_for="Mini trucks under 1 tonne (class_01) and 2-axle trucks of 2.5 tonnes or more (class_04).",
    ),
    VehicleClass(
        code="class_04",
        number=4,
        name_ko="소형화물 B",
        name_en="small cargo B, 2 axles, 2.5 tonnes or more",
        tra_column="TRA_04",
        axles="2",
        units=1,
        what="A cargo truck, 2 axles, one unit, maximum load of 2.5 tonnes or more. Still one rigid body.",
        signals="unit_count 1; axle_count 2; trailer_type none; load_tonnage_t >= 2.5.",
        not_for="Trucks under 2.5 tonnes (class_03) and any 2-unit tractor-trailer.",
    ),
    VehicleClass(
        code="class_05",
        number=5,
        name_ko="중형화물 A",
        name_en="medium cargo A, 3 axles, 1 unit",
        tra_column="TRA_05",
        axles="3",
        units=1,
        what="A cargo truck with 3 axles and one rigid unit.",
        signals="unit_count 1; axle_count 3; trailer_type none.",
        not_for="2-unit combinations and 4-axle rigid trucks (class_06).",
    ),
    VehicleClass(
        code="class_06",
        number=6,
        name_ko="중형화물 B",
        name_en="medium cargo B, 4 axles, 1 unit",
        tra_column="TRA_06",
        axles="4",
        units=1,
        what="A cargo truck with 4 axles and one rigid unit.",
        signals="unit_count 1; axle_count 4; trailer_type none.",
        not_for="4-axle 2-unit semi or full trailers (class_08, class_09).",
    ),
    VehicleClass(
        code="class_07",
        number=7,
        name_ko="중형화물 C",
        name_en="medium cargo C, 5 axles, 1 unit",
        tra_column="TRA_07",
        axles="5",
        units=1,
        what="A cargo truck with 5 axles and one rigid unit.",
        signals="unit_count 1; axle_count 5; trailer_type none.",
        not_for="5-axle 2-unit semi or full trailers (class_10, class_11).",
    ),
    VehicleClass(
        code="class_08",
        number=8,
        name_ko="대형화물 A",
        name_en="heavy cargo A, semi-trailer, 4 axles, 2 units",
        tra_column="TRA_08",
        axles="4",
        units=2,
        what=(
            "A semi-trailer combination with 4 axles in total and 2 units. "
            "Count the tractor and the semi-trailer as one vehicle."
        ),
        signals="unit_count 2; axle_count 4; trailer_type semi.",
        not_for="Full trailers (class_09) and 5-axle semi-trailers (class_10).",
    ),
    VehicleClass(
        code="class_09",
        number=9,
        name_ko="대형화물 B",
        name_en="heavy cargo B, full trailer, 4 axles, 2 units",
        tra_column="TRA_09",
        axles="4",
        units=2,
        what=(
            "A full-trailer combination with 4 axles in total and 2 units. "
            "Count the towing vehicle and the full trailer as one vehicle."
        ),
        signals="unit_count 2; axle_count 4; trailer_type full.",
        not_for="Semi-trailers (class_08) and 5-axle full trailers (class_11).",
    ),
    VehicleClass(
        code="class_10",
        number=10,
        name_ko="대형화물 C",
        name_en="heavy cargo C, semi-trailer, 5 axles, 2 units",
        tra_column="TRA_10",
        axles="5",
        units=2,
        what="A semi-trailer combination with 5 axles in total and 2 units, counted as one vehicle.",
        signals="unit_count 2; axle_count 5; trailer_type semi.",
        not_for="4-axle semis (class_08) and 6-or-more-axle semis (class_12).",
    ),
    VehicleClass(
        code="class_11",
        number=11,
        name_ko="대형화물 D",
        name_en="heavy cargo D, full trailer, 5 axles, 2 units",
        tra_column="TRA_11",
        axles="5",
        units=2,
        what="A full-trailer combination with 5 axles in total and 2 units, counted as one vehicle.",
        signals="unit_count 2; axle_count 5; trailer_type full.",
        not_for="4-axle full trailers (class_09) and semi-trailers.",
    ),
    VehicleClass(
        code="class_12",
        number=12,
        name_ko="대형화물 E",
        name_en="heavy cargo E, semi-trailer, 6 or more axles, 2 units",
        tra_column="TRA_12",
        axles="6+",
        units=2,
        what="A semi-trailer combination with 6 or more axles in total and 2 units, counted as one vehicle.",
        signals="unit_count 2; axle_count 6 or more; trailer_type semi.",
        not_for="5-axle semis (class_10). A full trailer with 6 or more axles is not this class.",
    ),
)


CLASS_BY_CODE: dict[str, VehicleClass] = {item.code: item for item in VEHICLE_CLASSES}
CLASS_CODES: tuple[str, ...] = tuple(item.code for item in VEHICLE_CLASSES)


def get_class(code: str) -> VehicleClass:
    """코드로 차종을 찾습니다. 없는 코드는 KeyError."""

    try:
        return CLASS_BY_CODE[code]
    except KeyError as exc:
        known = ", ".join(CLASS_CODES)
        raise KeyError(f"알 수 없는 차종 코드 {code!r}. 정의된 코드: {known}") from exc
