"""도로교통량 조사지침 12종 차종.

코드 ``class_01`` … ``class_12``는 연재 내내 바꾸지 않습니다.
한글 명칭은 Choice criteria에 그대로 들어가고, 영어 설명은 Jev가
이웃 차종을 가르도록 붙인 판단 기준입니다.

Jev의 주 학습 언어는 영어입니다. 한글 이름만 두고 설명을 비우면
축수·톤수 경계가 흐려지므로, 이름은 한글로 고정하고 경계는 영어로
적습니다. 차종 집합 자체는 아래 12개이며 검토 라우팅은 차종이 아닙니다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleClass:
    """지침 차종 하나.

    ``code``가 Choice 선택지 이름입니다. 애플리케이션은 이 코드로
    집계하고, 사람한테는 ``name_ko``를 보여 줍니다.
    """

    code: str
    number: int
    name_ko: str
    name_en: str
    what: str
    signals: str
    not_for: str

    def criteria(self) -> dict[str, str]:
        """Jev Choice criteria 값. 한글 이름이 설명 안에 있습니다."""

        return {
            "name_ko": self.name_ko,
            "what": f"{self.name_ko}. {self.what}",
            "signals": self.signals,
            "not_for": self.not_for,
        }


# 경계는 연재 1회를 읽기 위한 작업용 설명입니다.
# 1톤 이하는 4종, 1톤을 넘고 2.5톤까지는 5종, 2.5톤을 넘으면서 2축이면 6종.
# 현장 예외와 임계값 보정은 이후 회차의 데이터로 맞춥니다.
VEHICLE_CLASSES: tuple[VehicleClass, ...] = (
    VehicleClass(
        code="class_01",
        number=1,
        name_ko="승용차",
        name_en="passenger car",
        what="A passenger car that carries people: sedan, hatchback, SUV, or a small passenger van. Two axles and an enclosed cabin, without a cargo bed.",
        signals="body sedan, suv, hatchback, or passenger_van; seating_layout passenger; about 3.5–5.2 m long; 2 axles; 4 wheels.",
        not_for="Buses, cargo trucks, tractors, trailers, and motorcycles.",
    ),
    VehicleClass(
        code="class_02",
        number=2,
        name_ko="버스(좌석)",
        name_en="seated bus",
        what="A bus whose passengers sit. Intercity, express, or chartered coach, not a city bus that is built for standing.",
        signals="body coach, or seating_layout seated or seated_only; long body often about 9–12 m; usually 2 axles; baggage bays.",
        not_for="City buses that allow standing (버스(입석), class_03) and passenger cars (class_01).",
    ),
    VehicleClass(
        code="class_03",
        number=3,
        name_ko="버스(입석)",
        name_en="standing bus",
        what="A city or local bus designed so passengers may stand, with a wide aisle and more than one door.",
        signals="body city_bus, or seating_layout standing or standing_allowed; usually 2 axles.",
        not_for="Seated-only coaches (버스(좌석), class_02) and cargo trucks.",
    ),
    VehicleClass(
        code="class_04",
        number=4,
        name_ko="소형화물(1톤 이하)",
        name_en="small cargo truck, at or under 1 tonne",
        what="A small cargo truck at or under about 1 tonne payload. Short cab-over with an open or covered bed.",
        signals="gross_weight_class_hint under_1t; body cargo_small; 2 axles; length often under about 5.5 m; a cargo bed.",
        not_for="Passenger cars with no cargo bed (class_01) and medium cargo above 1 tonne (class_05).",
    ),
    VehicleClass(
        code="class_05",
        number=5,
        name_ko="중형화물(1~2.5톤)",
        name_en="medium cargo truck, over 1 tonne up to 2.5 tonnes",
        what="A medium cargo truck above 1 tonne and up to about 2.5 tonnes. Usually still 2 axles, with a longer bed than a 1-ton truck.",
        signals="gross_weight_class_hint 1_to_2_5t; body cargo_medium; 2 axles.",
        not_for="Trucks at or under 1 tonne (class_04) and heavy 2-axle trucks over 2.5 tonnes (class_06).",
    ),
    VehicleClass(
        code="class_06",
        number=6,
        name_ko="대형화물(2.5톤 초과, 2축)",
        name_en="heavy cargo truck over 2.5 tonnes with 2 axles",
        what="A rigid heavy cargo truck over 2.5 tonnes that has exactly 2 axles. Not a tractor-trailer.",
        signals="axle_count 2; gross_weight_class_hint over_2_5t; body box_truck, dump, or cargo_large; is_towing false.",
        not_for="3-or-more-axle rigid trucks (class_07 and up) and tractor units (class_10).",
    ),
    VehicleClass(
        code="class_07",
        number=7,
        name_ko="대형화물(3축)",
        name_en="rigid heavy truck, 3 axles",
        what="A rigid heavy cargo truck with 3 axles, often a tandem rear axle. Not the tractor of a semi-trailer.",
        signals="axle_count 3; rigid cargo body; is_towing false; is_towed false.",
        not_for="Tractors (특수(견인), class_10), towed trailers (class_11), and 2-axle heavies (class_06).",
    ),
    VehicleClass(
        code="class_08",
        number=8,
        name_ko="대형화물(4축)",
        name_en="rigid heavy truck, 4 axles",
        what="A rigid heavy cargo truck with 4 axles.",
        signals="axle_count 4; rigid body; not towing and not a towed trailer.",
        not_for="3-axle and 5-axle rigid trucks, tractors, and trailers.",
    ),
    VehicleClass(
        code="class_09",
        number=9,
        name_ko="대형화물(5축)",
        name_en="rigid heavy truck, 5 axles",
        what="A rigid heavy cargo truck with 5 axles. This set has no separate class above 5 axles.",
        signals="axle_count 5; rigid cargo body; not a tractor and not a trailer.",
        not_for="4-axle rigid trucks (class_08), tractors (class_10), and towed trailers (class_11).",
    ),
    VehicleClass(
        code="class_10",
        number=10,
        name_ko="특수(견인)",
        name_en="tractor unit",
        what="The powered tractor that tows a semi-trailer or full trailer. Classify the tractor itself, not the vehicle it pulls.",
        signals="body tractor; is_towing true; often 2 or 3 axles on the unit; a fifth wheel rather than a long cargo body.",
        not_for="The towed trailer (특수(피견인), class_11) and rigid cargo trucks that are not towing.",
    ),
    VehicleClass(
        code="class_11",
        number=11,
        name_ko="특수(피견인)",
        name_en="towed trailer",
        what="A towed trailer or semi-trailer with no drive of its own.",
        signals="body trailer or semitrailer; is_towed true; the axles belong to the trailer.",
        not_for="The tractor that pulls it (특수(견인), class_10).",
    ),
    VehicleClass(
        code="class_12",
        number=12,
        name_ko="이륜차",
        name_en="motorcycle",
        what="A motorcycle or other two-wheeler.",
        signals="body motorcycle; wheel_count 2; short length, often under about 2.5 m.",
        not_for="Passenger cars and trucks.",
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
