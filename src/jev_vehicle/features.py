"""비전 계층과 Jev 계층 사이의 계약.

검출기는 화소 대신 ``VehicleFeatures``만 넘깁니다.
분류표가 보는 값은 축수, 단위, 트레일러 형식(semi|full), 최대 적재량입니다.
2단위는 견인차와 피견인차를 한 대로 센 조합입니다.

``to_state()``가 그 특징을 Jev가 읽는 JSON으로 바꿉니다.
Jev는 결정 모델이고, 연재 1회 입력은 텍스트와 JSON입니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

RECOMMENDED_BODY_TYPES: tuple[str, ...] = (
    "sedan",
    "suv",
    "hatchback",
    "passenger_van",
    "bus",
    "mini_truck",
    "cargo",
    "box_truck",
    "dump",
    "semi",
    "full",
    "motorcycle",
    "unknown",
)

TRAILER_TYPES: tuple[str, ...] = ("none", "semi", "full")

_FIELDS = (
    "source_id",
    "extractor",
    "axle_count",
    "unit_count",
    "trailer_type",
    "load_tonnage_t",
    "seat_count",
    "length_m",
    "width_m",
    "height_m",
    "wheel_count",
    "body_type",
    "body_hints",
    "lane",
    "speed_kmh",
    "notes",
)

_MAX_NOTES = 500
_MAX_HINTS = 12


@dataclass(frozen=True)
class VehicleFeatures:
    """차량 한 대, 또는 2단위 조합 한 대에 대한 관측.

    번호판, 얼굴, 영상 바이트는 필드가 아닙니다.
    결측은 None입니다. None은 값이 없다는 관측으로 state에 남습니다.
    """

    source_id: str
    extractor: str = "manual"
    axle_count: int | None = None
    unit_count: int | None = None
    trailer_type: str | None = None
    load_tonnage_t: float | None = None
    seat_count: int | None = None
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    wheel_count: int | None = None
    body_type: str | None = None
    body_hints: tuple[str, ...] = ()
    lane: str | None = None
    speed_kmh: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 저장할 수 있는 특징 객체."""

        return {
            "source_id": self.source_id,
            "extractor": self.extractor,
            "axle_count": self.axle_count,
            "unit_count": self.unit_count,
            "trailer_type": self.trailer_type,
            "load_tonnage_t": self.load_tonnage_t,
            "seat_count": self.seat_count,
            "length_m": self.length_m,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "wheel_count": self.wheel_count,
            "body_type": self.body_type,
            "body_hints": list(self.body_hints),
            "lane": self.lane,
            "speed_kmh": self.speed_kmh,
            "notes": self.notes,
        }

    def summary_ko(self) -> str:
        bits: list[str] = []
        if self.body_type:
            bits.append(f"차체 {self.body_type}")
        if self.axle_count is not None:
            bits.append(f"축 {self.axle_count}")
        if self.unit_count is not None:
            bits.append(f"단위 {self.unit_count}")
        if self.trailer_type:
            bits.append(f"트레일러 {self.trailer_type}")
        if self.load_tonnage_t is not None:
            bits.append(f"적재 {self.load_tonnage_t:g}톤")
        if self.seat_count is not None:
            bits.append(f"좌석 {self.seat_count}")
        if self.length_m is not None:
            bits.append(f"길이 {self.length_m:.2f} m")
        if self.wheel_count is not None:
            bits.append(f"바퀴 {self.wheel_count}")
        if self.lane:
            bits.append(f"차로 {self.lane}")
        if not bits:
            return "구조화 단서가 거의 없습니다."
        return ", ".join(bits)

    def summary_en(self) -> str:
        bits: list[str] = []
        if self.body_type:
            bits.append(f"body {self.body_type}")
        if self.axle_count is not None:
            bits.append(f"{self.axle_count} axles")
        if self.unit_count is not None:
            bits.append(f"{self.unit_count} unit(s)")
        if self.trailer_type:
            bits.append(f"trailer {self.trailer_type}")
        if self.load_tonnage_t is not None:
            bits.append(f"max load {self.load_tonnage_t:g} t")
        if self.seat_count is not None:
            bits.append(f"{self.seat_count} seats")
        if self.length_m is not None:
            bits.append(f"length {self.length_m:.2f} m")
        if self.wheel_count is not None:
            bits.append(f"{self.wheel_count} wheels")
        if self.body_hints:
            bits.append("hints " + ",".join(self.body_hints))
        if self.notes:
            bits.append(f"notes: {self.notes}")
        if not bits:
            return "Almost no structured evidence."
        return "; ".join(bits)

    def to_state(self) -> dict[str, Any]:
        """Jev ``state`` 인자로 넘길 JSON 객체.

        이미지는 넣지 않습니다. 2단위 조합은 한 대의 state입니다.
        """

        return {
            "schema": "jev-vehicle.features.v2",
            "task": "korea_road_traffic_survey_12class",
            "pipeline": {
                "episode": 1,
                "vision": "features_already_extracted",
                "image_included": False,
            },
            "summary_ko": self.summary_ko(),
            "summary_en": self.summary_en(),
            "features": self.to_dict(),
            "note": (
                "Vision was reduced to these fields before the decision model. "
                "There is no camera image in this state. "
                "unit_count 2 means one tractor-plus-trailer vehicle, not two classes. "
                "trailer_type is none, semi, or full. load_tonnage_t is maximum cargo tonnes."
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VehicleFeatures:
        """특징 JSON 객체를 검증하고 읽습니다."""

        if not isinstance(data, Mapping):
            raise TypeError("특징 JSON의 최상위는 객체여야 합니다.")
        unknown = sorted(set(data) - set(_FIELDS))
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"알 수 없는 특징 필드: {names}")

        source_id = data.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id는 비어 있지 않은 문자열이어야 합니다.")

        extractor = data.get("extractor", "manual")
        if not isinstance(extractor, str) or not extractor.strip():
            raise ValueError("extractor는 비어 있지 않은 문자열이어야 합니다.")

        hints = data.get("body_hints", [])
        if hints is None:
            hints = []
        if not isinstance(hints, list) or any(
            not isinstance(item, str) or not item.strip() for item in hints
        ):
            raise ValueError("body_hints는 비어 있지 않은 문자열의 배열이어야 합니다.")
        if len(hints) > _MAX_HINTS:
            raise ValueError(f"body_hints는 {_MAX_HINTS}개를 넘길 수 없습니다.")

        return cls(
            source_id=source_id.strip(),
            extractor=extractor.strip(),
            axle_count=_optional_int(data, "axle_count", minimum=1),
            unit_count=_optional_int(data, "unit_count", minimum=1),
            trailer_type=_optional_trailer(data),
            load_tonnage_t=_optional_float(data, "load_tonnage_t", minimum=0.0, positive=False),
            seat_count=_optional_int(data, "seat_count", minimum=1),
            length_m=_optional_float(data, "length_m", minimum=0.0, positive=True),
            width_m=_optional_float(data, "width_m", minimum=0.0, positive=True),
            height_m=_optional_float(data, "height_m", minimum=0.0, positive=True),
            wheel_count=_optional_int(data, "wheel_count", minimum=1),
            body_type=_optional_str(data, "body_type"),
            body_hints=tuple(item.strip() for item in hints),
            lane=_optional_str(data, "lane"),
            speed_kmh=_optional_float(data, "speed_kmh", minimum=0.0, positive=False),
            notes=_optional_notes(data),
        )


def _optional_trailer(data: Mapping[str, Any]) -> str | None:
    value = _optional_str(data, "trailer_type")
    if value is None:
        return None
    lowered = value.lower()
    if lowered not in TRAILER_TYPES:
        allowed = ", ".join(TRAILER_TYPES)
        raise ValueError(f"trailer_type은 {allowed} 또는 null이어야 합니다.")
    return lowered


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key}는 비어 있지 않은 문자열 또는 null이어야 합니다.")
    return value.strip()


def _optional_int(data: Mapping[str, Any], key: str, *, minimum: int) -> int | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key}는 {minimum} 이상의 정수 또는 null이어야 합니다.")
    if value < minimum:
        raise ValueError(f"{key}는 {minimum} 이상의 정수 또는 null이어야 합니다.")
    return value


def _optional_float(
    data: Mapping[str, Any],
    key: str,
    *,
    minimum: float,
    positive: bool,
) -> float | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key}는 숫자 또는 null이어야 합니다.")
    number = float(value)
    if positive and number <= minimum:
        raise ValueError(f"{key}는 0보다 큰 수 또는 null이어야 합니다.")
    if number < minimum:
        raise ValueError(f"{key}는 {minimum} 이상의 수 또는 null이어야 합니다.")
    return number


def _optional_notes(data: Mapping[str, Any]) -> str | None:
    if "notes" not in data or data["notes"] is None:
        return None
    value = data["notes"]
    if not isinstance(value, str):
        raise ValueError("notes는 문자열 또는 null이어야 합니다.")
    if len(value) > _MAX_NOTES:
        raise ValueError(f"notes는 {_MAX_NOTES}자를 넘길 수 없습니다.")
    stripped = value.strip()
    return stripped or None
