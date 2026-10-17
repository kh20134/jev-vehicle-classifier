"""비전 계층과 Jev 계층 사이의 계약.

검출기(연재 1회는 mock, 이후에는 YOLO 등)는 카메라 화소가 아니라
``VehicleFeatures``만 넘깁니다. ``to_state()``가 그 특징을 Jev가
읽는 JSON 객체로 바꿉니다. Jev는 결정 모델이라 이미지 CNN이 아니고,
현재 입력은 텍스트·JSON·텍스트 배열입니다.

같은 dataclass를 유지하면 검출기를 바꿔도 Choice 질문과 차종 코드는
그대로 둡니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# 연재 2 검출기가 맞춰 줄 권장 어휘입니다. 검증은 막지 않습니다.
# 새 라벨이 생겨도 state 스키마를 깨지 않게 문자열로 통과시키고,
# 의미를 모르면 body_type을 "unknown"으로 둡니다.
RECOMMENDED_BODY_TYPES: tuple[str, ...] = (
    "sedan",
    "suv",
    "hatchback",
    "passenger_van",
    "coach",
    "city_bus",
    "bus",
    "cargo_small",
    "cargo_medium",
    "box_truck",
    "dump",
    "cargo_large",
    "tractor",
    "trailer",
    "semitrailer",
    "motorcycle",
    "unknown",
)

# under_1t → 4종, 1_to_2_5t → 5종, over_2_5t → 6종 이상의 톤수 힌트.
# 저울값이 아니라 검출·차로 센서가 준 거친 구간입니다.
RECOMMENDED_WEIGHT_HINTS: tuple[str, ...] = (
    "under_1t",
    "1_to_2_5t",
    "over_2_5t",
    "unknown",
)

_FIELDS = (
    "source_id",
    "extractor",
    "axle_count",
    "length_m",
    "width_m",
    "height_m",
    "wheel_count",
    "body_type",
    "body_hints",
    "gross_weight_class_hint",
    "seating_layout",
    "has_trailer",
    "is_towing",
    "is_towed",
    "lane",
    "speed_kmh",
    "notes",
)

_MAX_NOTES = 500
_MAX_HINTS = 12


@dataclass(frozen=True)
class VehicleFeatures:
    """차량 한 대에 대한 구조화 관측.

    번호판, 얼굴, 영상 바이트는 필드가 아닙니다. 넣지 마세요.
    결측은 None입니다. None은 '안 재 봄'이 아니라 '값이 없다'는
    관측으로 state에 남깁니다.
    """

    source_id: str
    extractor: str = "manual"
    axle_count: int | None = None
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    wheel_count: int | None = None
    body_type: str | None = None
    body_hints: tuple[str, ...] = ()
    gross_weight_class_hint: str | None = None
    seating_layout: str | None = None
    has_trailer: bool | None = None
    is_towing: bool | None = None
    is_towed: bool | None = None
    lane: str | None = None
    speed_kmh: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 저장할 수 있는 특징 객체."""

        return {
            "source_id": self.source_id,
            "extractor": self.extractor,
            "axle_count": self.axle_count,
            "length_m": self.length_m,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "wheel_count": self.wheel_count,
            "body_type": self.body_type,
            "body_hints": list(self.body_hints),
            "gross_weight_class_hint": self.gross_weight_class_hint,
            "seating_layout": self.seating_layout,
            "has_trailer": self.has_trailer,
            "is_towing": self.is_towing,
            "is_towed": self.is_towed,
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
        if self.length_m is not None:
            bits.append(f"길이 {self.length_m:.2f} m")
        if self.height_m is not None:
            bits.append(f"높이 {self.height_m:.2f} m")
        if self.wheel_count is not None:
            bits.append(f"바퀴 {self.wheel_count}")
        if self.gross_weight_class_hint:
            bits.append(f"중량 힌트 {self.gross_weight_class_hint}")
        if self.seating_layout:
            bits.append(f"좌석 {self.seating_layout}")
        if self.is_towing is True:
            bits.append("견인")
        if self.is_towed is True:
            bits.append("피견인")
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
        if self.length_m is not None:
            bits.append(f"length {self.length_m:.2f} m")
        if self.height_m is not None:
            bits.append(f"height {self.height_m:.2f} m")
        if self.wheel_count is not None:
            bits.append(f"{self.wheel_count} wheels")
        if self.gross_weight_class_hint:
            bits.append(f"weight hint {self.gross_weight_class_hint}")
        if self.seating_layout:
            bits.append(f"seating {self.seating_layout}")
        if self.is_towing is True:
            bits.append("towing")
        if self.is_towed is True:
            bits.append("towed")
        if self.body_hints:
            bits.append("hints " + ",".join(self.body_hints))
        if self.notes:
            bits.append(f"notes: {self.notes}")
        if not bits:
            return "Almost no structured evidence."
        return "; ".join(bits)

    def to_state(self) -> dict[str, Any]:
        """Jev ``state`` 인자로 넘길 JSON 객체.

        이미지는 넣지 않습니다. 질문 문장도 넣지 않습니다.
        질문은 classifier가 Choice로 따로 보냅니다.
        """

        return {
            "schema": "jev-vehicle.features.v1",
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
                "There is no camera image in this state."
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
        if not isinstance(hints, list) or any(not isinstance(item, str) or not item.strip() for item in hints):
            raise ValueError("body_hints는 비어 있지 않은 문자열의 배열이어야 합니다.")
        if len(hints) > _MAX_HINTS:
            raise ValueError(f"body_hints는 {_MAX_HINTS}개를 넘길 수 없습니다.")

        return cls(
            source_id=source_id.strip(),
            extractor=extractor.strip(),
            axle_count=_optional_int(data, "axle_count", minimum=1),
            length_m=_optional_float(data, "length_m", minimum=0.0, positive=True),
            width_m=_optional_float(data, "width_m", minimum=0.0, positive=True),
            height_m=_optional_float(data, "height_m", minimum=0.0, positive=True),
            wheel_count=_optional_int(data, "wheel_count", minimum=1),
            body_type=_optional_str(data, "body_type"),
            body_hints=tuple(item.strip() for item in hints),
            gross_weight_class_hint=_optional_str(data, "gross_weight_class_hint"),
            seating_layout=_optional_str(data, "seating_layout"),
            has_trailer=_optional_bool(data, "has_trailer"),
            is_towing=_optional_bool(data, "is_towing"),
            is_towed=_optional_bool(data, "is_towed"),
            lane=_optional_str(data, "lane"),
            speed_kmh=_optional_float(data, "speed_kmh", minimum=0.0, positive=False),
            notes=_optional_notes(data),
        )


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key}는 비어 있지 않은 문자열 또는 null이어야 합니다.")
    return value.strip()


def _optional_bool(data: Mapping[str, Any], key: str) -> bool | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key}는 true, false, null 중 하나여야 합니다.")
    return value


def _optional_int(data: Mapping[str, Any], key: str, *, minimum: int) -> int | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    # bool은 int의 하위 타입이라 먼저 걸러야 합니다.
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
