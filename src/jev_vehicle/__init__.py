"""도로교통량 조사지침 12종을 TypeSafe Jev Choice로 분류합니다.

연재 1회는 검출기 없이 특징 JSON을 state로 만들어 Choice에 묻습니다.
"""

from __future__ import annotations

from jev_vehicle.classes import VEHICLE_CLASSES, VehicleClass
from jev_vehicle.classifier import ClassificationResult, classify
from jev_vehicle.features import VehicleFeatures

__version__ = "0.1.0"

__all__ = [
    "ClassificationResult",
    "VEHICLE_CLASSES",
    "VehicleClass",
    "VehicleFeatures",
    "classify",
    "__version__",
]
