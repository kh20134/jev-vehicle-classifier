"""연재 1회의 비전 자리표시자.

YOLO 같은 검출기는 아직 없습니다. 가중치도 받지 않습니다.
이 모듈은 샘플 JSON을 ``VehicleFeatures``로 읽기만 합니다.
연재 2에서 실제 검출기를 붙일 때는 같은 ``extract()``로
``VehicleFeatures``를 만들면 ``classifier``는 수정하지 않습니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Protocol

from jev_vehicle.features import VehicleFeatures

# CLI 별칭. 왼쪽 이름을 치면 오른쪽 샘플 파일을 읽습니다.
# 좌석/입석 버스와 이륜차 별칭은 옛 이름을 새 의미로 연결합니다.
# 이륜차 샘플은 12종이 아니라 TRA_미분류 데모입니다.
SAMPLE_ALIASES: dict[str, str] = {
    "bus_seated": "bus",
    "bus_city": "bus",
    "bus_standing": "bus",
    "coach": "bus",
    "city_bus": "bus",
    "truck25": "truck_2_5t",
    "cargo_b": "truck_2_5t",
    "cargo_a": "cargo_a",
    "rigid3": "truck3axle",
    "semi_5": "semi5",
    "semitrailer5": "semi5",
    "full_4": "full4",
    "fulltrailer4": "full4",
    "bike": "motorcycle",
    "motorbike": "motorcycle",
}


class FeatureExtractor(Protocol):
    """검출기 계약. 연재 2의 YOLO 추출기도 이 모양을 따릅니다."""

    name: str

    def extract(self, source: str | Path | Mapping[str, object]) -> VehicleFeatures:
        """이미지 경로나 특징 JSON을 차량 특징으로 바꿉니다."""


class MockVisionExtractor:
    """화소 대신 특징 JSON을 읽는 추출기.

    ``source``가 파일이면 JSON을 읽고, 딕셔너리면 그대로 검증합니다.
    이미지를 열지 않습니다.
    """

    name = "mock"

    def extract(self, source: str | Path | Mapping[str, object]) -> VehicleFeatures:
        if isinstance(source, Mapping):
            return VehicleFeatures.from_dict(source)
        path = Path(source)
        return load_features(path)


def samples_dir() -> Path:
    """패키지에 들어 있는 샘플 디렉터리."""

    packaged = Path(__file__).resolve().parent / "data" / "samples"
    if packaged.is_dir() and any(packaged.glob("*.json")):
        return packaged
    repo = Path(__file__).resolve().parents[2] / "examples" / "sample_features"
    if repo.is_dir():
        return repo
    raise FileNotFoundError(
        "샘플 특징 JSON을 찾지 못했습니다. examples/sample_features를 확인하세요."
    )


def list_sample_names() -> list[str]:
    """확장자를 뺀 샘플 이름. 정렬되어 있습니다."""

    return sorted(path.stem for path in samples_dir().glob("*.json"))


def resolve_sample_name(name: str) -> str:
    """별칭을 실제 파일 이름으로 바꿉니다."""

    key = name.strip()
    return SAMPLE_ALIASES.get(key, key)


def sample_path(name: str) -> Path:
    canonical = resolve_sample_name(name)
    path = samples_dir() / f"{canonical}.json"
    if not path.is_file():
        known = ", ".join(list_sample_names())
        aliases = ", ".join(sorted(SAMPLE_ALIASES))
        raise FileNotFoundError(
            f"샘플 {name!r}이 없습니다. 샘플: {known}. 별칭: {aliases}."
        )
    return path


def load_sample(name: str) -> VehicleFeatures:
    """내장 샘플 이름으로 특징을 읽습니다."""

    return load_features(sample_path(name))


def load_features(path: str | Path) -> VehicleFeatures:
    """특징 JSON 파일을 읽습니다. 영상 파일은 받지 않습니다."""

    file_path = Path(path)
    if file_path.suffix.lower() != ".json":
        raise ValueError(
            "연재 1회 추출기는 JSON 특징만 읽습니다. "
            f"이미지 파일은 이후 회차의 검출기가 처리합니다: {file_path.name}"
        )
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"특징 JSON을 해석하지 못했습니다: {file_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"특징 JSON의 최상위는 객체여야 합니다: {file_path}")
    return VehicleFeatures.from_dict(payload)
