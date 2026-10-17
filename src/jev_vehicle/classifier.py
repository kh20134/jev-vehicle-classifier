"""TypeSafe Jev Choice로 12종 중 하나를 고릅니다.

라이브 호출은 문서의 시그니처를 그대로 씁니다.

    with TypeSafeClient(model="jev-latest") as client:
        response = client.system_one(
            state=features.to_state(),
            questions={"vehicle_class": Choice(instructions=..., criteria=...)},
        )
        answer = response.choices["vehicle_class"]
        answer.choice, answer.probabilities, answer.confidence

이미지는 넘기지 않습니다. state는 ``VehicleFeatures.to_state()`` JSON입니다.

``needs_review``는 13번째 Choice 선택지가 아닙니다. 지침 차종은 12개이고,
확신이 낮을 때 사람 검토로 보내는 일은 애플리케이션 정책입니다.
Choice에 '미분류'를 넣으면 확률이 차종이 아닌 쪽으로 새고,
집계 코드도 지침과 어긋납니다. Jev는 항상 12개 중 하나를 고르게 두고,
``confidence``가 임계값보다 낮으면 ``needs_review``만 켭니다.

연재 1회의 임계값 0.45는 자리표시자입니다. 12지선다에서 confidence
0.45는 대략 최상위 확률 0.50에 해당합니다. 현장 보정은 이후 회차입니다.

API 키가 없거나 ``mock=True``이면 ``MockJevClient``가 같은 state를
읽고 결정적인 데모 분포를 돌려줍니다. 그 규칙은 지침도 아니고 Jev도
아닙니다. 오프라인 데모와 테스트용입니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from jev_vehicle.classes import CLASS_CODES, VEHICLE_CLASSES, VehicleClass, get_class
from jev_vehicle.features import VehicleFeatures

# 문서의 모델 별칭. 고정 버전으로 잠그지 않습니다.
JEV_MODEL = "jev-latest"
QUESTION_ID = "vehicle_class"

# (p_max - 1/12) / (1 - 1/12) = 0.45 → p_max ≈ 0.496
# 최상위 확률이 절반 근처일 때부터 자동 집계한다는 보수적인 시작점.
DEFAULT_REVIEW_CONFIDENCE = 0.45

_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "class_01": ("class_04", "class_12"),
    "class_02": ("class_03", "class_01"),
    "class_03": ("class_02", "class_06"),
    "class_04": ("class_01", "class_05"),
    "class_05": ("class_04", "class_06"),
    "class_06": ("class_05", "class_07"),
    "class_07": ("class_06", "class_08", "class_10"),
    "class_08": ("class_07", "class_09"),
    "class_09": ("class_08", "class_10"),
    "class_10": ("class_11", "class_07"),
    "class_11": ("class_10", "class_09"),
    "class_12": ("class_01", "class_04"),
}

_PASSENGER_BODIES = {"sedan", "suv", "hatchback", "passenger_van"}
_SEATED_LAYOUTS = {"seated", "seated_only"}
_STANDING_LAYOUTS = {"standing", "standing_allowed"}


class ClassificationError(Exception):
    """호출 결과를 12종 보고서로 만들 수 없을 때."""


class SystemOneClient(Protocol):
    """``system_one(state=..., questions=...)``만 있으면 됩니다.

    라이브 클라이언트와 ``MockJevClient``가 이 모양을 공유합니다.
    """

    def system_one(self, *, state: dict[str, Any], questions: Mapping[str, Any]) -> Any:
        """한 state에 대한 질문 응답."""


@dataclass(frozen=True)
class ClassificationResult:
    """차종 하나와 분포, 그리고 사람 검토가 필요한지."""

    class_code: str
    class_number: int
    name_ko: str
    probabilities: dict[str, float]
    confidence: float
    needs_review: bool
    threshold: float
    mode: str
    model: str | None
    source_id: str
    extractor: str
    summary_ko: str

    @property
    def vehicle_class(self) -> VehicleClass:
        return get_class(self.class_code)

    def to_dict(self) -> dict[str, Any]:
        ordered = {
            code: self.probabilities[code]
            for code in CLASS_CODES
            if code in self.probabilities
        }
        for code, probability in self.probabilities.items():
            ordered.setdefault(code, probability)
        return {
            "class_code": self.class_code,
            "class_number": self.class_number,
            "name_ko": self.name_ko,
            "probabilities": ordered,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "threshold": self.threshold,
            "mode": self.mode,
            "model": self.model,
            "source_id": self.source_id,
            "extractor": self.extractor,
            "summary_ko": self.summary_ko,
            "routing": "needs_review" if self.needs_review else "auto",
        }


def choice_confidence(probabilities: Mapping[str, float]) -> float:
    """Choice confidence. ``(p_max - 1/n) / (1 - 1/n)``.

    TypeSafe가 Choice 응답에 실어 주는 값과 같은 식입니다.
    데모 클라이언트가 분포에서 confidence를 만들 때 씁니다.
    """

    values = [float(value) for value in probabilities.values()]
    count = len(values)
    if count <= 1:
        return 1.0
    peak = max(values)
    return (peak - 1.0 / count) / (1.0 - 1.0 / count)


def vehicle_class_question() -> dict[str, Any]:
    """Choice로 감싸기 전의 질문 명세.

    라이브 경로에서만 ``Choice(instructions=..., criteria=...)``로
    바꿉니다. 선택지 키는 ``class_01`` … ``class_12``입니다.
    """

    return {
        "instructions": (
            "Choose the single vehicle class from Korea's road traffic-volume "
            "survey guideline (도로교통량 조사지침). Use only the structured "
            "observation in the state: axle count, length, body type, weight-class "
            "hint, seating layout, and whether the vehicle is towing or towed. "
            "The state is text and JSON, not a photograph. "
            "Pick the one criteria entry that best matches. "
            "Cargo classes split on the weight hint and axle count. "
            "Buses split on seating layout: seated versus standing. "
            "A tractor that tows is class_10 (특수(견인)); "
            "the towed trailer is class_11 (특수(피견인)). "
            "Do not invent a class outside the criteria."
        ),
        "criteria": {item.code: item.criteria() for item in VEHICLE_CLASSES},
    }


def api_key_present() -> bool:
    """공백만 있는 키는 없는 것으로 봅니다."""

    return bool(os.environ.get("TYPESAFE_API_KEY", "").strip())


def classify(
    features: VehicleFeatures,
    *,
    mock: bool = False,
    live: bool = False,
    threshold: float = DEFAULT_REVIEW_CONFIDENCE,
) -> ClassificationResult:
    """특징을 12종으로 분류합니다.

    ``mock``이면 데모 클라이언트입니다. ``live``이면 Jev입니다.
    둘 다 아니면, 키가 있을 때만 Jev를 호출합니다.
    """

    _check_threshold(threshold)
    if mock and live:
        raise ClassificationError("mock과 live를 동시에 지정할 수 없습니다.")
    if live or (not mock and api_key_present()):
        if not api_key_present():
            raise ClassificationError(
                "TYPESAFE_API_KEY가 없습니다. "
                "키 없이 데모를 보려면 mock=True 또는 CLI --mock을 쓰세요."
            )
        return classify_with(
            LiveJevClient(),
            features,
            mode="live",
            threshold=threshold,
            model=JEV_MODEL,
        )
    return classify_with(
        MockJevClient(),
        features,
        mode="mock",
        threshold=threshold,
        model=None,
    )


def classify_with(
    client: SystemOneClient,
    features: VehicleFeatures,
    *,
    mode: str,
    threshold: float = DEFAULT_REVIEW_CONFIDENCE,
    model: str | None = None,
) -> ClassificationResult:
    """이미 만든 클라이언트로 분류합니다. 테스트가 여기를 씁니다."""

    _check_threshold(threshold)
    question = vehicle_class_question()
    response = client.system_one(
        state=features.to_state(),
        questions={QUESTION_ID: question},
    )
    return result_from_response(
        response,
        features,
        mode=mode,
        threshold=threshold,
        model=model,
    )


class LiveJevClient:
    """``typesafe-sdk``로 Jev를 호출합니다. 키는 환경 변수에서 읽습니다."""

    def __init__(self, model: str = JEV_MODEL) -> None:
        self.model = model

    def system_one(self, *, state: dict[str, Any], questions: Mapping[str, Any]) -> Any:
        try:
            from typesafe_sdk import Choice, TypeSafeClient
        except ImportError as exc:
            raise ClassificationError(
                "typesafe-sdk가 없습니다. pip install -e \".[live]\" 로 설치하세요."
            ) from exc

        wrapped: dict[str, Any] = {}
        for key, question in questions.items():
            if isinstance(question, dict) and "criteria" in question:
                wrapped[key] = Choice(
                    instructions=question.get("instructions"),
                    criteria=question["criteria"],
                )
            else:
                wrapped[key] = question

        # 키가 없으면 SDK가 클라이언트 생성 단계에서 실패합니다.
        with TypeSafeClient(model=self.model) as client:
            return client.system_one(state=state, questions=wrapped)


class MockJevClient:
    """키 없이 쓰는 결정적 데모.

    state의 ``features``만 보고 12칸 확률을 만듭니다.
    confidence는 Choice와 같은 공식으로 계산합니다.
    메모 문장과 body_hints는 해석하지 않습니다. 그 문장은 라이브
    Jev만 봅니다. 단서가 빈약하면 한쪽으로 기울인 낮은 분포를 주어
    ``needs_review``가 켜지게 합니다.
    """

    def system_one(self, *, state: dict[str, Any], questions: Mapping[str, Any]) -> Any:
        if QUESTION_ID not in questions:
            raise ClassificationError(
                f"questions에 {QUESTION_ID} Choice가 없습니다."
            )
        features = _features_from_state(state)
        code, certainty = demo_decision(features)
        probabilities = demo_probabilities(code, certainty)
        confidence = choice_confidence(probabilities)
        answer = _DemoChoice(
            choice=code,
            probabilities=probabilities,
            confidence=confidence,
        )
        return _DemoResponse(choices={QUESTION_ID: answer})


@dataclass(frozen=True)
class _DemoChoice:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class _DemoResponse:
    choices: dict[str, _DemoChoice]


def demo_decision(features: VehicleFeatures) -> tuple[str, str]:
    """데모가 고를 코드와 ``high``/``low``.

    순서가 경계입니다. 견인·피견인을 축수보다 먼저 봐서,
    3축 트랙터가 3축 대형화물로 떨어지지 않게 합니다.
    6축 이상 단일 차체는 9종 칸이 없어서 9종으로 기울이고 low로 둡니다.
    """

    body = (features.body_type or "").lower()
    layout = (features.seating_layout or "").lower()
    weight = (features.gross_weight_class_hint or "").lower()
    axles = features.axle_count

    if features.is_towing is True and features.is_towed is True:
        return "class_10", "low"

    if body == "motorcycle" or features.wheel_count == 2:
        return "class_12", "high"
    if features.is_towed is True or body in {"trailer", "semitrailer"}:
        return "class_11", "high"
    if features.is_towing is True or body == "tractor":
        return "class_10", "high"
    if axles is not None and axles >= 6:
        return "class_09", "low"
    if axles == 5:
        return "class_09", "high"
    if axles == 4:
        return "class_08", "high"
    if axles == 3:
        return "class_07", "high"
    if body == "coach" or layout in _SEATED_LAYOUTS:
        return "class_02", "high"
    if body == "city_bus" or layout in _STANDING_LAYOUTS:
        return "class_03", "high"
    if body == "bus":
        return "class_02", "low"
    if weight == "under_1t" or body == "cargo_small":
        return "class_04", "high"
    if weight == "1_to_2_5t" or body == "cargo_medium":
        return "class_05", "high"
    if body in {"box_truck", "dump", "cargo_large"} and axles == 2:
        return "class_06", "high"
    if weight == "over_2_5t" and axles == 2:
        return "class_06", "high"
    if body in _PASSENGER_BODIES or layout == "passenger":
        return "class_01", "high"

    # 축도 차체도 없으면 길이로만 기울이고, 검토로 보냅니다.
    return _weak_guess(features.length_m), "low"


def demo_probabilities(code: str, certainty: str) -> dict[str, float]:
    """12칸이 모두 있고 합이 1인 분포."""

    if code not in _NEIGHBORS:
        raise ClassificationError(f"데모가 정의 밖 코드를 만들었습니다: {code}")
    neighbors = _NEIGHBORS[code]
    # 12칸 합이 정확히 1이 되도록 잡은 소수입니다. 이웃이 3개인 차종만 칸 수가 다릅니다.
    if certainty == "high" and len(neighbors) == 2:
        peak, neighbor_mass, share = 0.85, 0.03, 0.01
    elif certainty == "high":
        peak, neighbor_mass, share = 0.82, 0.04, 0.0075
    elif certainty == "low" and len(neighbors) == 2:
        peak, neighbor_mass, share = 0.31, 0.12, 0.05
    elif certainty == "low":
        peak, neighbor_mass, share = 0.26, 0.14, 0.04
    else:
        raise ClassificationError(f"알 수 없는 데모 확신: {certainty}")

    probabilities = {item: share for item in CLASS_CODES}
    probabilities[code] = peak
    for neighbor in neighbors:
        probabilities[neighbor] = neighbor_mass
    return {item: probabilities[item] for item in CLASS_CODES}


def result_from_response(
    response: Any,
    features: VehicleFeatures,
    *,
    mode: str,
    threshold: float,
    model: str | None,
) -> ClassificationResult:
    choice, probabilities, confidence = read_choice_answer(response)
    try:
        vehicle = get_class(choice)
    except KeyError as exc:
        raise ClassificationError(str(exc)) from exc
    # 임계값 '미만'만 검토. 같은 값은 자동으로 둡니다.
    needs_review = confidence < threshold
    return ClassificationResult(
        class_code=vehicle.code,
        class_number=vehicle.number,
        name_ko=vehicle.name_ko,
        probabilities=probabilities,
        confidence=confidence,
        needs_review=needs_review,
        threshold=threshold,
        mode=mode,
        model=model,
        source_id=features.source_id,
        extractor=features.extractor,
        summary_ko=features.summary_ko(),
    )


def read_choice_answer(response: Any) -> tuple[str, dict[str, float], float]:
    """SDK 응답의 ``choices``를 읽고, 없으면 ``answers``를 봅니다."""

    answer = None
    choices = getattr(response, "choices", None)
    if isinstance(choices, Mapping) and QUESTION_ID in choices:
        answer = choices[QUESTION_ID]
    if answer is None:
        answers = getattr(response, "answers", None)
        if isinstance(answers, Mapping):
            answer = answers.get(QUESTION_ID)
    if answer is None:
        raise ClassificationError("응답에 vehicle_class Choice가 없습니다.")

    raw_probabilities = getattr(answer, "probabilities", None)
    if not isinstance(raw_probabilities, Mapping):
        raise ClassificationError("Choice 응답에 probabilities가 없습니다.")
    probabilities = {str(key): float(value) for key, value in raw_probabilities.items()}
    try:
        choice = str(answer.choice)
        confidence = float(answer.confidence)
    except AttributeError as exc:
        raise ClassificationError("Choice 응답에 choice 또는 confidence가 없습니다.") from exc
    return choice, probabilities, confidence


def _features_from_state(state: Mapping[str, Any]) -> VehicleFeatures:
    if not isinstance(state, Mapping):
        raise ClassificationError("state는 JSON 객체여야 합니다.")
    features = state.get("features")
    if isinstance(features, Mapping):
        return VehicleFeatures.from_dict(features)
    return VehicleFeatures.from_dict(state)


def _weak_guess(length_m: float | None) -> str:
    if length_m is None:
        return "class_01"
    if length_m < 2.5:
        return "class_12"
    if length_m < 5.2:
        return "class_01"
    if length_m < 6.5:
        return "class_05"
    if length_m < 8.5:
        return "class_06"
    if length_m < 11.0:
        return "class_02"
    return "class_07"


def _check_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ClassificationError("검토 임계값은 0 이상 1 이하의 수여야 합니다.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ClassificationError("검토 임계값은 0 이상 1 이하의 수여야 합니다.")
