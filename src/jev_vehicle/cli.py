"""명령줄에서 샘플이나 특징 JSON을 12종으로 분류합니다.

    python -m jev_vehicle classify --sample sedan --mock
    python -m jev_vehicle classify --features examples/sample_features/truck3axle.json --mock
    python -m jev_vehicle classify --sample bus --live
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence, TextIO

from jev_vehicle.classes import VEHICLE_CLASSES
from jev_vehicle.classifier import (
    DEFAULT_REVIEW_CONFIDENCE,
    ClassificationError,
    ClassificationResult,
    classify,
)
from jev_vehicle.features import VehicleFeatures
from jev_vehicle.vision_mock import (
    SAMPLE_ALIASES,
    MockVisionExtractor,
    list_sample_names,
    load_sample,
)

_EXTRACTOR = MockVisionExtractor()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-vehicle",
        description=(
            "도로교통량조사 12종 차종분류를 TypeSafe Jev Choice로 고릅니다. "
            "연재 1회는 이미지 대신 특징 JSON을 읽습니다."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    classify_parser = sub.add_parser(
        "classify",
        help="내장 샘플 또는 특징 JSON을 12종으로 분류",
    )
    source = classify_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--sample",
        help="내장 샘플 이름. 예: sedan, bus, truck_2_5t, truck3axle, semi5, full4",
    )
    source.add_argument(
        "--features",
        help="특징 JSON 파일 경로",
    )
    mode = classify_parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock",
        action="store_true",
        help="API 키 없이 결정적 데모 응답을 사용",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="TypeSafe Jev 호출. 환경 변수 TYPESAFE_API_KEY가 필요합니다.",
    )
    classify_parser.add_argument(
        "--json",
        action="store_true",
        help="사람용 문장 대신 JSON으로 출력",
    )
    classify_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_REVIEW_CONFIDENCE,
        help=(
            "이 confidence 미만이면 needs_review. "
            f"기본값 {DEFAULT_REVIEW_CONFIDENCE} (연재 1회 자리표시자)"
        ),
    )

    sub.add_parser("classes", help="12종 코드와 한글명을 출력")
    sub.add_parser("samples", help="내장 샘플 이름과 별칭을 출력")
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    """종료 코드를 반환합니다. 테스트가 stdout을 바꿔 끼울 수 있습니다."""

    out = stdout or sys.stdout
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "classes":
            write_classes(out)
            return 0
        if args.command == "samples":
            write_samples(out)
            return 0
        if args.command == "classify":
            return run_classify(args, out)
    except (ClassificationError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    parser.error(f"알 수 없는 명령: {args.command}")
    return 2


def console_main() -> None:
    """설치 스크립트 진입점."""

    raise SystemExit(main())


def run_classify(args: argparse.Namespace, out: TextIO) -> int:
    features = _load(args)
    result = classify(
        features,
        mock=args.mock,
        live=args.live,
        threshold=args.threshold,
    )
    if args.json:
        json.dump(result.to_dict(), out, ensure_ascii=False, indent=2)
        out.write("\n")
    else:
        out.write(format_text_report(result))
    return 0


def format_text_report(result: ClassificationResult) -> str:
    """차종 코드, 한글명, 확률, confidence를 텍스트로."""

    mode_line = "mock (Jev를 호출하지 않음)" if result.mode == "mock" else "live"
    model_line = result.model or "(없음)"
    lines = [
        f"모드: {mode_line}",
        f"모델: {model_line}",
        f"추출기: {result.extractor}",
        f"source_id: {result.source_id}",
        f"요약: {result.summary_ko}",
        "",
        f"차종 코드: {result.class_code}",
        f"한글명: {result.name_ko}",
        f"종 번호: {result.class_number}",
        f"AVC 열: {result.tra_column} ({result.tra_column_ko})",
        f"집계: {result.counted_as}",
        f"confidence: {result.confidence:.4f}",
        f"needs_review: {_bool_word(result.needs_review)}",
        f"out_of_scope: {_bool_word(result.out_of_scope)}",
        f"검토 임계값: {result.threshold:.2f}",
        "",
        "확률 분포:",
    ]
    ranked = sorted(
        result.probabilities.items(),
        key=lambda item: (-item[1], item[0]),
    )
    labels = {item.code: item.name_ko for item in VEHICLE_CLASSES}
    for code, probability in ranked:
        name = labels.get(code, "")
        lines.append(f"  {probability:.4f}  {code}  {name}")
    if result.needs_review:
        lines.extend(
            [
                "",
                f"집계는 {result.counted_as}입니다. 미분류는 13번째 차종이 아닙니다.",
            ]
        )
        if result.review_reason:
            lines.append(result.review_reason)
    lines.append("")
    return "\n".join(lines)


def write_classes(out: TextIO) -> None:
    out.write("코드        번호  AVC 열     한글명\n")
    for item in VEHICLE_CLASSES:
        out.write(f"{item.code}  {item.number:>2}    {item.tra_column_ko:<8}  {item.name_ko}\n")


def write_samples(out: TextIO) -> None:
    out.write("샘플\n")
    for name in list_sample_names():
        out.write(f"  {name}\n")
    out.write("별칭\n")
    for alias, target in sorted(SAMPLE_ALIASES.items()):
        out.write(f"  {alias} -> {target}\n")


def _load(args: argparse.Namespace) -> VehicleFeatures:
    if args.sample:
        return load_sample(args.sample)
    # --features도 mock 추출기를 통과시켜, 이후 YOLO extract()와 같은 자리에 둡니다.
    return _EXTRACTOR.extract(args.features)


def _bool_word(value: bool) -> str:
    return "true" if value else "false"


if __name__ == "__main__":
    raise SystemExit(main())
