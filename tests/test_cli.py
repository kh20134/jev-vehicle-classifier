"""CLI --mock. API 키를 지우고 실행합니다."""

import json
import os
import subprocess
import sys
from pathlib import Path

from jev_vehicle.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TYPESAFE_API_KEY", None)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_mock_prints_code_name_probability_confidence():
    expected = {
        "sedan": ("class_01", "승용차"),
        "truck3axle": ("class_07", "대형화물(3축)"),
        "bus": ("class_02", "버스(좌석)"),
    }
    for sample, (code, name) in expected.items():
        completed = subprocess.run(
            [sys.executable, "-m", "jev_vehicle", "classify", "--sample", sample, "--mock"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=_env(),
        )
        assert completed.returncode == 0, completed.stderr
        assert f"차종 코드: {code}" in completed.stdout
        assert f"한글명: {name}" in completed.stdout
        assert "confidence:" in completed.stdout
        assert "확률 분포:" in completed.stdout
        assert code in completed.stdout


def test_features_path_and_json(capsys):
    path = ROOT / "examples" / "sample_features" / "motorcycle.json"
    code = main(["classify", "--features", str(path), "--mock", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["class_code"] == "class_12"
    assert payload["name_ko"] == "이륜차"
    assert payload["needs_review"] is False
    assert "confidence" in payload
    assert abs(sum(payload["probabilities"].values()) - 1) < 1e-9


def test_ambiguous_cli_routes_to_review(capsys):
    code = main(["classify", "--sample", "ambiguous", "--mock"])
    assert code == 0
    text = capsys.readouterr().out
    assert "needs_review: true" in text
    assert "13번째 차종이 아닙니다" in text


def test_unknown_sample_fails_without_calling_live(capsys):
    code = main(["classify", "--sample", "not-a-vehicle", "--mock"])
    assert code == 1
    assert "샘플" in capsys.readouterr().err
