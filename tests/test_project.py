"""문서와 비밀 없는 설정."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_explains_architecture_and_links_the_diagram():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    diagram = ROOT / "docs" / "diagrams" / "architecture.png"
    assert diagram.is_file()
    assert diagram.stat().st_size > 1000
    assert "docs/diagrams/architecture.png" in readme
    assert "Jev" in readme
    assert "needs_review" in readme
    # 비전을 state로 줄인 뒤에야 Choice에 묻는 이유를 적습니다.
    assert "결정" in readme
    assert "https://docs.typesafe.ai/primitives/choice" in readme
    assert "https://docs.typesafe.ai/concepts/system-one" in readme


def test_license_env_and_series():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Choi Gyuseon" in license_text
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "TYPESAFE_API_KEY=" in example
    value = example.split("TYPESAFE_API_KEY=", 1)[1].strip()
    assert value == ""
    assert "TYPESAFE_API_KEY" in (ROOT / ".gitignore").read_text(encoding="utf-8")
    series = (ROOT / "docs" / "SERIES.md").read_text(encoding="utf-8")
    assert "연재 1회" in series
    assert "YOLO" in series
    assert "임계값" in series
