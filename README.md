# Jev 차종 분류 (연재 1회)

최규선(Choi Gyuseon, 정보관리기술사, ITS)의 공개 연재입니다.
도로를 지나는 차를 **도로교통량조사 12종 차종분류** 중 하나로 고릅니다.
마지막 선택은 [TypeSafe Jev](https://docs.typesafe.ai/concepts/system-one)의 Choice입니다.

이번 v0.1.1은 그 12종이 한국건설기술연구원 분류표와 같도록 고친 뼈대입니다.
카메라 모델은 아직 없고, 사람이 만든 특징 JSON으로 같은 길을 걸어 봅니다.
다음 회차에 YOLO 같은 검출기를 앞에 붙여도 Jev에 묻는 질문과 `class_01` …
`class_12` 코드는 그대로 둡니다.

## 무엇을 하나요

교통량을 차종별로 세려면 차종이 먼저 정해져야 합니다.
공식 12종은 축수와 단위로 나뉩니다. 1단위는 차체 하나, 2단위는
견인차와 피견인차를 **한 대**로 센 조합입니다. 버스는 16인승 이상이면
한 종이고, 이륜차 칸은 없습니다.

일을 나눕니다.

1. **비전**이 화소에서 축수, 단위, 트레일러 형식, 적재 톤을 뽑습니다. 이번 회차는 mock이 이 자리를 대신합니다.
2. 그 힌트를 **state**라는 JSON으로 정리합니다.
3. **Jev Choice**가 12종 중 하나를 고르고, 종마다 확률과 confidence를 줍니다.
4. 표 밖이거나 confidence가 낮으면 프로그램이 집계를 `TRA_미분류`로 두고 `needs_review`를 켭니다.

## 왜 이미지를 Jev에 바로 넣지 않나요

Jev는 사진을 분류하는 CNN이 아닙니다.
글을 읽고 **정해진 답 중 하나**를 고르는 결정 모델입니다.
입력은 텍스트, JSON, 텍스트 배열입니다. 연재 1회의 본 경로는 이미지가 아닙니다.

검출기는 바뀝니다. Jev에게 필요한 것은 축수, 단위, `semi`인지 `full`인지,
최대 적재량이 몇 톤인지입니다. 그 JSON을 고정해 두면 앞단을 갈아 끼워도
Choice 층은 유지됩니다.

```text
영상 또는 샘플 JSON → 특징(VehicleFeatures) → state JSON → Jev Choice → 12종 + 확률
                                                         └ 표 밖이거나 confidence가 낮으면 TRA_미분류
```

## 구조

![Jev 차종 식별 파이프라인 (연재 1회)](docs/diagrams/architecture.png)

| 칸 | 이번 회차에서 |
| --- | --- |
| ① 입력 | 도로 CCTV를 가정합니다. 실제로는 샘플 JSON을 읽습니다. |
| ② 비전 특징 추출 | `MockVisionExtractor`. 이미지를 열지 않고, 가중치도 받지 않습니다. |
| ③ state | 축수, 단위(트레일러), 적재 톤. `VehicleFeatures.to_state()`. |
| ④ Jev | `Choice` 하나. 선택지는 `class_01` … `class_12`. |

그림 왼쪽의 `TRA_01`~`TRA_12`는 한국도로공사 AVC 원시자료 열이고,
`미분류`는 13번째 차종이 아닙니다. 오른쪽의 `needs_review`는
confidence가 임계값보다 낮을 때의 사람 검토입니다. 표 밖 차량도
같은 `TRA_미분류` 집계로 갑니다.

이후 회차는 [docs/SERIES.md](docs/SERIES.md)에 있습니다.

## 준비

Python 3.10 이상이 필요합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

`pip install -e .` 만으로도 패키지는 설치됩니다.
mock 모드와 테스트는 `typesafe-sdk`와 API 키가 없어도 됩니다.

## 빠른 시작 — mock (키 없음)

```bash
python -m jev_vehicle classify --sample sedan --mock
python -m jev_vehicle classify --sample bus --mock
python -m jev_vehicle classify --sample truck_2_5t --mock
python -m jev_vehicle classify --sample truck3axle --mock
python -m jev_vehicle classify --sample semi5 --mock
python -m jev_vehicle classify --sample full4 --mock
```

각 실행은 차종 코드, 한글명, AVC 열, 12칸 확률, confidence를 출력합니다.
이륜차와 야간 애매 샘플은 집계가 `TRA_미분류`입니다.

```bash
python -m jev_vehicle classify --sample motorcycle --mock
python -m jev_vehicle classify --sample ambiguous --mock
python -m jev_vehicle classify --features examples/sample_features/semi5.json --mock
python -m jev_vehicle classes
```

JSON이 필요하면 `--json`을 붙입니다.

## 빠른 시작 — live (Jev)

키는 [TypeSafe 콘솔](https://docs.typesafe.ai/introduction/quickstart)에서 발급합니다.
저장소에 넣지 마세요.

```bash
pip install -e ".[live]"
cp .env.example .env
# .env에 TYPESAFE_API_KEY=... 를 적습니다.
set -a && source .env && set +a
python -m jev_vehicle classify --sample sedan --live
```

`--mock`도 `--live`도 없으면, 키가 있을 때만 Jev를 호출합니다.
키가 없으면 mock입니다. 모델 이름은 `jev-latest`입니다.

퀵스타트 응답 JSON의 `answers`를 Python SDK에서는 이렇게 읽습니다.

```python
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient(model="jev-latest") as client:
    response = client.system_one(state=state, questions={"vehicle_class": question})

answer = response.answers["vehicle_class"]
answer.choice          # "class_01" 같은 코드
answer.probabilities   # 12종 확률
answer.confidence      # 0~1. 분포가 한곳에 몰릴수록 큽니다.
```

질문 문장과 경계는 `jev_vehicle.classifier.vehicle_class_question()`에 있습니다.
한글 차종명은 각 criteria의 `name_ko`와 `what` 앞에 있습니다.
Jev의 주 학습 언어가 영어라, 축수·단위·톤수 경계는 영어로 적었습니다.

## 12종

출처는 한국건설기술연구원 교통량정보제공시스템
[도로교통량조사 12종 차종분류표](http://www.road.re.kr/intro/intro_04.asp)입니다.
페이지의 행 이름은 승용차 미니트럭, 버스, 소형화물차 A처럼 `차`가 붙어 있고,
아래 한글명은 같은 행을 짧게 쓴 것입니다. 축수, 단위, 적재 톤 경계는
그 표의 문장을 따릅니다.

AVC 열은 한국도로공사
[고속도로 AVC 원시자료](https://www.data.go.kr/data/15045525/fileData.do)의
`TRA_01종` … `TRA_12종`, `TRA_미분류`입니다. `class_01`은 `TRA_01종`이고
`class_12`는 `TRA_12종`입니다.

| 코드 | 번호 | AVC 열 | 한글명 | 분류 기준 |
| --- | --- | --- | --- | --- |
| `class_01` | 1 | `TRA_01종` | 승용차·미니트럭 | 2축 1단위. 16인승 미만 여객, 또는 적재 1톤 미만 미니트럭 |
| `class_02` | 2 | `TRA_02종` | 버스 | 16인승 이상, 2축 1단위. 좌석형·입석형을 나누지 않음 |
| `class_03` | 3 | `TRA_03종` | 소형화물 A | 2축 1단위, 최대 적재 1톤 이상 2.5톤 미만 |
| `class_04` | 4 | `TRA_04종` | 소형화물 B | 2축 1단위, 최대 적재 2.5톤 이상 |
| `class_05` | 5 | `TRA_05종` | 중형화물 A | 3축 1단위 |
| `class_06` | 6 | `TRA_06종` | 중형화물 B | 4축 1단위 |
| `class_07` | 7 | `TRA_07종` | 중형화물 C | 5축 1단위 |
| `class_08` | 8 | `TRA_08종` | 대형화물 A | 세미트레일러, 4축 2단위 |
| `class_09` | 9 | `TRA_09종` | 대형화물 B | 풀트레일러, 4축 2단위 |
| `class_10` | 10 | `TRA_10종` | 대형화물 C | 세미트레일러, 5축 2단위 |
| `class_11` | 11 | `TRA_11종` | 대형화물 D | 풀트레일러, 5축 2단위 |
| `class_12` | 12 | `TRA_12종` | 대형화물 E | 세미트레일러, 6축 이상 2단위 |

2단위는 견인차와 피견인차를 따로 세지 않습니다. 조합 전체의 축수를 셉니다.
6축 이상 풀트레일러와 이륜차는 표에 없으므로 `TRA_미분류`입니다.
2.5톤은 4종에 포함되고, 2.5톤 미만이면서 1톤 이상이면 3종입니다.

내장 샘플은 `examples/sample_features/`와 패키지 안에 같은 파일로 있습니다.

| 샘플 | 데모가 집계하는 열 |
| --- | --- |
| `sedan` | `TRA_01` 승용차·미니트럭 |
| `bus` | `TRA_02` 버스 |
| `cargo_a` | `TRA_03` 소형화물 A (1.5톤) |
| `truck_2_5t` | `TRA_04` 소형화물 B (4.5톤, 2.5톤 이상) |
| `truck3axle` | `TRA_05` 중형화물 A |
| `truck4axle` | `TRA_06` 중형화물 B |
| `truck5axle` | `TRA_07` 중형화물 C |
| `semi4` | `TRA_08` 대형화물 A |
| `full4` | `TRA_09` 대형화물 B |
| `semi5` | `TRA_10` 대형화물 C |
| `full5` | `TRA_11` 대형화물 D |
| `semi6` | `TRA_12` 대형화물 E |
| `motorcycle` | `TRA_미분류` (표 밖) |
| `ambiguous` | `TRA_미분류` (confidence 낮음) |

## 낮은 confidence와 미분류

Choice는 항상 12개 중 하나를 고릅니다. `TRA_미분류`를 선택지에 넣지 않은
이유는, 그러면 확률이 차종이 아닌 쪽으로 빠지고 AVC의 12개 열과
어긋나기 때문입니다. 모르는 상태와 표 밖 차량은 **집계 열**로 다룹니다.

```text
이륜차 등 표 밖            →  집계 TRA_미분류, needs_review
confidence < 임계값        →  집계 TRA_미분류, needs_review
그 외                      →  집계 TRA_01 … TRA_12
```

Choice가 고른 코드는 리포트에 남습니다. 자동 집계에 넣는 값은 `집계` 칸입니다.
기본 임계값은 `0.45`입니다. 12지선다에서 이 값은 대략 최상위 확률 0.50에
해당합니다. confidence는 고른 차종이 맞을 확률 그 자체가 아니라, 확률이
한곳에 얼마나 몰렸는지입니다. `0.45`는 자리표시자이고 현장 보정은 연재 4회입니다.

```bash
python -m jev_vehicle classify --sample sedan --mock --threshold 0.99
```

## 코드 지도

```text
src/jev_vehicle/classes.py       12종, TRA_01종~TRA_12종, Choice criteria
src/jev_vehicle/features.py      축수, 단위, trailer_type, load_tonnage_t, to_state()
src/jev_vehicle/vision_mock.py   연재 1회 추출기
src/jev_vehicle/classifier.py    Jev 호출, MockJevClient, TRA_미분류
src/jev_vehicle/cli.py           python -m jev_vehicle
examples/sample_features/       샘플 특징 JSON
tests/                           키 없이 도는 단위 테스트
```

연재 2에서 검출기를 붙일 때는 `extract()`가 같은 `VehicleFeatures`를
돌려주게 하면 됩니다.

```python
from jev_vehicle import classify
from jev_vehicle.vision_mock import load_sample

features = load_sample("semi5")   # 나중에는 yolo.extract(image_path)
result = classify(features, mock=True)
print(result.class_code, result.name_ko, result.counted_as, result.needs_review)
```

특징 JSON에 정답 코드는 넣지 않습니다. 번호판, 얼굴, 영상 바이트도 필드가 아닙니다.

## 주의

`MockJevClient`는 오프라인 데모입니다. 분류표의 공식 판정도 아니고 Jev도 아닙니다.
축, 단위, 적재 톤이 분명한 샘플을 항상 같은 차종으로 돌려 테스트가
네트워크 없이 통과하게 만든 규칙입니다. 운영 교통량은 `--live`로 집계합니다.

## 연재

| 회차 | 상태 | 내용 |
| --- | --- | --- |
| 1회 v0.1.1 | 완료 | 공식 12종, mock 특징, state, Jev Choice, TRA_미분류 |
| 2회 | 계획 | YOLO 등 실제 검출. Jev 입력은 계속 state |
| 3회 | 계획 | 라벨 데이터와 혼동행렬 |
| 4회 | 계획 | confidence 임계값 보정 |
| 5회 | 계획 | AVC 원시자료 `TRA_01종`~`TRA_12종`, `TRA_미분류` |

자세한 설명은 [docs/SERIES.md](docs/SERIES.md)입니다.

## 라이선스

[MIT](LICENSE). 분류표와 TypeSafe 문서의 저작권은 각 권리자에게 있습니다.
이 저장소는 그 표를 코드로 가져오는 실습용 골격입니다.

## 문서

- [도로교통량조사 12종 차종분류표](http://www.road.re.kr/intro/intro_04.asp)
- [한국도로공사 AVC 원시자료](https://www.data.go.kr/data/15045525/fileData.do)
- [TypeSafe 빠른 시작](https://docs.typesafe.ai/introduction/quickstart)
- [System One](https://docs.typesafe.ai/concepts/system-one)
- [State](https://docs.typesafe.ai/concepts/state)
- [Choice](https://docs.typesafe.ai/primitives/choice)
- [Confidence](https://docs.typesafe.ai/confidence)
- [Python SDK](https://docs.typesafe.ai/sdk/python)
- [TypeSafe skill](https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md)
