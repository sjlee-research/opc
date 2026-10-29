# OPC toolkit (Python / PyTorch)

KLayout에서 그린 target layout(GDS layer 1) → lithography simulation / OPC → 결과 GDS를 KLayout에서 확인.

## 실행

```bash
pip install -r requirements.txt
python run.py correct examples/square.gds              # 표준 model-based OPC → outputs/square_opc.gds + .png
python run.py correct my_drawing.png --pixel-nm 5      # 손으로 그린 PNG (흰 바탕, 1 pixel = 5 nm)
python run.py simulate examples/square.gds             # 보정 없이 찍히는 모양만
python run.py stage 1 2 3 4 5                          # 보고서용 비교 실험 → outputs/report/
python -m pytest -q                                    # 테스트
```

- 입력: GDS면 layer 1에 원하는 모양(target)을, PNG면 흰 바탕에 검은 도형으로 그려서 `examples/`에 넣으면 돼.
- 출력: `outputs/<이름>_opc.gds` (layer 1 target, 2 OPC mask, 3 보정 전 contour, 4 보정 후 contour)와
  `outputs/<이름>_opc.png` (왼쪽: 원래 mask와 그걸로 찍힌 모양, 오른쪽: OPC mask와 그걸로 찍힌 모양) 두 개.
- 설정은 전부 `config.yaml` 하나에 있어 (optics, resist, fragmentation, MB-OPC, ILT, MRC …).

## 파일 지도

```
opc_project/
├── run.py            실행 진입점 (correct / simulate / stage)
├── config.yaml       모든 설정
├── test_opc.py       테스트
├── examples/         입력 예시: square.gds (200 nm 정사각형), hand_drawn.png (5 nm/pixel)
├── opc/              코드 — 위에서 아래 순서로 의존
│   ├── config.py       설정 dataclass
│   ├── layout.py       geometry · exact rasterization · GDS/PNG 입출력 · 표준 test pattern
│   ├── litho.py        optics(Hopkins TCC → SOCS) · torch simulator · resist · EPE/CD 측정
│   ├── correction.py   fragmentation · 평가(EPE, PV band, MRC) · rule-based · model-based · ILT
│   ├── pipeline.py     layout 하나에 OPC 적용하고 GDS + PNG 저장 (run.py correct)
│   ├── experiments.py  보고서용 1~5단계 비교 실험 (run.py stage)
│   └── viz.py          figure
└── outputs/          결과 (correct → 여기에 바로, stage → outputs/report/)
```

## 보고서용 비교 실험 (`run.py stage`, 결과는 `outputs/report/`)

| 단계 | 내용 | figure / table |
|---|---|---|
| 1 | simulator 검증 (SOCS vs Abbe), 보정 전 proximity effect | Fig01–05, FigS3–S4, T1–T4 |
| 2 | rule-based OPC (bias table + serif + hammerhead) | Fig06, FigS2, T5 |
| 3 | model-based OPC (fragmentation + EPE feedback) | Fig07–09 |
| 4 | contribution 후보 A/B/C + ILT 비교 | Fig10–13, FigS1, T6–T7, results.gds |
| 5 | `summary.md` 생성 | — |

## 비교 실험 결과 GDS (`outputs/report/gds/results.gds`)

행 = method, 열 = pattern. KLayout에서 `File > Load Layer Properties`로 `layers.lyp`를 열면 색과 이름이 붙어.

| layer | 내용 |
|---|---|
| 1 | target |
| 2 | OPC mask |
| 3 | 보정 전 printed contour |
| 4 | 보정 후 printed contour |
| 10 | method / pattern 이름 (text) |

`run.py correct`의 결과(`outputs/<이름>_opc.gds`)도 layer 1~4는 같아.

## 방법 요약

- **Imaging**: scalar Hopkins, thin (Kirchhoff) dark-field binary mask. TCC = AᴴA를 Gram matrix로 고유분해해서 SOCS kernel 추출, defocus는 pupil phase. 전부 torch라 autograd 가능.
- **Resist**: constant threshold, anchor(60 nm line @ 180 nm pitch) dose-to-size calibration.
- **Rule-based OPC**: 1D 시뮬레이션으로 만든 space별 bias table + corner serif + line-end hammerhead.
- **Model-based OPC**: edge fragmentation → control site EPE → `d ← clip(d − λ·EPE)`, MRC guard. 결과는 Manhattan polygon.
- **Contribution 후보** (`config.yaml`의 `mbopc`): A `adaptive` (EPE 부호 이력으로 segment별 damping 조절), B `init: rule` (rule-based 결과로 시작), C `pw_aware` (dose/defocus corner EPE 평균 반영).
- **ILT** (비교용): sigmoid-relaxed pixel mask + Adam.
