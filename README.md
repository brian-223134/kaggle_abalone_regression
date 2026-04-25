# Abalone Regression

Kaggle Playground Series S4E4 Abalone 데이터셋으로 `Rings`를 예측하는 회귀 프로젝트입니다. 평가지표는 RMSLE이며, 원본 데이터 탐색, 피처 생성, baseline 검증, 모델링, 제출 파일 생성 순서로 진행합니다.

## 환경 세팅 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Directory 구성

```
abalone-regression/
├── .venv/
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── proceed/
│       ├── train_fe_v1.csv
│       └── test_fe_v1.csv
├── notebooks/
│   ├── README.md
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_baseline.ipynb
├── requirements.txt
└── README.md
```

## 진행 상태

| 단계 | 파일 | 상태 | 설명 |
| --- | --- | --- | --- |
| EDA | `notebooks/01_eda.ipynb` | 완료 | 데이터 품질, 타깃 분포, 상관관계, 이상치, train/test drift 확인 |
| Feature Engineering | `notebooks/02_feature_engineering.ipynb` | 완료 | `data/proceed/train_fe_v1.csv`, `data/proceed/test_fe_v1.csv` 생성 |
| Baseline | `notebooks/03_baseline.ipynb` | 완료 | v1 feature dataset 기준 RMSLE baseline 측정 |
| Modeling | `notebooks/04_modeling.ipynb` | 예정 | 피처/모델/하이퍼파라미터 고도화 |
| Submission | `notebooks/05_submission.ipynb` | 예정 | 최종 test 예측 및 제출 파일 생성 |

## 현재 feature dataset

`02_feature_engineering.ipynb` 실행 결과로 v1 피처셋이 생성되어 있습니다.

| 파일 | shape | 설명 |
| --- | ---: | --- |
| `data/proceed/train_fe_v1.csv` | 90,615 x 19 | `id` + 17개 피처 + `Rings` |
| `data/proceed/test_fe_v1.csv` | 60,411 x 18 | `id` + 17개 피처 |

v1 피처는 원본 수치 피처, `Sex` one-hot encoding, `Volume`, `Density`, weight ratio, `Height_is_zero` indicator를 포함합니다.

## 현재 baseline

`03_baseline.ipynb`에서 v1 피처셋을 5-fold RMSLE로 평가했습니다.

| 모델 | CV RMSLE mean | CV RMSLE std |
| --- | ---: | ---: |
| HistGradientBoosting + log target | 0.149917 | 0.000889 |
| HistGradientBoosting | 0.150623 | 0.000923 |
| RandomForest | 0.151209 | 0.000756 |
| Ridge + log target | 0.158517 | 0.001720 |
| Ridge | 0.159990 | 0.001744 |
| Dummy median | 0.286738 | 0.001278 |

현재 best baseline은 `HistGradientBoosting_log_target`이며 OOF RMSLE는 `0.149919`입니다.

## 다음 작업

다음 단계는 `notebooks/04_modeling.ipynb` 생성입니다. 현재 best baseline인 `HistGradientBoosting_log_target`을 기준으로 feature v2, 이상치 처리, 모델 튜닝을 비교하는 것이 좋습니다.
