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
│       ├── test_fe_v1.csv
│       ├── train_fe_v2.csv
│       └── test_fe_v2.csv
├── notebooks/
│   ├── README.md
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_submission.ipynb
├── submissions/
│   └── submission_v2_hgb_log_leaf45_clip.csv
├── requirements.txt
└── README.md
```

## 진행 상태

| 단계 | 파일 | 상태 | 설명 |
| --- | --- | --- | --- |
| EDA | `notebooks/01_eda.ipynb` | 완료 | 데이터 품질, 타깃 분포, 상관관계, 이상치, train/test drift 확인 |
| Feature Engineering | `notebooks/02_feature_engineering.ipynb` | 완료 | `data/proceed/train_fe_v1.csv`, `data/proceed/test_fe_v1.csv` 생성 |
| Baseline | `notebooks/03_baseline.ipynb` | 완료 | v1 feature dataset 기준 RMSLE baseline 측정 |
| Modeling | `notebooks/04_modeling.ipynb` | 완료 | v2 feature dataset 생성 및 HGBR tuning |
| Submission | `notebooks/05_submission.ipynb` | 완료 | 최종 test 예측 및 제출 파일 생성 |

## 현재 feature dataset

`02_feature_engineering.ipynb`와 `04_modeling.ipynb` 실행 결과로 feature dataset이 생성되어 있습니다.

| 파일 | shape | 설명 |
| --- | ---: | --- |
| `data/proceed/train_fe_v1.csv` | 90,615 x 19 | `id` + 17개 피처 + `Rings` |
| `data/proceed/test_fe_v1.csv` | 60,411 x 18 | `id` + 17개 피처 |
| `data/proceed/train_fe_v2.csv` | 90,615 x 41 | `id` + 39개 피처 + `Rings` |
| `data/proceed/test_fe_v2.csv` | 60,411 x 40 | `id` + 39개 피처 |

v1 피처는 원본 수치 피처, `Sex` one-hot encoding, `Volume`, `Density`, weight ratio, `Height_is_zero` indicator를 포함합니다. v2는 여기에 shape ratio, component weight balance, log 피처를 추가합니다.

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

## 현재 modeling 결과

`04_modeling.ipynb`에서 v2 피처와 HGBR 튜닝 후보를 비교했습니다.

| 실험 | Feature set | CV RMSLE mean | CV RMSLE std |
| --- | --- | ---: | ---: |
| `v2_hgb_log_leaf45_clip` | v2 | 0.149826 | 0.000907 |
| `v2_hgb_log_leaf45` | v2 | 0.149875 | 0.000856 |
| `v1_hgb_log_baseline` | v1 | 0.149917 | 0.000889 |
| `v2_hgb_log_lr0035_iter650` | v2 | 0.149947 | 0.000891 |
| `v2_hgb_log_clip_005_995` | v2 | 0.149987 | 0.000932 |
| `v2_hgb_log_baseline` | v2 | 0.150008 | 0.000911 |

현재 best modeling 후보는 `v2_hgb_log_leaf45_clip`이며 OOF RMSLE는 `0.149828`입니다.

## 현재 submission

`05_submission.ipynb`에서 최종 모델을 전체 train에 재학습해 제출 파일을 생성했습니다.

| 파일 | rows | columns | 설명 |
| --- | ---: | --- | --- |
| `submissions/submission_v2_hgb_log_leaf45_clip.csv` | 60,411 | `id`, `Rings` | Kaggle 제출용 예측 파일 |

제출 파일은 `sample_submission.csv`와 동일한 id 순서를 유지하며, 예측값에는 결측/무한대/음수 값이 없습니다.

## 다음 작업

Kaggle에 `submissions/submission_v2_hgb_log_leaf45_clip.csv`를 제출한 뒤 public score를 기록합니다. 이후 개선은 public/private gap을 보며 feature v3, HGBR 추가 튜닝, 다른 부스팅 모델 비교 순서로 진행하면 됩니다.
