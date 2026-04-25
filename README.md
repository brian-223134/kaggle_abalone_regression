# Abalone Regression

Kaggle Playground Series S4E4 Abalone 데이터셋으로 `Rings`를 예측하는 회귀 프로젝트입니다. 평가지표는 RMSLE이며, 원본 데이터 탐색, 피처 생성, baseline 검증, 모델링, 제출 파일 생성 순서로 진행합니다.

## 환경 세팅 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 재현 방법

원본 Kaggle CSV는 `data/raw/`에 둔 상태에서 아래 순서로 노트북을 실행합니다.

1. `notebooks/01_eda.ipynb`: 원본 데이터 품질, 타깃 분포, 상관관계, 이상치, train/test drift 확인
2. `notebooks/02_feature_engineering.ipynb`: v1 feature dataset 생성
3. `notebooks/03_baseline.ipynb`: v1 기준 baseline 모델 비교
4. `notebooks/04_modeling.ipynb`: v2 feature dataset 생성, HGBR tuning, 최종 실험 후보 선택
5. `notebooks/05_submission.ipynb`: 최종 모델 전체 train 재학습, test 예측, submission CSV 생성

`data/raw/`의 원본 파일은 수정하지 않고, 재사용 가능한 중간 산출물은 `data/proceed/`에 버전별 CSV로 저장합니다. `id`는 식별자이므로 학습 피처에서는 제외하고 제출 파일 생성 시에만 사용합니다.

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

## EDA 핵심 결론

`01_eda.ipynb`에서 확인한 내용은 이후 feature engineering과 validation 설계의 기준으로 사용했습니다.

- 결측치와 중복 행은 없어 기본 정제 비용은 낮습니다.
- `Rings`는 오른쪽 꼬리가 있는 count형 회귀 타깃이며, Kaggle 평가지표가 RMSLE이므로 `log1p(Rings)` 타깃 변환을 baseline에서 함께 비교했습니다.
- `Shell_weight`, `Height`, `Diameter`, `Length`, `Whole_weight`가 `Rings`와 비교적 강한 양의 상관을 보입니다.
- `Sex`별 `Rings` 분포 차이가 있어 one-hot encoding을 기본 처리로 사용했습니다.
- `Height == 0` 행과 비율 피처 극단값이 일부 있으므로 제거보다 indicator, median fill, quantile clipping을 CV로 비교하는 방향을 택했습니다.
- train/test 입력 분포 drift는 작아 보이지만, test 정보는 drift 점검에만 사용하고 validation은 train 내부 fold로만 구성했습니다.

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

## 최종 모델 구성

최종 submission은 `04_modeling.ipynb`에서 가장 좋은 CV 성능을 보인 `v2_hgb_log_leaf45_clip` 설정을 사용합니다.

- feature set: v2, 39개 학습 피처
- estimator: `HistGradientBoostingRegressor`
- target transform: `log1p(Rings)`로 학습 후 `expm1`로 복원
- selected params: `learning_rate=0.035`, `max_iter=600`, `max_leaf_nodes=45`, `min_samples_leaf=30`, `l2_regularization=0.04`
- outlier handling: train fold 기준 quantile clipping 0.5%~99.5%
- validation 기준: `Rings` quantile bin 기반 `StratifiedKFold`, 동일한 5-fold RMSLE에서 mean/std와 OOF RMSLE를 함께 확인
- prediction post-processing: RMSLE 계산과 제출 안정성을 위해 음수 예측은 0 이상으로 clipping

## 회귀 최적화 프로세스

이 프로젝트의 최적화는 단순히 모델을 바꿔가며 점수를 보는 방식이 아니라, RMSLE라는 목표 지표에 맞춰 데이터, 피처, validation, 모델, 진단을 순서대로 고정해 나가는 방식으로 진행했습니다.

### 1. 문제와 metric을 먼저 고정

타깃은 `Rings`이고 평가지표는 RMSLE입니다. RMSLE는 실제값과 예측값의 절대 차이보다 비율 차이에 더 민감하고, 큰 값에서의 과도한 오차보다 상대적 오차를 완화해서 봅니다. 따라서 일반 RMSE 최적화와 달리 아래 원칙을 적용했습니다.

- 예측값은 0 미만이 되면 안 되므로 평가와 제출 전 `np.clip(pred, 0, None)`을 적용합니다.
- `Rings`의 오른쪽 꼬리를 줄이기 위해 `log1p(Rings)` 타깃 변환을 비교합니다.
- 모델 선택은 train score가 아니라 validation RMSLE와 OOF RMSLE 기준으로 판단합니다.

### 2. 데이터 누수와 검증 기준을 통제

test set은 최종 예측 대상이므로, test 분포는 drift 확인 용도로만 사용하고 모델 선택에는 사용하지 않았습니다. 피처 결측치 보정, quantile clipping 같은 train 통계 기반 처리는 fold 내부 train split에서만 학습되도록 구성해 validation 누수를 줄였습니다.

검증은 `Rings`를 quantile bin으로 나눈 뒤 `StratifiedKFold` 5-fold CV로 수행했습니다. count형 연속 타깃을 그대로 stratify할 수 없기 때문에, 타깃 구간을 binning해서 fold별 타깃 분포가 크게 흔들리지 않도록 했습니다. 단일 holdout보다 fold별 변동을 함께 볼 수 있어 작은 개선이 우연인지 판단하기 쉽습니다. 모델 선택 시에는 `CV RMSLE mean`뿐 아니라 `CV RMSLE std`와 `OOF RMSLE`도 함께 확인했습니다.

### 3. EDA로 feature hypothesis 생성

EDA에서는 모델링 전에 어떤 정보가 타깃에 영향을 줄 수 있는지 가설을 세웠습니다.

- 크기 계열: `Length`, `Diameter`, `Height`
- 무게 계열: `Whole_weight`, `Shucked_weight`, `Viscera_weight`, `Shell_weight`
- 생물학적/물리적 비율: volume, density, shell ratio, shucked ratio
- 범주형 차이: `Sex`
- 이상치 후보: `Height == 0`, 비율 피처의 극단값

이 가설을 바로 복잡한 모델에 넣기보다 v1, v2 feature set으로 나누어 단계적으로 검증했습니다.

### 4. baseline으로 개선 기준선 확보

먼저 `Dummy_median`, `Ridge`, `RandomForest`, `HistGradientBoostingRegressor`를 같은 fold와 같은 RMSLE 함수로 비교했습니다. 이 단계의 목적은 최고 점수를 내는 것이 아니라, 이후 실험이 실제로 의미 있는 개선인지 판단할 기준선을 만드는 것입니다.

v1 기준 best baseline은 `HistGradientBoosting_log_target`이었고, OOF RMSLE는 `0.149919`였습니다. 이 값이 이후 v2 피처와 tuning의 비교 기준이 됩니다.

### 5. 피처셋을 버전으로 관리하며 실험

v1은 안정적인 시작점으로 구성했습니다.

- 원본 수치 피처 7개
- `Sex` one-hot encoding 3개
- `Volume`, `Density`, weight ratio, `Height_is_zero` 등 물리 기반 파생 피처 7개

v2는 v1에 shape ratio, component weight balance, log 피처를 추가했습니다. 피처 수는 17개에서 39개로 늘었고, 산출물은 `train_fe_v2.csv`, `test_fe_v2.csv`로 저장했습니다. 이렇게 버전을 분리하면 어떤 피처셋으로 어떤 점수가 나왔는지 추적하기 쉽고, 성능이 나빠졌을 때 되돌리기도 쉽습니다.

### 6. 모델 tuning은 한 번에 하나씩 비교

`04_modeling.ipynb`에서는 v1 baseline 재현, v2 baseline, clipping, learning rate/iteration 조합, leaf node 조정을 같은 validation 조건에서 비교했습니다. 실험 조건을 한 번에 많이 바꾸면 개선 원인을 알기 어렵기 때문에, 피처셋과 주요 하이퍼파라미터 변화를 실험명으로 명시했습니다.

가장 좋은 후보는 `v2_hgb_log_leaf45_clip`입니다.

- CV RMSLE mean: `0.149826`
- CV RMSLE std: `0.000907`
- OOF RMSLE: `0.149828`

baseline OOF `0.149919` 대비 개선폭은 크지 않지만, 동일한 fold 기준에서 반복적으로 비교한 결과 best 후보로 선택했습니다.

### 7. OOF diagnostics로 약점 확인

평균 점수만 보면 어떤 구간에서 모델이 실패하는지 알기 어렵습니다. 그래서 best model의 OOF prediction으로 residual 분포와 `Rings` 구간별 RMSLE를 확인했습니다. 특히 고연령 `Rings` 구간의 오차가 추가 개선 후보로 남았습니다.

Permutation importance는 어떤 피처가 validation 성능에 실제로 기여하는지 확인하는 보조 진단으로 사용했습니다. 중요도가 낮거나 불안정한 피처는 다음 feature v3 실험에서 제거 후보가 될 수 있습니다.

### 8. 최종 학습과 제출 파일 생성

최종 후보를 정한 뒤에는 validation fold가 아니라 전체 train 데이터로 같은 설정의 모델을 다시 학습했습니다. test 예측은 `sample_submission.csv`의 `id` 순서를 유지한 채 `Rings` 컬럼만 교체했고, 결측/무한대/음수 예측이 없는지 검증한 뒤 제출 파일로 저장했습니다.

이후 최적화는 Kaggle public/private score 차이를 확인한 뒤 진행하는 것이 좋습니다. CV와 public score가 함께 좋아지면 같은 방향의 feature/model 실험을 확장하고, public만 좋아지고 CV가 불안정하면 leaderboard overfitting 가능성을 의심해야 합니다.

## 현재 submission

`05_submission.ipynb`에서 최종 모델을 전체 train에 재학습해 제출 파일을 생성했습니다.

| 파일 | rows | columns | 설명 |
| --- | ---: | --- | --- |
| `submissions/submission_v2_hgb_log_leaf45_clip.csv` | 60,411 | `id`, `Rings` | Kaggle 제출용 예측 파일 |

제출 파일은 `sample_submission.csv`와 동일한 id 순서를 유지하며, 예측값에는 결측/무한대/음수 값이 없습니다.

## 다음 작업

Kaggle에 `submissions/submission_v2_hgb_log_leaf45_clip.csv`를 제출한 뒤 public score를 기록합니다. 이후 개선은 public/private gap을 보며 feature v3, HGBR 추가 튜닝, 다른 부스팅 모델 비교 순서로 진행하면 됩니다.
