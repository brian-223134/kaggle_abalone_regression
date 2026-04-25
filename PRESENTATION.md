# Abalone Regression 발표 준비 자료

## 1. 발표 목적

이 프로젝트는 Kaggle Playground Series S4E4 Abalone 데이터셋에서 전복의 `Rings`를 예측하는 회귀 문제를 다룹니다. 평가지표는 RMSLE이며, 단순히 제출 파일 하나를 만드는 것보다 데이터 탐색, 피처 생성, 검증 설계, 모델 개선, 제출 파일 생성까지의 과정을 재현 가능한 실험 흐름으로 정리하는 데 초점을 두었습니다.

발표의 핵심 메시지는 다음과 같습니다.

- RMSLE 문제에서는 타깃 분포와 작은 값 구간의 상대 오차를 먼저 이해해야 합니다.
- 모델 성능 개선은 무작정 모델을 바꾸는 방식보다, 같은 validation 기준에서 하나씩 비교해야 해석할 수 있습니다.
- 초기 HGBR 단일 모델은 OOF RMSLE `0.149828` 수준이었고, target encoding, UCI 원본 데이터, boosting ensemble을 거쳐 `0.147529`까지 개선했습니다.

## 2. 문제 정의

Abalone 데이터는 전복의 길이, 지름, 높이, 무게, 성별 정보로부터 `Rings`를 예측하는 문제입니다. 실제 전복의 나이는 `Rings + 1.5`로 알려져 있지만, 대회 타깃은 나이가 아니라 `Rings` 자체입니다.

평가지표는 RMSLE입니다.

```text
RMSLE = sqrt(mean((log1p(pred) - log1p(actual))^2))
```

이 지표는 일반 RMSE와 달리 절대 오차보다 비율 오차에 더 민감합니다. 따라서 `Rings`가 작은 구간에서의 상대적 실수가 점수에 크게 반영되고, 예측값이 음수가 되지 않도록 항상 clipping이 필요합니다. 이 특성 때문에 대부분의 모델에서 `log1p(Rings)`를 학습하고 `expm1`로 복원하는 전략을 사용했습니다.

## 3. 노트북을 나눈 이유

처음부터 하나의 노트북에 EDA, 피처 생성, 모델링, 제출 파일 생성을 모두 넣으면 실험 결과를 추적하기 어렵습니다. 그래서 각 노트북은 하나의 질문에 답하도록 분리했습니다.

| 파일 | 담당 질문 | 역할 |
| --- | --- | --- |
| `01_eda.ipynb` | 데이터가 어떤 형태인가? | 품질, 분포, 상관관계, 이상치, train/test drift 확인 |
| `02_feature_engineering.ipynb` | 어떤 정보를 학습 가능한 숫자로 만들 것인가? | v1 피처셋 생성 |
| `03_baseline.ipynb` | 개선 기준선은 어디인가? | 단순 모델 비교와 RMSLE baseline 확보 |
| `04_modeling.ipynb` | 기존 방향에서 얼마나 개선되는가? | v2 피처, HGBR tuning, OOF diagnostics |
| `05_submission.ipynb` | 제출 파일이 올바르게 만들어지는가? | 최종 HGBR 학습과 submission 검증 |
| `06_target_encoding_lgbm.py` | 상위권 접근의 핵심 피처가 먹히는가? | OOF-safe target encoding + LGBM |
| `07_external_data_lgbm.py` | 원본 UCI 데이터가 도움이 되는가? | 외부 데이터 결합 실험 |
| `08_boosting_ensemble.py` | 모델 다양성으로 더 낮출 수 있는가? | LGBM/XGBoost/CatBoost 앙상블 |

이 구조 덕분에 특정 실험이 좋아졌을 때 무엇이 원인인지 추적할 수 있고, 성능이 나빠진 실험도 쉽게 되돌릴 수 있습니다.

## 4. 전체 실험 흐름

프로젝트는 세 단계로 나눌 수 있습니다.

1. 기본 파이프라인 구축
2. 기존 모델링 방향의 한계 확인
3. 상위권 풀이 기반 개선

처음에는 EDA를 통해 타깃 분포, 컬럼 의미, 이상치, train/test drift를 확인했습니다. 이후 물리적으로 의미 있는 ratio와 log 피처를 만들어 HGBR 모델을 튜닝했습니다. 이 단계에서 OOF RMSLE는 `0.149828`까지 개선됐지만, baseline 대비 개선폭이 작아 fold 변동보다 뚜렷하게 크지 않았습니다.

이후 Public 상위권 풀이를 참고해 방향을 바꿨습니다. 핵심은 `Shell_weight`, `Height`, `Whole_weight` 같은 강한 원본 피처를 target mean encoding으로 바꾸는 것이었습니다. 다만 target encoding은 validation 누수가 발생하기 쉬우므로, 각 fold의 train split에서만 mapping을 학습하고 validation split에는 그 mapping만 적용했습니다.

## 5. 각 노트북의 원리와 역할

### `01_eda.ipynb`: 탐색과 가설 생성

이 노트북은 모델을 학습하기 전에 데이터를 이해하기 위한 단계입니다.

확인한 내용:

- 결측치와 중복 행 여부
- `Rings` 타깃 분포
- 수치 피처와 `Rings`의 상관관계
- `Sex`별 타깃 분포 차이
- `Height == 0` 같은 이상치
- train/test 입력 분포 drift

이 단계에서 얻은 주요 결론은 `Rings`가 오른쪽 꼬리를 가진 count형 타깃이라는 점입니다. 따라서 RMSLE와 잘 맞도록 log target 학습을 비교해야 한다는 판단을 했습니다. 또한 `Shell_weight`, `Height`, `Whole_weight` 계열이 중요한 정보일 가능성이 높다는 점을 확인했습니다.

### `02_feature_engineering.ipynb`: v1 피처셋 생성

이 노트북은 raw CSV를 모델 입력용 피처셋으로 바꾸는 역할입니다.

주요 처리:

- 헷갈리는 컬럼명 rename
- `Sex` one-hot encoding
- `Volume`, `Density` 생성
- `Shucked_ratio`, `Viscera_ratio`, `Shell_ratio`, `Shell_to_shucked` 생성
- `Height_is_zero` indicator 추가
- 결측/무한대 값 median fill

이 노트북은 성능을 직접 평가하기보다, 이후 여러 모델이 같은 입력을 사용하도록 안정적인 v1 데이터셋을 만드는 데 목적이 있습니다.

### `03_baseline.ipynb`: 비교 기준선 확보

baseline의 목적은 최고 점수를 만드는 것이 아니라, 이후 실험이 실제 개선인지 판단할 기준을 만드는 것입니다.

비교한 모델:

- Dummy median
- Ridge
- Ridge + log target
- RandomForest
- HistGradientBoosting
- HistGradientBoosting + log target

검증은 `Rings`를 구간화한 `StratifiedKFold`를 사용했습니다. 회귀 문제라 원래 stratify가 직접 되지 않지만, 타깃을 quantile bin으로 나누면 fold별 타깃 분포를 안정적으로 맞출 수 있습니다.

결과적으로 `HistGradientBoosting + log target`이 OOF RMSLE `0.149919`로 가장 좋았습니다.

### `04_modeling.ipynb`: v2 피처와 HGBR 개선

이 단계에서는 v1 피처에 shape ratio, component balance, log 피처를 추가한 v2 피처셋을 만들고, HGBR 하이퍼파라미터를 조정했습니다.

추가한 피처 예시:

- `Area`
- `Diameter_to_Length`
- `Height_to_Length`
- `Component_weight_sum`
- `Residual_weight_ratio`
- `log1p_...` 계열 피처

최종 HGBR 후보는 `v2_hgb_log_leaf45_clip`이며 OOF RMSLE는 `0.149828`이었습니다. baseline 대비 개선은 있었지만, 개선폭이 크지 않았습니다. 이 결과는 “단순 ratio/log 피처와 HGBR 튜닝만으로는 상위권 접근에 부족하다”는 판단의 근거가 되었습니다.

### `05_submission.ipynb`: 제출 파일 생성

이 노트북은 모델 실험과 제출 파일 생성을 분리하기 위해 만들었습니다.

수행 내용:

- 선택된 HGBR 모델을 전체 train에 재학습
- test prediction 생성
- 음수 예측 clipping
- `sample_submission.csv`와 id 순서 일치 검증
- submission CSV 저장

이 단계의 핵심은 성능 개선이 아니라 재현성과 제출 형식 검증입니다.

### `06_target_encoding_lgbm.py`: target encoding 전환

기존 제출의 등수가 낮았던 것을 확인한 후, Public 상위권 풀이를 참고해 가장 큰 차이를 만든 단계입니다.

추가한 target encoding 피처:

- `Sweight_avg`: `Shell_weight`별 평균 `Rings`
- `Height_avg`: clipping된 `Height`별 평균 `Rings`
- `Weight_avg`: `Whole_weight`별 평균 `Rings`
- `Viscera_weight_avg`: `Viscera_weight`별 평균 `Rings`
- `Sex_avg`: `Sex`별 평균 `Rings`

중요한 점은 target encoding을 전체 train으로 만든 뒤 validation에 적용하면 누수가 생긴다는 것입니다. 그래서 각 fold마다 train split으로만 mapping을 만들고 validation split에는 그 mapping만 적용했습니다. test 예측을 만들 때만 전체 train으로 mapping을 다시 학습했습니다.

이 단계에서 OOF RMSLE는 `0.147765`까지 내려갔습니다. 기존 HGBR best인 `0.149828`과 비교하면 가장 큰 개선입니다.

### `07_external_data_lgbm.py`: UCI 원본 데이터 추가

Kaggle Playground 데이터는 UCI Abalone 원본 데이터 기반으로 합성된 데이터입니다. 따라서 UCI 원본 4,177건을 추가 학습 데이터로 붙여 모델이 원래 분포를 더 잘 보도록 했습니다.

검증 시 주의한 점:

- validation은 Kaggle train fold에 대해서만 계산
- UCI 데이터는 해당 fold의 train split에만 결합
- UCI 데이터도 Kaggle v2와 같은 피처 스키마로 변환

이 실험은 OOF RMSLE를 `0.147607`로 소폭 개선했습니다. 큰 폭은 아니지만, 상위권 풀이에서 언급된 public score 개선 가능성을 재현 가능한 방식으로 반영했습니다.

### `08_boosting_ensemble.py`: 모델 앙상블

마지막으로 LGBM과 같은 피처 조건에서 XGBoost, CatBoost를 추가 학습하고 OOF prediction을 저장했습니다. 세 모델의 예측을 단순 평균한 결과 OOF RMSLE는 `0.147529`로 더 낮아졌습니다.

결과:

| 모델 또는 앙상블 | OOF RMSLE |
| --- | ---: |
| LGBM | 0.147607 |
| XGBoost | 0.147903 |
| CatBoost | 0.147773 |
| 1:1:1 평균 앙상블 | 0.147529 |

여기서 중요한 점은 단일 모델이 가장 좋은 LGBM이더라도, XGBoost와 CatBoost가 서로 다른 오차 패턴을 가지면 평균 앙상블이 더 안정적일 수 있다는 것입니다.

## 6. 성능 개선 요약

| 단계 | 실험 | OOF RMSLE | 해석 |
| --- | --- | ---: | --- |
| Baseline | HGBR + log target | 0.149919 | 첫 기준선 |
| Modeling | v2 HGBR tuning | 0.149828 | 작은 개선 |
| Target Encoding | LGBM + target encoding | 0.147765 | 핵심 개선 |
| External Data | UCI 원본 추가 | 0.147607 | 소폭 개선 |
| Ensemble | LGBM/XGB/CatBoost 평균 | 0.147529 | 최종 후보 |

발표에서는 이 표를 중심으로 “어떤 변화가 실제로 점수를 움직였는가”를 설명하면 됩니다. 가장 큰 전환점은 HGBR 튜닝이 아니라 target encoding입니다.

## 7. 발표 흐름 제안

1. 문제 소개: 전복의 물리 측정값으로 `Rings`를 예측한다.
2. Metric 소개: RMSLE라서 log target과 음수 clipping이 중요하다.
3. 데이터 탐색: 타깃 분포, 주요 피처, 이상치, drift를 확인했다.
4. Baseline: 같은 fold와 metric으로 기준선을 만들었다.
5. 첫 모델링: 물리 파생 피처와 HGBR 튜닝으로는 개선폭이 작았다.
6. 개선 방향 전환: 상위권 풀이에서 target encoding이 핵심임을 확인했다.
7. 누수 방지: target encoding은 fold train에서만 fit했다.
8. 외부 데이터: UCI 원본 데이터를 fold train에만 결합했다.
9. 앙상블: 세 boosting 모델을 평균해 OOF를 더 낮췄다.
10. 결론: 최종 후보는 `submission_v5_boosting_ensemble.csv`이다.

## 8. 발표 시 강조할 포인트

- `id`는 식별자이므로 학습 피처에서 제외했습니다.
- test set은 drift 확인과 최종 예측에만 사용했고, 모델 선택에는 사용하지 않았습니다.
- target encoding은 강력하지만 누수 위험이 크므로 OOF-safe 방식으로 구현했습니다.
- CV가 좋아지는 방향과 public score가 항상 일치하지 않을 수 있으므로 제출 로그를 별도로 관리해야 합니다.
- HGBR에서 LGBM/XGBoost/CatBoost로 넘어간 이유는 tabular boosting 모델의 표현력과 상위권 풀이의 실증 결과 때문입니다.

## 9. 다음 개선 후보

- Optuna로 LGBM/XGBoost/CatBoost 하이퍼파라미터 튜닝
- target encoding smoothing 적용
- `Shell_weight`, `Height`, `Whole_weight` rounding 후 target encoding 비교
- tail 구간(`Rings >= 16`) 전용 진단과 calibration
- Kaggle public/private score를 README 또는 별도 submission log에 기록
