# Notebooks

이 디렉터리는 Abalone `Rings` 회귀 예측 프로젝트의 실험 흐름을 기록합니다. 노트북은 원본 데이터 탐색에서 시작해 피처셋 생성, baseline 검증, 모델 고도화, 제출 파일 생성 순서로 진행합니다.

## 원칙

- `../data/raw/`의 원본 CSV는 수정하지 않습니다.
- 전처리와 피처 엔지니어링 결과는 `../data/proceed/`에 버전별 CSV로 저장합니다.
- 모델 검증 기준은 Kaggle 평가 지표인 RMSLE입니다.
- `id`는 식별자이므로 학습 피처에서 제외하고, 제출 파일 생성 시에만 사용합니다.
- 새 피처셋을 만들 때는 `train_fe_v1.csv`, `test_fe_v1.csv`처럼 버전 suffix를 붙입니다.

## 현재 노트북

| 노트북 | 상태 | 목적 | 주요 산출물 |
| --- | --- | --- | --- |
| `01_eda.ipynb` | 작성됨 | 데이터 품질, 타깃 분포, 상관관계, 이상치, train/test drift 확인 | EDA findings, 피처 엔지니어링 후보 |

`01_eda.ipynb`에서 확인한 다음 모델링 방향은 아래와 같습니다.

- `Whole weight.1`, `Whole weight.2`는 각각 `Shucked_weight`, `Viscera_weight`로 rename합니다.
- `Sex`는 범주형 변수이므로 one-hot encoding을 기본 후보로 둡니다.
- `Rings`는 오른쪽 꼬리가 있어 `log1p(Rings)` 타깃 변환을 baseline에서 비교합니다.
- `Height == 0` 행이 일부 있으므로 제거, 클리핑, indicator 추가, 무처리 중 무엇이 좋은지 CV로 비교합니다.
- `Volume`, `Density`, `Shucked_ratio`, `Viscera_ratio`, `Shell_ratio` 같은 물리 기반 파생 피처를 후보로 둡니다.

## 권장 노트북 순서

### 1. `01_eda.ipynb`

이미 작성된 탐색 노트북입니다. 다음 단계를 위한 의사결정 근거를 남기는 용도이며, `data/proceed`에는 파일을 저장하지 않습니다.

### 2. `02_feature_engineering.ipynb`

다음에 만드는 것이 가장 좋습니다. `data/README.md`에 설명된 `../data/proceed/` 폴더를 실제로 채우는 노트북입니다.

권장 작업:

- `../data/raw/train.csv`, `../data/raw/test.csv` 로드
- 컬럼 rename
  - `Whole weight` -> `Whole_weight`
  - `Whole weight.1` -> `Shucked_weight`
  - `Whole weight.2` -> `Viscera_weight`
  - `Shell weight` -> `Shell_weight`
- 공통 피처 함수 작성
  - `Volume = Length * Diameter * Height`
  - `Density = Whole_weight / Volume`
  - `Shucked_ratio = Shucked_weight / Whole_weight`
  - `Viscera_ratio = Viscera_weight / Whole_weight`
  - `Shell_ratio = Shell_weight / Whole_weight`
  - `Shell_to_shucked = Shell_weight / Shucked_weight`
  - `Height_is_zero` indicator
- `Sex` one-hot encoding
- `id`, `Rings` 보존 여부 확인
  - train: `id`, 피처, `Rings`
  - test: `id`, 피처
- train/test 컬럼 정렬과 결측/무한대 값 검증
- 결과 저장
  - `../data/proceed/train_fe_v1.csv`
  - `../data/proceed/test_fe_v1.csv`

이 노트북은 모델 성능을 직접 평가하기보다, 재사용 가능한 첫 번째 feature dataset을 만드는 데 집중하는 것이 좋습니다.

### 3. `03_baseline.ipynb`

`../data/proceed/train_fe_v1.csv`와 `../data/proceed/test_fe_v1.csv`를 입력으로 받아 baseline 성능을 측정합니다.

권장 작업:

- RMSLE metric 함수 구현
- KFold 또는 StratifiedKFold-like split 구성
  - `Rings` 구간을 binning해서 fold별 타깃 분포를 맞추는 방식 권장
- 단순 모델 비교
  - Ridge 또는 ElasticNet
  - RandomForestRegressor
  - HistGradientBoostingRegressor
- `Rings` 직접 예측과 `log1p(Rings)` 학습 후 `expm1` 복원 비교
- feature v1의 CV RMSLE 기록

### 4. `04_modeling.ipynb`

baseline 이후 성능 개선 실험을 모읍니다.

권장 작업:

- feature v2 후보 실험
- 이상치 처리 전략 비교
- 트리 기반 모델 하이퍼파라미터 튜닝
- 앙상블 또는 stacking 검토
- 최종 validation score와 선택 이유 기록

### 5. `05_submission.ipynb`

최종 모델로 test 예측을 만들고 Kaggle 제출 파일을 생성합니다.

권장 작업:

- 최종 모델 재학습
- test prediction 생성
- 음수 예측 방지: `np.clip(pred, 0, None)`
- `sample_submission.csv` 형식 검증
- `../submissions/submission_*.csv` 저장

## 다음 작업

바로 다음에는 `02_feature_engineering.ipynb`를 생성해 `../data/proceed/train_fe_v1.csv`와 `../data/proceed/test_fe_v1.csv`를 만드는 것이 좋습니다. 그래야 baseline 노트북이 원본 CSV를 매번 다시 가공하지 않고, 고정된 feature dataset을 기준으로 모델 성능을 비교할 수 있습니다.
