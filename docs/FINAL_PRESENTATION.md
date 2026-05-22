# Abalone Regression 최종 발표자료 — PPT 생성 명세서

> 이 문서는 최종 발표용 슬라이드 덱을 만들기 위한 단일 입력 명세서입니다.
> 기존 `docs/PRESENTATION.md`가 중간 발표식 실험 흐름 정리라면, 이 문서는 최종 결과와 검증 논리를 앞에 두는 발표 구조입니다.
> 각 슬라이드는 `Layout / Body / Visual / Notes` 형식으로 작성되어 있으며, PPT 생성 에이전트는 이 구조를 그대로 슬라이드 초안으로 변환하면 됩니다.

---

## 0. 최종 발표 작성 원칙

### 0.1. 발표 목표

- 발표 분량: 15분 발표 + 5분 Q&A
- 청중: 머신러닝/데이터 분석에 익숙한 동료
- 발표 목적: Kaggle Abalone 회귀 문제에서 최종 후보 모델이 어떻게 만들어졌는지, 그리고 성능 개선이 재현 가능한 검증 설계 위에서 이루어졌음을 설명
- 최종 후보: `submissions/submission_v5_boosting_ensemble.csv`
- 최종 OOF RMSLE: **0.147529**

### 0.2. 중간 발표에서 최종 발표로 바꿀 점

| 구분 | 중간 발표 | 최종 발표 |
| --- | --- | --- |
| 중심 질문 | 무엇을 실험할 것인가? | 최종적으로 무엇이 효과가 있었는가? |
| 발표 순서 | EDA → baseline → 개선 계획 | 최종 결과 → 검증 설계 → 핵심 개선 요인 |
| 강조 대상 | 프로젝트 구성과 실험 흐름 | target encoding, 외부 데이터, 앙상블의 기여 |
| 결론 | 다음 실험 후보 | 최종 제출 후보와 남은 한계 |

### 0.3. 최종 발표의 핵심 메시지

1. **RMSLE 문제에서는 `log1p(target)` 학습과 음수 clipping이 기본 전제다.**
2. **작은 점수 차이를 해석하려면 fold, metric, OOF 기준을 고정해야 한다.**
3. **가장 큰 개선은 HGBR 튜닝이 아니라 fold-safe target encoding에서 나왔다.**

### 0.4. Placeholder

- 발표자: `[발표자]`
- 발표 일자: `[YYYY-MM-DD]`
- 소속/조직: `[조직]`
- Kaggle public score: `[public score TBD]`
- Kaggle private score: `[private score TBD]`

---

## 1. 최종 발표 서사

이 발표는 단순히 "점수가 좋아졌다"는 이야기가 아니라, **RMSLE 회귀 문제를 재현 가능한 실험 단위로 쪼개고, 각 단계의 의사결정을 검증한 결과 최종 OOF RMSLE 0.147529까지 도달했다**는 이야기입니다.

발표 흐름은 다음 4단계입니다.

1. 문제와 metric을 먼저 고정했다.
2. baseline과 validation 기준을 만들어 작은 개선을 해석 가능하게 만들었다.
3. HGBR 중심의 feature/tuning 개선은 한계가 있었고, target encoding이 전환점이었다.
4. 외부 UCI 데이터와 LGBM/XGBoost/CatBoost 앙상블로 최종 후보를 만들었다.

---

## 2. 슬라이드 블루프린트

### 슬라이드 1 — 제목

- Layout: `title`
- Body:
  - `Abalone Regression`
  - `RMSLE 최적화와 최종 제출 후보 정리`
  - `OOF RMSLE 0.149828 → 0.147529`
  - `[발표자] · [조직] · [YYYY-MM-DD]`
- Visual: `none`
- Notes: 발표 시작. 이 발표는 Abalone `Rings` 예측 문제에서 어떤 실험 설계를 통해 최종 OOF RMSLE 0.147529까지 개선했는지 설명한다. 핵심은 단순 모델 교체가 아니라 metric에 맞춘 target transform, 누수 없는 validation, target encoding, 외부 데이터, 앙상블을 순서대로 검증한 과정이다.

### 슬라이드 2 — 최종 결과 한눈에

- Layout: `summary`
- Body:
  - 최종 후보: `submission_v5_boosting_ensemble.csv`
  - 최종 OOF RMSLE: **0.147529**
  - 기존 HGBR best: `0.149828`
  - 가장 큰 개선: target encoding 적용 구간
  - Kaggle score: `[public score TBD] / [private score TBD]`
- Visual: `V1` — 전체 성능 개선 미니 라인 차트
- Notes: 결과를 먼저 제시한다. HGBR 튜닝만으로는 0.1498 부근에서 한계가 있었고, target encoding 이후 0.147대에 진입했다. Kaggle public/private 점수가 아직 없다면 현재 평가는 OOF RMSLE 기준임을 명확히 말한다.

### 슬라이드 3 — 문제 정의와 평가지표

- Layout: `bullets`
- Body:
  - 문제: Kaggle Playground S4E4 Abalone `Rings` 회귀
  - 입력: 크기, 무게, `Sex`
  - 타깃: `Rings` (`Age = Rings + 1.5`)
  - 평가지표: RMSLE
  - 원칙: `log1p(Rings)` 학습, `expm1` 복원, `np.clip(pred, 0, None)`
- Visual: `V2` — RMSLE 민감도 모식도
- Notes: RMSLE는 절대 오차보다 비율 오차에 민감하다. 같은 2만큼의 오차라도 실제값이 작은 구간에서 더 크게 반영된다. 따라서 모든 주요 모델은 log target으로 학습하고 예측 복원 후 음수 값을 clipping했다.

### 슬라이드 4 — 데이터와 프로젝트 디렉토리

- Layout: `diagram`
- Body:
  - `data/raw`: 원본 Kaggle CSV
  - `data/proceed`: v1/v2 feature dataset, OOF/test prediction
  - `notebooks`: `01_eda`부터 `08_boosting_ensemble`까지 실험 단위
  - `submissions`: 제출 후보 CSV
  - `docs`: 발표 자료와 PPT 산출물
- Visual: `V3` — 프로젝트 디렉토리 기반 재현 흐름도
- Notes: README의 디렉토리 구조를 발표용으로 압축해 보여준다. 원본 데이터는 수정하지 않고, 재사용 가능한 중간 산출물은 `data/proceed`에 저장했다. 실험 코드와 제출 파일, 발표 문서를 분리해 결과 추적과 재현성을 확보했다.

### 슬라이드 5 — EDA에서 얻은 모델링 가설

- Layout: `bullets`
- Body:
  - 결측치/중복 없음 → 정제 비용 낮음
  - `Rings`는 오른쪽 꼬리가 있는 count형 타깃
  - `Shell_weight`, `Height`, `Diameter`, `Length`, `Whole_weight`가 강한 신호
  - `Sex`별 분포 차이 → one-hot + target encoding 후보
  - train/test drift는 작아 보이나 모델 선택에는 test 미사용
- Visual: `V4` — `Rings` 분포 + 주요 상관 피처 콜아웃
- Notes: EDA는 슬라이드 한 장으로 압축한다. 최종 발표에서는 EDA 자체보다 EDA가 이후 선택을 어떻게 만들었는지가 중요하다. log target, weight/shape ratio, `Sex` 처리, drift 점검이라는 네 가지 의사결정으로 연결한다.

### 슬라이드 6 — 검증 설계

- Layout: `process`
- Body:
  - `Rings` quantile bin 기반 5-fold StratifiedKFold
  - 모델 선택 기준: CV RMSLE mean/std + OOF RMSLE
  - test set은 drift 확인과 최종 예측에만 사용
  - fold 내부 train 통계만 validation에 적용
  - 누수 가능성이 큰 target encoding은 별도 OOF-safe 구현
- Visual: `V5` — train/fold/test 역할 분리 다이어그램
- Notes: 작은 점수 차이를 해석하려면 validation 체계를 먼저 고정해야 한다. 특히 target encoding은 validation 정보가 mapping에 들어가면 점수가 과대평가된다. 그래서 fold train에서만 통계를 학습하고 validation에는 적용만 하는 원칙을 세웠다.

### 슬라이드 7 — 실험 파이프라인

- Layout: `timeline`
- Body:
  - `01_eda`: 데이터와 metric 이해
  - `02_feature_engineering`: v1 feature dataset
  - `03_baseline`: 개선 기준선
  - `04_modeling`~`05_submission`: HGBR 후보와 제출 검증
  - `06`~`08`: target encoding, 외부 데이터, 앙상블
- Visual: `V6` — `01`부터 `08`까지 노트북/스크립트 흐름도
- Notes: 노트북을 나눈 이유는 각 파일이 답하는 질문이 다르기 때문이다. EDA, feature engineering, baseline, modeling, submission, 개선 실험을 분리하면 어떤 변화가 점수에 영향을 줬는지 추적하기 쉽다. 이 구조가 README의 프로젝트 구성과 연결된다.

### 슬라이드 8 — Baseline: 출발점 만들기

- Layout: `chart`
- Body:
  - 같은 fold, 같은 RMSLE 함수로 6개 모델 비교
  - best baseline: `HistGradientBoosting + log target`
  - CV RMSLE mean: `0.149917`
  - OOF RMSLE: `0.149919`
  - 이후 모든 개선의 기준선
- Visual: `V7` — baseline 모델 가로 막대 차트
- Notes: baseline의 목적은 최고 점수가 아니라 기준선을 만드는 것이다. Ridge나 RandomForest보다 HGBR + log target이 명확히 좋았고, 이 값을 기준으로 v2 feature와 HGBR tuning을 평가했다.

### 슬라이드 9 — HGBR 튜닝의 한계

- Layout: `chart`
- Body:
  - v1: 17개 피처
  - v2: 39개 피처, shape ratio/component balance/log 피처 추가
  - best: `v2_hgb_log_leaf45_clip`
  - OOF RMSLE: `0.149828`
  - baseline 대비 개선폭은 약 `0.0001`
- Visual: `V8` — HGBR modeling 후보 비교 막대
- Notes: v2 feature와 HGBR tuning은 개선을 만들었지만 폭이 작았다. CV std가 약 0.0009 수준임을 감안하면 이 방향만으로는 큰 전환이 어렵다고 판단했다. 여기서 상위권 풀이 기반의 target encoding으로 방향을 바꿨다.

### 슬라이드 10 — 전환점: Target Encoding

- Layout: `highlight`
- Body:
  - 추가 피처: `Sweight_avg`, `Height_avg`, `Weight_avg`, `Viscera_weight_avg`, `Sex_avg`
  - 모델: LGBM + log target
  - OOF RMSLE: **0.147765**
  - HGBR best 대비 약 `-0.002063`
  - 전체 실험에서 가장 큰 성능 개선 구간
- Visual: `V9` — target encoding 전후 성능 비교
- Notes: 최종 발표의 핵심 슬라이드다. 모델 교체만으로도 일부 개선이 있었지만, 가장 큰 낙폭은 target encoding에서 나왔다. 강한 원본 피처를 target mean 정보로 바꾸되 validation 누수를 막는 방식이 중요했다.

### 슬라이드 11 — Target Encoding 누수 방지

- Layout: `diagram`
- Body:
  - fold train split에서만 target mapping 학습
  - validation split에는 mapping 적용만 수행
  - test 예측 시에는 전체 train으로 mapping refit
  - 미매칭 값은 fallback mean으로 처리
  - validation split → mapping 학습 경로는 금지
- Visual: `V10` — fold-safe target encoding 흐름도
- Notes: target encoding은 강력하지만 누수에 취약하다. 전체 train으로 mapping을 만든 뒤 validation에 적용하면 validation target 정보가 통계에 섞인다. 이 프로젝트에서는 각 fold의 train split만 사용해 mapping을 만들고, validation에는 적용만 해서 OOF 점수를 계산했다.

### 슬라이드 12 — UCI 외부 데이터 결합

- Layout: `diagram`
- Body:
  - UCI 원본 Abalone 4,177건 추가
  - Kaggle 데이터와 동일한 v2 feature schema 적용
  - 각 fold의 train split에만 결합
  - validation은 Kaggle fold validation으로만 계산
  - OOF RMSLE: `0.147607`
- Visual: `V11` — fold train + UCI external 결합 다이어그램
- Notes: 외부 데이터는 validation에 넣지 않고 fold train에만 추가했다. 개선폭은 0.00016 정도로 작지만, 데이터 보강 효과가 OOF 기준에서 확인됐다. 최종 모델의 학습 데이터 구성을 설명하는 데 중요한 슬라이드다.

### 슬라이드 13 — Boosting Ensemble

- Layout: `chart`
- Body:
  - 단일 모델: LGBM `0.147607`, XGBoost `0.147903`, CatBoost `0.147773`
  - 앙상블: 1:1:1 단순 평균
  - `equal_linear` OOF RMSLE: **0.147529**
  - `equal_log` OOF RMSLE: `0.147530`
  - 최종 후보: `submission_v5_boosting_ensemble.csv`
- Visual: `V12` — 단일 모델 vs 앙상블 막대 차트
- Notes: 단일 best는 LGBM이지만, 세 모델의 오차 패턴이 완전히 같지는 않기 때문에 단순 평균이 더 낮은 OOF를 만들었다. log 평균도 거의 같은 점수였으므로, 복잡한 가중치 튜닝보다 모델 다양성 자체가 의미 있었다고 해석한다.

### 슬라이드 14 — 전체 성능 개선 흐름

- Layout: `chart`
- Body:
  - Baseline: `0.149919`
  - HGBR tuning: `0.149828`
  - Target Encoding: `0.147765`
  - External Data: `0.147607`
  - Ensemble: **`0.147529`**
- Visual: `V1` — 전체 성능 개선 라인 차트
- Notes: 전체 여정을 한 장으로 회수한다. 가장 큰 낙폭은 HGBR tuning 이후 target encoding을 적용한 구간이다. 최종 메시지는 "피처와 검증 설계가 모델 선택만큼 중요했다"로 정리한다.

### 슬라이드 15 — 결론과 다음 작업

- Layout: `closing`
- Body:
  - 결론 1: RMSLE에 맞춘 log target + clipping 유지
  - 결론 2: OOF-safe 검증으로 작은 개선을 해석
  - 결론 3: target encoding이 핵심 전환점
  - 다음 작업: public/private score 기록, TE smoothing, Optuna, tail calibration
  - Q&A
- Visual: `none`
- Notes: 세 가지 핵심 메시지를 다시 말하고 마무리한다. 아직 Kaggle public/private 점수가 비어 있다면 최종 제출 후 score log를 README 또는 별도 표로 관리하겠다고 말한다. 다음 개선은 leaderboard 과적합을 피하면서 CV와 public/private gap을 함께 보는 방향으로 진행한다.

---

## 3. 시각 자료 명세

### V1 — 전체 성능 개선 라인 차트

- x축: Baseline → HGBR tuning → Target Encoding → External Data → Ensemble
- y축: OOF RMSLE
- 값: `0.149919`, `0.149828`, `0.147765`, `0.147607`, `0.147529`
- 강조: HGBR tuning → Target Encoding 구간을 강조색으로 표시
- 캡션: "가장 큰 개선은 target encoding 적용 구간에서 발생"

### V2 — RMSLE 민감도 모식도

- 비교: 실제값 2/예측값 4 vs 실제값 20/예측값 22
- 메시지: 같은 절대 오차라도 작은 실제값에서 RMSLE penalty가 더 큼
- 하단 문구: "`log1p(target)` 학습과 음수 clipping이 기본 전제"

### V3 — 프로젝트 디렉토리 기반 재현 흐름도

- 흐름: `data/raw` → `notebooks` → `data/proceed` → `submissions` → `docs`
- 보조 라벨:
  - `raw`: 원본 보존
  - `proceed`: 재사용 가능한 feature/prediction 산출물
  - `submissions`: 제출 후보
  - `docs`: 발표/보고 자료

### V4 — EDA 요약 시각화

- 좌측: `Rings` 분포 히스토그램, 오른쪽 꼬리 강조
- 우측: 주요 신호 피처 리스트
  - `Shell_weight`
  - `Height`
  - `Diameter`
  - `Length`
  - `Whole_weight`
- 캡션: "EDA는 log target, ratio feature, target encoding 후보를 만들었다."

### V5 — 검증 설계 다이어그램

- `train`을 5개 fold로 나누고, 각 fold에서 train split과 validation split을 구분
- `test`는 별도 박스로 분리
- 라벨:
  - "model selection: train folds only"
  - "test: drift check + final prediction"

### V6 — 실험 파이프라인 흐름도

- 박스 8개:
  - `01_eda`
  - `02_feature_engineering`
  - `03_baseline`
  - `04_modeling`
  - `05_submission`
  - `06_target_encoding_lgbm`
  - `07_external_data_lgbm`
  - `08_boosting_ensemble`
- `01`~`05`: 기본 파이프라인
- `06`~`08`: 최종 성능 개선 라인

### V7 — Baseline 모델 가로 막대 차트

- 데이터: T2
- 정렬: RMSLE가 큰 순서에서 작은 순서
- 강조: `HistGradientBoosting + log target`
- 캡션: "best baseline = HGBR + log target, OOF 0.149919"

### V8 — HGBR modeling 후보 비교 막대

- 데이터: T3
- x축 범위: `0.1496` ~ `0.1502`
- 강조: `v2_hgb_log_leaf45_clip`
- 캡션: "v2 + HGBR tuning 개선폭은 약 0.0001"

### V9 — Target Encoding 전후 성능 비교

- 막대 3개:
  - `v2_hgb_log_leaf45_clip`: `0.149828`
  - `v2_lgbm_log`: `0.149206`
  - `v2_te_lgbm_log`: `0.147765`
- 강조: `v2_te_lgbm_log`
- 캡션: "target encoding 적용 후 0.147대 진입"

### V10 — Fold-safe Target Encoding 흐름도

- fold train split → target mapping 학습
- target mapping → validation split 적용
- validation split → target mapping 학습 경로에는 빨간 X
- test 예측: 전체 train → mapping refit → test 적용
- 캡션: "validation target 정보가 mapping에 섞이지 않도록 분리"

### V11 — External Data 결합 다이어그램

- UCI external 4,177건을 각 fold train split에만 결합
- validation split에는 결합하지 않음
- validation score는 Kaggle fold validation에서만 계산
- 캡션: "외부 데이터는 학습 보강용, validation 대체용이 아님"

### V12 — 단일 모델 vs 앙상블 막대 차트

- 데이터: T5
- y축: LGBM, XGBoost, CatBoost, `equal_linear`, `equal_log`
- 강조: `equal_linear`
- 캡션: "1:1:1 단순 평균이 최종 OOF 0.147529"

---

## 4. 재사용 가능한 데이터

### T1 — 전체 성능 개선 흐름

| 단계 | 실험 | OOF RMSLE | 해석 |
| --- | --- | ---: | --- |
| Baseline | HGBR + log target | 0.149919 | 첫 기준선 |
| HGBR tuning | `v2_hgb_log_leaf45_clip` | 0.149828 | 작은 개선 |
| Target Encoding | `v2_te_lgbm_log` | 0.147765 | 핵심 개선 |
| External Data | `v2_te_lgbm_log_external` | 0.147607 | 소폭 개선 |
| Ensemble | `equal_linear` | **0.147529** | 최종 후보 |

### T2 — Baseline 모델 비교

| 모델 | CV RMSLE mean | CV RMSLE std |
| --- | ---: | ---: |
| HistGradientBoosting + log target | 0.149917 | 0.000889 |
| HistGradientBoosting | 0.150623 | 0.000923 |
| RandomForest | 0.151209 | 0.000756 |
| Ridge + log target | 0.158517 | 0.001720 |
| Ridge | 0.159990 | 0.001744 |
| Dummy median | 0.286738 | 0.001278 |

best baseline OOF RMSLE: **0.149919**

### T3 — HGBR Modeling 후보

| 실험 | Feature set | CV RMSLE mean | CV RMSLE std |
| --- | --- | ---: | ---: |
| `v2_hgb_log_leaf45_clip` | v2 | 0.149826 | 0.000907 |
| `v2_hgb_log_leaf45` | v2 | 0.149875 | 0.000856 |
| `v1_hgb_log_baseline` | v1 | 0.149917 | 0.000889 |
| `v2_hgb_log_lr0035_iter650` | v2 | 0.149947 | 0.000891 |
| `v2_hgb_log_clip_005_995` | v2 | 0.149987 | 0.000932 |
| `v2_hgb_log_baseline` | v2 | 0.150008 | 0.000911 |

best modeling OOF RMSLE: **0.149828**

### T4 — 순위 개선 실험

| 실험 | 설명 | OOF RMSLE | 제출 파일 |
| --- | --- | ---: | --- |
| `v2_hgb_log_leaf45_clip` | 기존 HGBR best | 0.149828 | `submissions/submission_v2_hgb_log_leaf45_clip.csv` |
| `v2_lgbm_log` | v2 + LGBM log target | 0.149206 | - |
| `v2_te_lgbm_log` | v2 + target encoding + LGBM | 0.147765 | `submissions/submission_v3_lgbm_target_encoding.csv` |
| `v2_te_lgbm_log_external` | UCI 원본 4,177건 fold train 결합 | 0.147607 | `submissions/submission_v4_lgbm_te_external.csv` |
| `equal_linear` | LGBM/XGBoost/CatBoost 1:1:1 평균 | 0.147529 | `submissions/submission_v5_boosting_ensemble.csv` |

### T5 — Boosting Ensemble

| 모델 또는 앙상블 | OOF RMSLE |
| --- | ---: |
| LGBM | 0.147607 |
| XGBoost | 0.147903 |
| CatBoost | 0.147773 |
| 1:1:1 평균 (`equal_linear`) | **0.147529** |
| 1:1:1 log 평균 (`equal_log`) | 0.147530 |

### T6 — Feature Dataset

| 파일 | shape | 설명 |
| --- | ---: | --- |
| `data/proceed/train_fe_v1.csv` | 90,615 x 19 | `id` + 17개 피처 + `Rings` |
| `data/proceed/test_fe_v1.csv` | 60,411 x 18 | `id` + 17개 피처 |
| `data/proceed/train_fe_v2.csv` | 90,615 x 41 | `id` + 39개 피처 + `Rings` |
| `data/proceed/test_fe_v2.csv` | 60,411 x 40 | `id` + 39개 피처 |
| `data/proceed/external_uci_fe_v2.csv` | 4,177 x 41 | UCI 원본 데이터에 v2 피처셋 적용 |

### T7 — Target Encoding 피처

| 피처 | 의미 |
| --- | --- |
| `Sweight_avg` | `Shell_weight`별 평균 `Rings` |
| `Height_avg` | clipping된 `Height`별 평균 `Rings` |
| `Weight_avg` | `Whole_weight`별 평균 `Rings` |
| `Viscera_weight_avg` | `Viscera_weight`별 평균 `Rings` |
| `Sex_avg` | `Sex`별 평균 `Rings` |

### T8 — 제출 후보

| 파일 | rows | columns | 설명 |
| --- | ---: | --- | --- |
| `submissions/submission_v2_hgb_log_leaf45_clip.csv` | 60,411 | `id`, `Rings` | 기존 HGBR 제출 파일 |
| `submissions/submission_v3_lgbm_target_encoding.csv` | 60,411 | `id`, `Rings` | target encoding + LGBM |
| `submissions/submission_v4_lgbm_te_external.csv` | 60,411 | `id`, `Rings` | target encoding + UCI 원본 데이터 + LGBM |
| `submissions/submission_v5_boosting_ensemble.csv` | 60,411 | `id`, `Rings` | LGBM/XGBoost/CatBoost 1:1:1 앙상블 |

---

## 5. 발표자 메모

### 5.1. 시간 배분

| 구간 | 슬라이드 | 권장 시간 |
| --- | --- | ---: |
| 결과와 문제 정의 | 1~3 | 2분 |
| 데이터, 검증, 파이프라인 | 4~7 | 4분 |
| baseline과 HGBR 한계 | 8~9 | 2분 |
| target encoding | 10~11 | 3분 |
| 외부 데이터와 앙상블 | 12~13 | 2분 |
| 종합 결론 | 14~15 | 2분 |

### 5.2. Q&A 대비 포인트

- **왜 test를 모델 선택에 쓰지 않았는가?**
  - test는 최종 예측 대상이므로 drift 확인과 최종 prediction에만 사용했다.
- **target encoding이 누수 아닌가?**
  - fold train split에서만 mapping을 학습하고 validation에는 적용만 했다.
- **왜 HGBR에서 LGBM/XGB/CatBoost로 바꿨는가?**
  - HGBR tuning의 개선폭이 작았고, tabular boosting 계열의 다양성과 target encoding 조합이 OOF에서 더 좋았다.
- **왜 단순 평균인가?**
  - `equal_linear`와 `equal_log`가 거의 같은 점수를 냈고, 복잡한 weight tuning 없이도 단일 모델보다 낮은 OOF를 얻었다.
- **현재 한계는 무엇인가?**
  - Kaggle public/private score가 아직 반영되지 않았다면 OOF 기준 결론이며, 제출 후 score log로 검증해야 한다.

