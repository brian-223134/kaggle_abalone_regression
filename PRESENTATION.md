# Abalone Regression 발표자료 — Claude 에이전트용 PPT 생성 명세서

> 이 문서는 Claude 에이전트가 **PPT 초안(슬라이드 덱)** 을 생성할 수 있도록 작성된 단일 입력 명세서입니다.
> 본문은 (1) 에이전트 가이드 → (2) 발표 컨텍스트 → (3) 슬라이드 블루프린트 → (4) 시각 자료 명세 → (5) 재사용 가능한 데이터 → (6) 용어 사전 → (7) 참고 서술 순서로 구성됩니다.
> 에이전트는 슬라이드를 만들 때 (3)을 1:1 매핑 기준으로 사용하고, (4)와 (5)에서 차트·표를 가져오며, (7)은 발표자 노트나 행간을 채울 때 참고하세요.

---

## 0. 에이전트 사용 가이드 (먼저 읽기)

### 0.1. 출력 목표

- 산출물: PowerPoint 초안 1개. 각 슬라이드는 `# 슬라이드 N — 제목` 형식의 markdown으로 먼저 출력하고, 각 슬라이드 아래에 다음 4개 블록을 둔다.
  - `Layout:` 슬라이드 레이아웃 키워드 (예: `title`, `agenda`, `bullets`, `table`, `chart`, `diagram`, `closing`)
  - `Body:` 슬라이드에 보일 텍스트 (제목 외, 불릿/표/캡션). 한 슬라이드당 5줄 이내, 핵심 키워드 위주.
  - `Visual:` 슬라이드에 들어갈 시각 자료의 명세. 4절의 항목 이름을 그대로 인용. 시각 자료가 없으면 `none`.
  - `Notes:` 발표자 노트 3~5문장. 청중이 그 슬라이드를 보면서 들어야 할 말.

### 0.2. 작성 원칙

- **본문은 한국어**, 데이터/표 헤더와 모델명은 원문(영문) 유지.
- 한 슬라이드에 들어가는 텍스트는 짧게. 긴 설명은 `Notes`로 빼라.
- 수치는 5절 표에서 그대로 인용한다. 추정/반올림 금지.
- 차트/다이어그램은 4절 명세 그대로의 데이터·축·범례를 따른다. 새로운 시각 자료를 만들지 말라.
- 슬라이드 수는 **15장 ± 1장**을 목표로 한다. 3절의 블루프린트가 이미 15장 기준이다.
- 회사·발표자 이름·일자는 `[발표자 이름]`, `[발표 일자]` 같은 placeholder를 둔다(아래 0.3 참고).

### 0.3. 미정 입력값(placeholder)

다음 항목은 발표자가 채울 자리이므로 placeholder로만 둔다.

- 발표자 이름: `[발표자]`
- 발표 일자: `[YYYY-MM-DD]`
- 소속/조직: `[조직]`
- Kaggle public/private leaderboard 점수: `[public score TBD]`, `[private score TBD]` (아직 미제출)

### 0.4. 자료 출처(이 명세서로 충분, 추가 파일 읽지 않아도 됨)

이 명세서는 아래 파일들에서 정리된 정보를 모두 포함하고 있다. 에이전트는 별도 파일을 읽지 않고 이 문서만으로 슬라이드를 작성할 수 있다.

- `README.md` — 프로젝트 전체 README
- `notebooks/README.md` — 실험 단계별 노트북 README
- `data/README.md` — 데이터셋 설명
- `notebooks/01_eda.ipynb` ~ `notebooks/08_boosting_ensemble.py` — 실제 실험 코드(슬라이드에는 파일명만 인용)

---

## 1. 발표 컨텍스트

| 항목 | 값 |
| --- | --- |
| 발표 제목 | Abalone Regression — RMSLE 최적화 실험 정리 |
| 한 줄 요약 | Kaggle Playground S4E4 Abalone에서 OOF RMSLE 0.149828 → 0.147529까지 단계별로 개선한 실험 기록 |
| 청중 | 머신러닝/데이터 분석에 어느 정도 익숙한 동료 (트리 부스팅, 교차검증은 알고 있다고 가정) |
| 발표 분량 | 15분 발표 + 5분 Q&A |
| 톤 | 정량적·실험 기록 위주. "왜 이 단계를 거쳤는가"를 강조. 마케팅톤 금지. |
| 시각 톤 | 단색 + 강조색 1개. 차트는 라인/막대 위주. 픽토그램·일러스트 지양. |

### 핵심 메시지 (제목 슬라이드와 마무리 슬라이드에 동일하게 등장)

1. **RMSLE는 비율 오차에 민감한 지표다.** → 항상 `log1p(target)` 학습과 음수 clipping을 같이 가져갔다.
2. **모델 성능 개선은 한 번에 하나씩 비교해야 해석된다.** → fold·metric을 고정한 baseline 위에 변화 한 가지씩 얹어 비교했다.
3. **가장 큰 점수 개선은 모델 교체가 아니라 target encoding(누수 방지 포함)이었다.** → HGBR 튜닝(0.149828) 대비 LGBM + target encoding(0.147765)이 결정적 전환점이었다.

---

## 2. 한눈에 보는 결과

| 단계 | 실험명 | OOF RMSLE | 비고 |
| --- | --- | ---: | --- |
| Baseline | `HistGradientBoosting + log target` | 0.149919 | v1 피처 |
| Modeling | `v2_hgb_log_leaf45_clip` | 0.149828 | v2 피처, HGBR 튜닝 |
| Target Encoding | `v2_te_lgbm_log` | 0.147765 | 핵심 개선점 |
| External Data | `v2_te_lgbm_log_external` | 0.147607 | UCI 원본 4,177건 결합 |
| Ensemble | `equal_linear` (LGBM·XGB·CatBoost 평균) | **0.147529** | 최종 후보 |

최종 제출 후보 파일: `submissions/submission_v5_boosting_ensemble.csv`.

---

## 3. 슬라이드 블루프린트 (15장 기준)

> 각 슬라이드는 0.1절의 출력 포맷에 맞춰 `Layout / Body / Visual / Notes`로 변환하라.
> 표 안의 `Layout`, `Visual` 키워드와 4절의 `V#` 식별자, 5절의 `T#` 식별자가 그대로 매핑된다.

### 슬라이드 1 — 제목

- Layout: `title`
- Body:
  - 제목: `Abalone Regression — RMSLE 최적화 실험 정리`
  - 부제: `OOF RMSLE 0.149828 → 0.147529, 단계별 비교`
  - 발표자/일자: `[발표자] · [YYYY-MM-DD]`
- Visual: `none`
- Notes: 발표 인사. 이 발표가 단순히 점수 한 개를 만든 기록이 아니라, RMSLE 회귀 문제를 어떤 순서로 풀었는지에 대한 설명임을 1문장으로 명시.

### 슬라이드 2 — 목차

- Layout: `agenda`
- Body: 다음 7개 섹션을 순서대로 나열
  1. 문제 정의와 평가지표
  2. 데이터 탐색(EDA) 핵심 결론
  3. 실험 파이프라인(노트북 분리 이유)
  4. Baseline → Modeling 진행
  5. 전환점: target encoding
  6. 외부 데이터와 앙상블
  7. 결과·강조 포인트·다음 단계
- Visual: `none`
- Notes: 청중에게 어떤 흐름으로 따라오면 되는지 안내. "결과만 보면 0.149 → 0.147 변화지만, 그 사이의 의사결정을 같이 보겠다"고 강조.

### 슬라이드 3 — 문제 정의

- Layout: `bullets`
- Body:
  - 데이터: Kaggle Playground Series S4E4 Abalone (UCI 원본 기반 합성)
  - 입력: 길이/지름/높이, 4가지 무게, `Sex` (M/F/I)
  - 타깃: `Rings` (Age = Rings + 1.5)
  - 행 수: train 90,615 / test 60,411
  - 평가지표: RMSLE
- Visual: `V1` — 데이터셋 한눈에 (행 수, 컬럼, 타깃 위치 표)
- Notes: `Whole weight.1`, `Whole weight.2`가 사실상 Shucked/Viscera라는 점만 빠르게 설명. EDA 단계에서 `Shucked_weight`, `Viscera_weight`로 rename했다고 언급.

### 슬라이드 4 — 평가지표 RMSLE

- Layout: `bullets`
- Body:
  - 정의: `RMSLE = sqrt(mean((log1p(pred) - log1p(actual))^2))`
  - 특성 1: 비율 오차에 민감 → 작은 `Rings`에서의 실수가 점수에 크게 반영
  - 특성 2: 음수 예측이 들어가면 정의되지 않음 → 항상 `np.clip(pred, 0, None)`
  - 적용: 모든 모델은 `log1p(Rings)`로 학습 후 `expm1`로 복원
- Visual: `V2` — RMSLE vs RMSE 민감도 모식도
- Notes: "절대 오차가 같아도 작은 값에서 더 크게 깎인다"는 점이 왜 log target 학습으로 이어지는지를 1분 설명. 이 슬라이드에서 청중이 "왜 굳이 log를 거치는지" 이해하지 못하면 이후 결과 비교가 의미 없음.

### 슬라이드 5 — EDA 핵심 결론

- Layout: `bullets`
- Body:
  - 결측치 없음, 중복 없음 → 정제 비용 낮음
  - `Rings`는 5~15에 집중되고 오른쪽 꼬리가 김 → log target 후보
  - `Shell_weight`, `Height`, `Diameter`, `Length`, `Whole_weight`가 강한 양의 상관
  - `Sex`별 분포 차이 → one-hot encoding 기본
  - `Height == 0`, 비율 피처 극단값 → indicator/median fill/quantile clipping을 CV로 비교
  - train/test 입력 drift 작음 → test는 drift 점검용으로만 사용
- Visual: `V3` — `Rings` 분포 히스토그램 (오른쪽 꼬리 강조)
- Notes: 5분짜리 EDA 결과를 8줄로 압축한 슬라이드. 데이터가 깨끗했기 때문에 시간을 모델링·검증 설계에 더 쓸 수 있었다고 결론.

### 슬라이드 6 — 노트북 분리 구조

- Layout: `table`
- Body: T1 표 (5절). 8개 노트북·스크립트가 각각 어떤 질문에 답하는지.
- Visual: `V4` — 파이프라인 흐름도 (`01_eda` → `02_fe` → `03_baseline` → `04_modeling` → `05_submission` → `06_te` → `07_external` → `08_ensemble`)
- Notes: "왜 한 노트북에 다 안 넣었는가" 설명. 단계별로 질문이 다르고, 실패한 실험을 되돌리기 쉬워야 한다는 점을 강조.

### 슬라이드 7 — Baseline 비교

- Layout: `table`
- Body: T2 표 (5절). 6개 모델의 CV RMSLE mean / std.
- Visual: `V5` — Baseline 모델 가로 막대 차트 (CV RMSLE mean 기준 정렬)
- Notes: best는 `HistGradientBoosting + log target` (OOF 0.149919). Ridge보다 부스팅이 약 0.01 점 낮다. 이 시점에서 "tabular는 부스팅 + log target이 출발점"이라는 결정을 내렸다.

### 슬라이드 8 — v2 피처 + HGBR 튜닝

- Layout: `bullets`
- Body:
  - v1: 17개 피처 (원본·one-hot·물리 비율)
  - v2: 39개 피처 (shape ratio, component balance, log 피처 추가)
  - 튜닝 후보: `leaf45`, `clip`, `lr0035_iter650` 등
  - best: `v2_hgb_log_leaf45_clip` — OOF RMSLE 0.149828
  - 한계: baseline 0.149919 대비 개선폭이 작음 (≈ -0.0001)
- Visual: `V6` — Modeling 후보 비교 막대 (T3 표 데이터)
- Notes: "ratio·log 피처와 HGBR 튜닝만으로는 한계"라는 결론을 분명히 한다. 다음 슬라이드에서 방향을 바꾸게 된 트리거.

### 슬라이드 9 — 전환점: Target Encoding

- Layout: `diagram`
- Body:
  - 추가 피처(5개): `Sweight_avg`, `Height_avg`, `Weight_avg`, `Viscera_weight_avg`, `Sex_avg`
  - 누수 방지: 각 fold의 train split에서만 mapping 학습 → validation에는 적용만
  - test 예측 시에만 전체 train으로 mapping 재학습
  - 결과: OOF RMSLE 0.147765 (HGBR best 대비 -0.002)
- Visual: `V7` — fold-safe target encoding 흐름도
- Notes: 이 슬라이드가 발표의 클라이맥스. "왜 모델 교체보다 피처 변경이 더 컸는가"를 설명. 누수 방지 부분에서 "전체 train으로 mapping을 만들면 validation 점수가 더 좋아 보이지만 leaderboard에서 망가진다"는 점을 강조.

### 슬라이드 10 — UCI 외부 데이터 결합

- Layout: `bullets`
- Body:
  - Kaggle 데이터는 UCI 원본 기반 합성 → 원본 4,177건을 학습 보강
  - validation은 Kaggle fold validation 행에 대해서만 계산
  - UCI 데이터는 해당 fold의 train split에만 결합 (UCI는 절대 validation에 들어가지 않음)
  - UCI 데이터에도 v2 피처 스키마 동일 적용
  - 결과: OOF RMSLE 0.147607 (-0.00016)
- Visual: `V8` — fold + external data 결합 다이어그램
- Notes: 개선폭은 작지만 재현 가능한 방식으로 외부 데이터를 더했다는 점이 중요. Public/Private 차이를 보고 다음에 비중 조정 가능.

### 슬라이드 11 — Boosting Ensemble

- Layout: `table`
- Body: T4 표 (5절). LGBM/XGB/CatBoost 단일 점수 + 평균 앙상블 점수.
- Visual: `V9` — 단일 vs 앙상블 막대 비교
- Notes: 단일 best는 LGBM이지만, 세 모델의 오차 패턴이 달라 1:1:1 평균이 더 안정적. log 평균(`equal_log`)도 0.147530으로 거의 동일했다는 점을 부언해 "튜닝보다 다양성"이라는 메시지를 보강.

### 슬라이드 12 — 성능 개선 흐름 한눈에

- Layout: `chart`
- Body: 슬라이드 본문은 캡션만. 본문 텍스트는 "단계마다 어떤 의사결정이 있었는가"를 1줄씩, 5줄로 요약 (T5 표 기반).
- Visual: `V10` — OOF RMSLE 진행 라인 차트 (x: 단계, y: OOF RMSLE, 5점)
- Notes: 가장 큰 낙폭은 Modeling → Target Encoding 구간. 청중에게 "여기가 발표의 결정적 구간이었다"고 못박는다.

### 슬라이드 13 — 발표 시 강조 포인트

- Layout: `bullets`
- Body:
  - `id`는 식별자이므로 학습 피처 제외, 제출 파일 생성 시에만 사용
  - test set은 drift 확인 + 최종 예측에만 사용 (모델 선택에는 사용 안 함)
  - target encoding은 OOF-safe 구현 (fold train에서만 mapping 학습)
  - CV와 public score가 항상 일치하지 않을 수 있음 → 제출 로그를 별도 관리
  - 모델 교체(HGBR → LGBM/XGB/CatBoost)는 상위권 풀이의 실증 결과 + 표현력 차이 때문
- Visual: `none`
- Notes: 이 슬라이드는 "이 프로젝트가 검증·재현성 측면에서 어떤 원칙을 지켰는가"를 정리하는 슬라이드. Q&A에서 가장 많이 받을 만한 질문을 미리 답하는 의도.

### 슬라이드 14 — 다음 개선 후보

- Layout: `bullets`
- Body:
  - Optuna로 LGBM/XGB/CatBoost 하이퍼파라미터 튜닝
  - target encoding smoothing 적용
  - `Shell_weight`/`Height`/`Whole_weight` rounding 후 target encoding 비교
  - tail 구간(`Rings >= 16`) 전용 진단·calibration
  - Kaggle public/private 점수 로그를 README/별도 표에 기록
- Visual: `none`
- Notes: 현재 OOF는 0.147529. public/private 점수가 들어오면 어느 방향이 leaderboard와 일치하는지 보고 우선순위를 결정하겠다고 마무리.

### 슬라이드 15 — 마무리·Q&A

- Layout: `closing`
- Body:
  - 핵심 메시지 3개 재확인 (1절 핵심 메시지)
  - 최종 제출 후보: `submissions/submission_v5_boosting_ensemble.csv`
  - 코드/데이터 위치: `Desktop/abalone-regression/`
  - "Q&A"
- Visual: `none`
- Notes: 짧게 마무리. 청중 질문 받기.

---

## 4. 시각 자료 명세

> 각 항목은 슬라이드 블루프린트의 `Visual: V#` 식별자와 매핑된다.
> 차트는 단일 강조색 1개 + 회색 톤 권장. 모든 수치는 5절에서 가져온다.

### V1 — 데이터셋 한눈에 (표 형식 인포그래픽)

- 좌측: train 90,615행 / test 60,411행 / 외부(UCI) 4,177행
- 우측: 컬럼 카테고리 4개 — 식별자(`id`), 범주형(`Sex`), 수치 8개(길이·지름·높이·무게 4개), 타깃(`Rings`)
- 캡션: "타깃은 `Rings`. Age = Rings + 1.5."

### V2 — RMSLE 민감도 모식도

- 단순한 도식 1장. 좌측에 "실제값=2, 예측값=4"와 "실제값=20, 예측값=22"를 두고, RMSE는 같지만 RMSLE는 전자가 훨씬 크게 패널티된다는 것을 화살표·숫자로 표현.
- 캡션: "같은 절대 오차여도 RMSLE는 작은 값에서 더 크게 깎인다."

### V3 — `Rings` 분포 히스토그램

- x축: `Rings` 값 (1~25 정도 범위)
- y축: 빈도(count)
- 5~15 구간에 막대 집중, 오른쪽 꼬리 회색으로 약하게 강조
- 캡션: "`Rings`는 오른쪽 꼬리가 있는 count형 타깃."

### V4 — 노트북 파이프라인 흐름도

- 좌→우 단방향 화살표 다이어그램
- 박스 8개: `01_eda` → `02_feature_engineering` → `03_baseline` → `04_modeling` → `05_submission` (HGBR 라인) ↘
  `06_target_encoding_lgbm` → `07_external_data_lgbm` → `08_boosting_ensemble` (개선 라인)
- 두 라인이 위/아래로 분리돼 있고, "기본 파이프라인"과 "상위권 풀이 기반 개선" 라벨로 구분
- 캡션 없음(라벨로 충분)

### V5 — Baseline 모델 가로 막대 차트

- y축(카테고리): Dummy median, Ridge, Ridge + log, RandomForest, HistGradientBoosting, **HistGradientBoosting + log**(강조색)
- x축(값): CV RMSLE mean (T2 표)
- 정렬: 값이 큰 순 → 작은 순 (Dummy가 위, HGBR + log가 아래)
- 캡션: "best baseline = HistGradientBoosting + log target (CV 0.149917)."

### V6 — Modeling 후보 비교 막대

- y축: T3 표의 6개 실험명
- x축: CV RMSLE mean
- best (`v2_hgb_log_leaf45_clip`)만 강조색
- 차이가 매우 작으므로 x축 zoom: `[0.1496, 0.1502]` 범위
- 캡션: "v2 + HGBR 튜닝의 개선폭은 0.0001 수준 — 다음 단계 트리거."

### V7 — Fold-safe Target Encoding 흐름도

- 5-fold 중 하나의 fold를 단순화해 그린다.
- 박스 흐름:
  1. `Train data` → 5-fold split
  2. 한 fold 안에서 `train split` 박스에서만 화살표가 `target mapping` 박스로 이어짐
  3. `target mapping`이 `validation split` 박스에 적용 (mapping은 학습되지 않고 적용만)
  4. 최종 test 예측 시에는 별도 박스: `전체 train` → `target mapping (refit)` → `test`
- 빨간 X 표시로 "validation split → target mapping" 화살표가 금지됨을 명시
- 캡션: "validation split은 mapping 학습에 절대 들어가지 않는다."

### V8 — Fold + External Data 결합 다이어그램

- V7과 같은 fold 도식 위에 `UCI external (4,177)` 박스를 추가
- 화살표: `UCI external` → `train split (fold k)` (validation split으로는 절대 안 감 → 빨간 X)
- 캡션: "UCI 4,177건은 fold train에만 결합. validation 계산은 Kaggle fold validation에서만."

### V9 — 단일 모델 vs 앙상블 막대

- y축: LGBM, XGBoost, CatBoost, **equal_linear**(강조색)
- x축: OOF RMSLE
- x축 zoom: `[0.1473, 0.1480]`
- 캡션: "단일 best는 LGBM이지만 1:1:1 평균이 OOF 0.147529로 더 낮음."

### V10 — OOF RMSLE 진행 라인 차트

- x축(범주): Baseline → Modeling → Target Encoding → External → Ensemble
- y축(값): OOF RMSLE
- 점 5개, 라인으로 연결, 각 점 위에 값(예: 0.149919) 라벨
- y축 zoom: `[0.1470, 0.1505]` 권장
- Modeling → Target Encoding 구간을 강조색 라인으로 (다른 구간은 회색)
- 캡션: "가장 큰 낙폭: Modeling → Target Encoding (-0.002)."

---

## 5. 재사용 가능한 데이터 (표·수치)

> 모든 수치는 이 절에서 인용한다. 슬라이드에 표를 넣을 때는 이 절을 그대로 복사하라.

### T1 — 노트북 분리 구조

| 파일 | 담당 질문 | 역할 |
| --- | --- | --- |
| `01_eda.ipynb` | 데이터가 어떤 형태인가? | 품질, 분포, 상관관계, 이상치, train/test drift 확인 |
| `02_feature_engineering.ipynb` | 어떤 정보를 학습 가능한 숫자로 만들 것인가? | v1 피처셋 생성 |
| `03_baseline.ipynb` | 개선 기준선은 어디인가? | 단순 모델 비교, RMSLE baseline 확보 |
| `04_modeling.ipynb` | 기존 방향에서 얼마나 개선되는가? | v2 피처, HGBR tuning, OOF diagnostics |
| `05_submission.ipynb` | 제출 파일이 올바르게 만들어지는가? | 최종 HGBR 학습과 submission 검증 |
| `06_target_encoding_lgbm.py` | 상위권 접근의 핵심 피처가 먹히는가? | OOF-safe target encoding + LGBM |
| `07_external_data_lgbm.py` | 원본 UCI 데이터가 도움이 되는가? | 외부 데이터 결합 실험 |
| `08_boosting_ensemble.py` | 모델 다양성으로 더 낮출 수 있는가? | LGBM/XGBoost/CatBoost 앙상블 |

### T2 — Baseline 모델 비교 (`03_baseline.ipynb`)

| 모델 | CV RMSLE mean | CV RMSLE std |
| --- | ---: | ---: |
| HistGradientBoosting + log target | 0.149917 | 0.000889 |
| HistGradientBoosting | 0.150623 | 0.000923 |
| RandomForest | 0.151209 | 0.000756 |
| Ridge + log target | 0.158517 | 0.001720 |
| Ridge | 0.159990 | 0.001744 |
| Dummy median | 0.286738 | 0.001278 |

best baseline OOF RMSLE: **0.149919** (`HistGradientBoosting + log target`).

### T3 — Modeling 후보 비교 (`04_modeling.ipynb`)

| 실험 | Feature set | CV RMSLE mean | CV RMSLE std |
| --- | --- | ---: | ---: |
| `v2_hgb_log_leaf45_clip` | v2 | 0.149826 | 0.000907 |
| `v2_hgb_log_leaf45` | v2 | 0.149875 | 0.000856 |
| `v1_hgb_log_baseline` | v1 | 0.149917 | 0.000889 |
| `v2_hgb_log_lr0035_iter650` | v2 | 0.149947 | 0.000891 |
| `v2_hgb_log_clip_005_995` | v2 | 0.149987 | 0.000932 |
| `v2_hgb_log_baseline` | v2 | 0.150008 | 0.000911 |

best modeling OOF RMSLE: **0.149828** (`v2_hgb_log_leaf45_clip`).

### T4 — Boosting Ensemble (`08_boosting_ensemble.py`)

| 모델 또는 앙상블 | OOF RMSLE |
| --- | ---: |
| LGBM | 0.147607 |
| XGBoost | 0.147903 |
| CatBoost | 0.147773 |
| 1:1:1 평균 (`equal_linear`) | **0.147529** |
| 1:1:1 log 평균 (`equal_log`) | 0.147530 |

### T5 — 전체 성능 개선 흐름

| 단계 | 실험 | OOF RMSLE | 해석 |
| --- | --- | ---: | --- |
| Baseline | HGBR + log target | 0.149919 | 첫 기준선 |
| Modeling | v2 HGBR tuning | 0.149828 | 작은 개선 |
| Target Encoding | LGBM + target encoding | 0.147765 | 핵심 개선 |
| External Data | UCI 원본 추가 | 0.147607 | 소폭 개선 |
| Ensemble | LGBM/XGB/CatBoost 평균 | 0.147529 | 최종 후보 |

### T6 — 피처셋 버전

| 파일 | shape | 설명 |
| --- | ---: | --- |
| `data/proceed/train_fe_v1.csv` | 90,615 × 19 | `id` + 17개 피처 + `Rings` |
| `data/proceed/test_fe_v1.csv` | 60,411 × 18 | `id` + 17개 피처 |
| `data/proceed/train_fe_v2.csv` | 90,615 × 41 | `id` + 39개 피처 + `Rings` |
| `data/proceed/test_fe_v2.csv` | 60,411 × 40 | `id` + 39개 피처 |
| `data/proceed/external_uci_fe_v2.csv` | 4,177 × 41 | UCI 원본에 v2 피처셋 적용 |

v1 피처: 원본 수치 7개 + `Sex` one-hot 3개 + 파생 7개(`Volume`, `Density`, `Shucked_ratio`, `Viscera_ratio`, `Shell_ratio`, `Shell_to_shucked`, `Height_is_zero`).
v2 추가: shape ratio (`Area`, `Diameter_to_Length`, `Height_to_Length`, `Height_to_Diameter`), component balance (`Component_weight_sum`, `Component_sum_ratio`, `Residual_weight`, `Residual_weight_ratio`), log 피처 (`log1p_*`).

### T7 — Target Encoding 추가 피처

| 피처 | 의미 |
| --- | --- |
| `Sweight_avg` | `Shell_weight`별 평균 `Rings` |
| `Height_avg` | clipping된 `Height`별 평균 `Rings` |
| `Weight_avg` | `Whole_weight`별 평균 `Rings` |
| `Viscera_weight_avg` | `Viscera_weight`별 평균 `Rings` |
| `Sex_avg` | `Sex`별 평균 `Rings` |

구현 원칙: 각 fold train split에서만 mapping 학습, validation에는 적용만. `Height`는 fold train 기준 IQR alpha 2.0 fence로 clipping 후 `Height_avg` 산출. 결측은 `Sweight_avg` → `Viscera_weight_avg` → global mean 순으로 fallback.

### T8 — 제출 파일

| 파일 | rows | columns | 설명 |
| --- | ---: | --- | --- |
| `submissions/submission_v2_hgb_log_leaf45_clip.csv` | 60,411 | `id`, `Rings` | 기존 HGBR 제출 파일 |
| `submissions/submission_v3_lgbm_target_encoding.csv` | 60,411 | `id`, `Rings` | target encoding + LGBM |
| `submissions/submission_v4_lgbm_te_external.csv` | 60,411 | `id`, `Rings` | target encoding + UCI 원본 + LGBM |
| `submissions/submission_v5_boosting_ensemble.csv` | 60,411 | `id`, `Rings` | LGBM/XGBoost/CatBoost 1:1:1 앙상블 (최종 후보) |

---

## 6. 용어 사전 (필요 시 슬라이드 본문/노트에 활용)

- **RMSLE (Root Mean Squared Logarithmic Error)**: `sqrt(mean((log1p(pred) - log1p(actual))^2))`. 예측과 실제값의 비율 차이에 민감.
- **OOF (Out-Of-Fold) prediction**: 각 fold의 validation split에서 만든 예측을 모두 모은 것. 전체 train을 한 번씩 validation으로 본 결과. 단일 holdout보다 변동을 잘 잡아낸다.
- **CV RMSLE mean / std**: 5-fold 교차검증으로 측정한 RMSLE의 평균/표준편차. std가 작을수록 fold에 강건.
- **target encoding**: 범주형 또는 고-카디널리티 피처 값을 해당 그룹의 타깃 평균으로 치환하는 인코딩. 강력하지만 누수 위험이 큼.
- **fold-safe / OOF-safe**: validation split 정보가 학습 단계에 새지 않도록, fold마다 train split에서만 통계를 학습하는 방식.
- **HGBR (HistGradientBoostingRegressor)**: scikit-learn 내장 부스팅 모델. tabular baseline으로 자주 사용.
- **StratifiedKFold (회귀용)**: 회귀 타깃을 quantile bin으로 나눠 fold별 분포를 맞춘 뒤 stratify 하는 변형.
- **drift**: train과 test 입력 분포 차이. 작으면 train 내부에서만 검증해도 안전, 크면 별도 처리 필요.

---

## 7. 참고 서술 (slide 작성 시 행간/노트 보강용 — 직접 슬라이드에 옮기지 말 것)

> 아래 본문은 발표자가 작성한 원본 서술이다. 에이전트는 이 본문에서 슬라이드 본문을 자르지 말고, 발표자 노트의 톤과 의도를 잡는 데만 사용하라.

### 7.1. 발표 목적

이 프로젝트는 Kaggle Playground Series S4E4 Abalone 데이터셋에서 전복의 `Rings`를 예측하는 회귀 문제를 다룹니다. 평가지표는 RMSLE이며, 단순히 제출 파일 하나를 만드는 것보다 데이터 탐색, 피처 생성, 검증 설계, 모델 개선, 제출 파일 생성까지의 과정을 재현 가능한 실험 흐름으로 정리하는 데 초점을 두었습니다. 핵심 메시지는 (1) RMSLE에서는 타깃 분포와 작은 값 구간의 상대 오차를 먼저 이해해야 하고, (2) 모델 성능 개선은 같은 validation 기준에서 한 번에 하나씩 비교해야 해석할 수 있으며, (3) 초기 HGBR 단일 모델 OOF RMSLE 0.149828에서 target encoding, UCI 원본 데이터, boosting ensemble을 거쳐 0.147529까지 개선했다는 것입니다.

### 7.2. 노트북을 나눈 이유

처음부터 하나의 노트북에 EDA, 피처 생성, 모델링, 제출 파일 생성을 모두 넣으면 실험 결과를 추적하기 어렵습니다. 그래서 각 노트북은 하나의 질문에 답하도록 분리했습니다(T1 표). 이 구조 덕분에 특정 실험이 좋아졌을 때 무엇이 원인인지 추적할 수 있고, 성능이 나빠진 실험도 쉽게 되돌릴 수 있습니다.

### 7.3. 전체 흐름

프로젝트는 세 단계로 나뉩니다 — (1) 기본 파이프라인 구축, (2) 기존 모델링 방향의 한계 확인, (3) 상위권 풀이 기반 개선. 처음에는 EDA로 타깃 분포·이상치·drift를 확인하고, 물리적으로 의미 있는 ratio·log 피처를 만들어 HGBR을 튜닝했습니다. OOF RMSLE는 0.149828까지 내려갔지만, baseline 대비 개선폭이 fold 변동보다 뚜렷하지 않았습니다. 이후 Public 상위권 풀이를 참고해 방향을 바꿨고, 핵심은 `Shell_weight`, `Height`, `Whole_weight` 같은 강한 원본 피처를 target mean encoding으로 바꾸는 것이었습니다. target encoding은 validation 누수가 발생하기 쉬우므로, 각 fold의 train split에서만 mapping을 학습하고 validation split에는 그 mapping만 적용했습니다.

### 7.4. 단계별 핵심

- **EDA**: `Rings`는 오른쪽 꼬리를 가진 count형 타깃. RMSLE와 잘 맞도록 log target 학습 비교 결정. `Shell_weight`, `Height`, `Whole_weight` 계열의 중요성 확인.
- **Feature v1**: 컬럼 rename, `Sex` one-hot, `Volume`/`Density`, weight ratio, `Height_is_zero`. 성능 평가가 아니라 안정적인 첫 입력 데이터셋 확보.
- **Baseline**: 같은 fold·metric으로 6개 모델 비교. best는 `HistGradientBoosting + log target` (OOF 0.149919).
- **Modeling (v2 + HGBR)**: shape ratio·component balance·log 피처 추가. `v2_hgb_log_leaf45_clip` OOF 0.149828. 한계 확인.
- **Submission(HGBR)**: 모델 실험과 제출 파일 생성을 분리. 음수 clipping, `sample_submission.csv` 정합성 검증.
- **Target Encoding**: `Sweight_avg`, `Height_avg`, `Weight_avg`, `Viscera_weight_avg`, `Sex_avg` 추가. fold-safe 구현. OOF 0.147765 — 가장 큰 개선.
- **External Data**: UCI 원본 4,177건을 v2 스키마로 변환해 fold train에만 결합. OOF 0.147607.
- **Ensemble**: LGBM/XGB/CatBoost 1:1:1 평균. OOF 0.147529 — 최종 후보.

### 7.5. 강조 포인트

- `id`는 식별자이므로 학습 피처에서 제외했습니다.
- test set은 drift 확인과 최종 예측에만 사용했고, 모델 선택에는 사용하지 않았습니다.
- target encoding은 강력하지만 누수 위험이 크므로 OOF-safe 방식으로 구현했습니다.
- CV가 좋아지는 방향과 public score가 항상 일치하지 않을 수 있으므로 제출 로그를 별도로 관리해야 합니다.
- HGBR에서 LGBM/XGBoost/CatBoost로 넘어간 이유는 tabular boosting 모델의 표현력과 상위권 풀이의 실증 결과 때문입니다.

### 7.6. 다음 개선 후보

- Optuna로 LGBM/XGBoost/CatBoost 하이퍼파라미터 튜닝
- target encoding smoothing 적용
- `Shell_weight`, `Height`, `Whole_weight` rounding 후 target encoding 비교
- tail 구간(`Rings >= 16`) 전용 진단과 calibration
- Kaggle public/private score를 README 또는 별도 submission log에 기록
