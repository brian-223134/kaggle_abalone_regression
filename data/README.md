# Abalone Regression Dataset

Kaggle **Playground Series - Season 4, Episode 4** (2024.04) 대회 데이터셋입니다.
원본은 UCI Machine Learning Repository의 [Abalone Dataset](https://archive.ics.uci.edu/dataset/1/abalone)이며, Kaggle 대회용으로 딥러닝 모델로 합성·증강된 버전입니다. 따라서 분포 자체는 원본과 유사하지만 행 수가 훨씬 많습니다.

## 과제 (Task)

전복(Abalone)의 **나이(Age)** 를 예측하는 **회귀(Regression)** 문제입니다.

- 실제 나이는 껍질을 잘라 현미경으로 **고리(Rings)** 수를 세고 `Age = Rings + 1.5 (년)` 으로 계산합니다.
- 이 작업은 시간이 많이 들기 때문에, 측정이 더 쉬운 신체 치수·무게로부터 `Rings` 를 예측하는 것이 목표입니다.
- 타깃 컬럼은 `Rings` 입니다. (Age 변환은 후처리로 직접 수행)

### 평가 지표 (Evaluation Metric)

대회에서는 **RMSLE (Root Mean Squared Logarithmic Error)** 를 사용했습니다.

$$
\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\log(1+\hat{y}_i) - \log(1+y_i)\right)^2}
$$

작은 `Rings` 값에서의 상대 오차를 더 강하게 패널티하기 때문에, 어린 전복의 예측 정확도가 점수에 큰 영향을 줍니다.

## 파일 구성

| 파일 | 행 수 (header 제외) | 설명 |
| --- | ---: | --- |
| `train.csv` | 90,615 | 학습용 데이터. 입력 피처 + 타깃(`Rings`) 포함 |
| `test.csv` | 60,411 | 추론용 데이터. 타깃(`Rings`) 미포함 |
| `sample_submission.csv` | 60,411 | 제출 파일 양식 (`id`, `Rings` 두 컬럼) |

## 컬럼 설명

| 컬럼명 | 단위 | 자료형 | 설명 |
| --- | --- | --- | --- |
| `id` | - | int | 행 식별자 (모델 학습에 사용하지 않음) |
| `Sex` | - | category | 성별. `M`(수컷), `F`(암컷), `I`(infant, 미성숙 개체) |
| `Length` | mm (1/200 단위로 스케일됨) | float | 전복의 가장 긴 축 길이 |
| `Diameter` | mm (스케일됨) | float | `Length` 에 수직 방향 지름 |
| `Height` | mm (스케일됨) | float | 살을 포함한 껍질 높이 |
| `Whole weight` | g (1/200 단위로 스케일됨) | float | **전체 무게** (전복 통째) |
| `Whole weight.1` | g (스케일됨) | float | **Shucked weight** — 껍질을 제거한 살(meat)의 무게 |
| `Whole weight.2` | g (스케일됨) | float | **Viscera weight** — 피를 뺀 내장 무게 |
| `Shell weight` | g (스케일됨) | float | 건조 후 껍질 무게 |
| `Rings` | - | int | **타깃 변수**. 껍질 단면의 고리 수. `Age(년) = Rings + 1.5` |

> **주의 — 컬럼명이 헷갈리는 부분**
> 원본 UCI 데이터셋에서는 `Whole weight`, `Shucked weight`, `Viscera weight`, `Shell weight` 의 4가지 무게 컬럼으로 구성되어 있습니다. Kaggle 대회 CSV에서는 앞의 세 무게가 **`Whole weight`, `Whole weight.1`, `Whole weight.2`** 로 들어가 있는데, 이는 원본의
> - `Whole weight.1` ↔ **Shucked weight (살 무게)**
> - `Whole weight.2` ↔ **Viscera weight (내장 무게)**
> 에 각각 대응합니다. 이름만 보고 같은 변수 3개로 오해하지 않도록 주의해야 합니다. EDA 시 컬럼명을 의미 있는 이름으로 rename 해두는 것을 권장합니다.

## 데이터 특성 / EDA 포인트

- **결측치 없음** — 원본·합성 모두 결측치가 없는 깔끔한 표 형식 데이터.
- **`Sex` 는 범주형** — `M`, `F`, `I` 세 클래스. One-hot 또는 target encoding 등으로 처리.
- **무게 변수들 간 강한 다중공선성** — `Whole weight ≈ Shucked weight + Viscera weight + Shell weight + (수분손실)` 의 관계로 인해 높은 상관관계를 가짐. 트리 기반 모델은 영향이 적지만 선형 모델은 정규화/차원축소 필요.
- **`Height` 이상치** — 원본 UCI 데이터에 `Height = 0` 인 행이 존재. 합성본에도 비슷한 극단치가 일부 섞여 있을 수 있어 확인 권장.
- **타깃 분포의 비대칭성** — `Rings` 는 주로 5~15 구간에 집중되고 오른쪽 꼬리가 김. 평가 지표가 RMSLE인 점과 결합해 `log1p(Rings)` 변환 후 회귀하는 전략이 흔함.

## 제출 파일 형식

```csv
id,Rings
90615,10
90616,10
...
```

- `id` 는 `test.csv` 의 `id` 와 동일해야 함.
- `Rings` 는 정수가 아니어도 됨 (RMSLE 평가이므로 실수 예측값 그대로 제출).

## 참고 자료

- Kaggle 대회 페이지: <https://www.kaggle.com/competitions/playground-series-s4e4>
- 원본 UCI Abalone 데이터셋: <https://archive.ics.uci.edu/dataset/1/abalone>
- Nash, W.J. et al. (1994), *The Population Biology of Abalone in Tasmania* — 원본 데이터의 출처 논문
