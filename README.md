## 환경 세팅 방법
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## Directory 구성안
abalone-regression/
├── .venv/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── features.py
│   └── train.py
├── submissions/
├── requirements.txt
└── README.md