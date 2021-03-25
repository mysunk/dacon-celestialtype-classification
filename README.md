천체 유형 분류 대회
=======================================

Sloan Digital Sky Survey 데이터를 통해 천체 유형을 분류하는 대회입니다.
대회 링크:
https://www.dacon.io/competitions/official/235573/overview/description/


Dataset
==================
이 저장소에 데이터셋은 제외되어 있습니다.  
데이터셋 출처: 
https://www.dacon.io/competitions/official/235573/data/

Structure
==================
```setup
.
└── main.py
└── cv_objs.py
└── cv_run.py
└── utils.py
└── val_objs.py
└── val_run.py
```
* main.py: main 모델링 파일
* cv_objs & cv_run.py: cross validation 튜닝 파일
* val_objs & val_run.py: holdout validation 튜닝 파일
* utils.py: custom 함수가 정의된 파일

Results
==================
* 평가지표: LogLoss
* MSE 결과: 0.35
* private rank: 45/352