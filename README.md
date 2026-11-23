# Principle of Deep Learning Final Project

EDA report: [link to EDA report](./reports/kuaisar_v2_eda.md)

user-item interaction 외에도 다양한 task(like, share 등)에 대한 데이터를 가지고 있는 KuaiSAR 데이터셋에서 CTR prediction을 확장한 XTR prediction에 대한 general RS 성능을 확인하고 분석하는 프로젝트. 

XTR: Click 외에 다른 behavior prediction task.

# XTR Prediction Candidate

| 후보 과제 | 예측할 대상 / 라벨 | 데이터에서의 가능성 | 
|---|---|---|
| Like prediction | 추천된 아이템에 `like=1`이 발생할 확률 | pos 459,741건, 약 3.15% | 
| Search prediction | 추천 아이템을 보다가 `search=1`이 발생할 확률 | pos 58,027건, 약 **0.40%** | 
| Follow prediction | 해당 아이템에서 작성자를 팔로우할 확률 | pos 65,885건, 약 **0.45%** | 
| Share prediction (forward) | `forward=1`이 발생할 확률 | pos 44,752건, 약 **0.31%** |

4가지와 추가로 'reports/시간별 END 전이 분석.pdf'의 내용대로 XTR의 종료조건에 대한 prediction까지 총 8개의 XTR task에 대해 실험 예정.

# Experiments details

General RS baseline: LightGCN, DeepFM, DCNv2, Hyformer(최근 transformer 기반 CTR prediction 모델)
4개의 baseline에서 공통 hyper parameter 조건에서 XTR 8개 task 성능 분석.
(BCE loss base로 설계)

Metric: AUC 평가

Dataset constraint: 1:10 sampling data, origin data 두 가지 조건에서 평가. (1:10 sampling은 EDA결과에서 pos수가 작았던 점을 고려해 추가 확인할 예정)
