# KuaiSAR XTR Leaderboard

모든 수치는 validation AUC로 선택한 best checkpoint의 test 결과다. AUC는 높을수록,
LogLoss는 낮을수록 좋다. 현재 등록된 결과는 seed 42 단일 실행이며, 최종 보고용
평균과 표준편차는 seed 42/43/44가 모두 완료된 뒤 추가한다.

## 담당별 진행 현황

| 담당 | Task pair | 상태 | 현재 최고 Test AUC |
|---|---|---|---|
| Member 1 | Like / Like END30 | 미등록 | - |
| Member 2 | Search / Search END30 | **완료 (16/16)** | Search 0.854329 / Search END30 0.718563 |
| Member 3 | Follow / Follow END30 | 미등록 | - |
| Member 4 | Share / Share END30 | 미등록 | - |

## Member 2: Search

Test rows: 1,460,572. Search의 최고 결과는 original 데이터의 DCNv2다.

### Original

| 순위 | Model | Valid AUC | Test AUC | Test LogLoss | Epochs | Time (s) |
|---:|---|---:|---:|---:|---:|---:|
| **1** | **DCNv2** | 0.863176 | **0.854329** | **0.021050** | 4 | 251.184 |
| 2 | DeepFM | 0.855980 | 0.848212 | 0.021348 | 4 | 337.470 |
| 3 | HyFormer | 0.853341 | 0.846653 | 0.021375 | 4 | 195.237 |
| 4 | LightGCN | 0.613019 | 0.573664 | 0.660509 | 10 | 109.192 |

### Sampled 1:10

| 순위 | Model | Valid AUC | Test AUC | Test LogLoss | Epochs | Time (s) |
|---:|---|---:|---:|---:|---:|---:|
| **1** | **DCNv2** | 0.842004 | **0.833547** | 0.037934 | 5 | 20.269 |
| 2 | DeepFM | 0.512336 | 0.530516 | **0.034159** | 4 | 29.976 |
| 3 | LightGCN | 0.495254 | 0.492879 | 0.693147 | 4 | 9.661 |
| 4 | HyFormer | 0.478863 | 0.480933 | 0.047922 | 6 | 25.073 |

## Member 2: Search END30

`search=1`인 이벤트만 모집단으로 하며 test rows는 5,246이다. Search END30의 최고
결과도 original 데이터의 DCNv2다.

### Original

| 순위 | Model | Valid AUC | Test AUC | Test LogLoss | Epochs | Time (s) |
|---:|---|---:|---:|---:|---:|---:|
| **1** | **DCNv2** | 0.739069 | **0.718563** | **0.455824** | 4 | 145.928 |
| 2 | DeepFM | 0.721317 | 0.711925 | 0.464822 | 4 | 138.831 |
| 3 | HyFormer | 0.729148 | 0.697741 | 0.472214 | 4 | 166.522 |
| 4 | LightGCN | 0.511026 | 0.502692 | 0.693102 | 10 | 52.918 |

### Sampled 1:10

| 순위 | Model | Valid AUC | Test AUC | Test LogLoss | Epochs | Time (s) |
|---:|---|---:|---:|---:|---:|---:|
| **1** | **DCNv2** | 0.735863 | **0.713523** | **0.462332** | 9 | 36.935 |
| 2 | DeepFM | 0.644839 | 0.656433 | 0.496778 | 8 | 17.908 |
| 3 | HyFormer | 0.594046 | 0.583159 | 0.547523 | 10 | 32.146 |
| 4 | LightGCN | 0.502402 | 0.499327 | 0.693145 | 4 | 3.430 |

## Sampling 영향

아래 값은 `sampled_1_10 Test AUC - original Test AUC`다. 음수는 sampling 후 성능
하락을 뜻한다.

| Model | Search ΔAUC | Search END30 ΔAUC |
|---|---:|---:|
| LightGCN | -0.080784 | -0.003365 |
| DeepFM | -0.317696 | -0.055492 |
| DCNv2 | **-0.020781** | **-0.005040** |
| HyFormer | -0.365720 | -0.114582 |

현재 seed 42에서는 두 task 모두 original regime이 우세했다. DCNv2는 sampling 후에도
가장 높은 AUC를 유지했고 성능 하락 폭도 context baseline 중 가장 작았다. Search
positive가 전체 노출의 약 0.4%이므로 1:10 sampling은 학습 분포를 크게 바꾸며,
validation/test를 원본 분포로 평가할 때 특히 DeepFM과 HyFormer의 AUC 하락이 컸다.

LightGCN은 두 task 모두 train Search graph 47,288 edges를 공유한다. END30 label은
graph 구성에 사용하지 않았다.

## 재현 정보

- Seed: 42
- Split: global event-time 80/10/10
- Regimes: original, train-negative sampled 1:10
- Selection: best validation AUC
- Test metrics: AUC, LogLoss
- Source: `runs/{model}__{task}__{regime}__seed42/metrics.json`
- Result commit 기준: 실험 산출물의 각 `config.json` 및 `metrics.json` 참조

