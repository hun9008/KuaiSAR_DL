# XTR Prediction Tasks

| 담당 | 과제 | 예측할 대상 / 라벨 | 데이터 통계 |
|---|---|---|---:|
| Member 1 | Like | 추천 노출에서 `like=1`일 확률 | 전체 데이터 pos 459,741건, 약 3.15% |
| Member 1 | Like END 30 | `like=1`인 시점으로부터 다음 추천 이벤트까지 30분을 초과할 확률 | END 12,597 / known 371,120 (3.39%) |
| Member 2 | Search | 추천 노출에서 `search=1`일 확률 | 전체 데이터 pos 58,027건, 약 0.40% |
| Member 2 | Search END 30 | `search=1`인 시점으로부터 다음 추천 이벤트까지 30분을 초과할 확률 | END 11,335 / known 47,053 (24.09%) |
| Member 3 | Follow | 추천 노출에서 `follow=1`일 확률 | 전체 데이터 pos 65,885건, 약 0.45% |
| Member 3 | Follow END 30 | `follow=1`인 시점으로부터 다음 추천 이벤트까지 30분을 초과할 확률 | END 2,731 / known 53,295 (5.12%) |
| Member 4 | Share (`forward`) | 추천 노출에서 `forward=1`일 확률 | 전체 데이터 pos 44,752건, 약 0.31% |
| Member 4 | Share END 30 (`forward_end_30m`) | `forward=1`인 시점으로부터 다음 추천 이벤트까지 30분을 초과할 확률 | END 1,521 / known 36,737 (4.14%) |

`{action}_end_30m`은 독립적인 전체 노출 과제가 아니다. 해당 action이 발생한 행만
라벨 모집단에 포함하며, 같은 사용자의 다음 추천 이벤트와의 간격이 1,800초보다 크면
1, 1,800초 이하면 0이다. action이 발생하지 않은 행과 사용자별 마지막 이벤트는
라벨을 `NULL`로 두고 학습 및 평가에서 제외한다.


# CTR positive ratio

| 라벨 | 건수 | 비율 |
|---|---:|---:|
| pos (`click=1`) | 7,244,890 | **49.60%** |
| neg (`click=0`) | 7,360,826 | **50.40%** |
| 전체 | 14,605,716 | 100% |
