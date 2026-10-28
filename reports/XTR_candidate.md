# XTR Prediction Candidate

| 후보 과제 | 예측할 대상 / 라벨 | 데이터에서의 가능성 | 
|---|---|---|
| Like prediction | 추천된 아이템에 `like=1`이 발생할 확률 | pos 459,741건, 약 3.15% | 
| Search prediction | 추천 아이템을 보다가 `search=1`이 발생할 확률 | pos 58,027건, 약 **0.40%** | 
| Follow prediction | 해당 아이템에서 작성자를 팔로우할 확률 | pos 65,885건, 약 **0.45%** | 
| Share prediction (forward) | `forward=1`이 발생할 확률 | pos 44,752건, 약 **0.31%** |
| Completion / Long-view prediction | 예: `playing_time / duration_ms ≥ 0.9`, 또는 시청 시간 ≥ 정한 임계값 | 기존 필드로 파생 가능. pos 비율은 별도 집계 필요 | 


# CTR positive ratio

| 라벨 | 건수 | 비율 |
|---|---:|---:|
| pos (`click=1`) | 7,244,890 | **49.60%** |
| neg (`click=0`) | 7,360,826 | **50.40%** |
| 전체 | 14,605,716 | 100% |