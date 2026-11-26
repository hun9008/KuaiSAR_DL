# END30 학습 배치 구성 수정

## 문제

로더가 4096행씩 배치를 먼저 자르고, 그다음에 라벨이 NULL인 행을 뺐다.
END30 task의 original train은 라벨이 있는 행이 전체의 약 0.3%뿐이다.
그래서 배치가 거의 비어버렸다.

| `forward_end_30m` original train | 수정 전 | 수정 후 |
|---|---|---|
| epoch당 배치 수 | 2,898 | 9 |
| 배치 크기 | 평균 12.7행, 최대 28행 (1행짜리도 생김) | 4,096행 (마지막 배치만 3,969행) |

이 때문에 두 가지 문제가 있었다.

- **학습 중단**: 1행짜리 배치가 들어오면 BatchNorm이 에러를 낸다
  (`Expected more than 1 value per channel when training`).
  DeepFM, DCNv2, HyFormer의 END30 original 실험이 이 에러로 멈췄다.
- **불공정한 비교**: END30 original은 배치당 약 13행으로, sampled는 4096행으로 학습했다.
  그래서 regime 간 차이가 샘플링 효과인지 배치 크기 차이인지 구분할 수 없었다.

모든 행동의 END30 task(`like/search/follow/forward_end_30m`)에 해당한다.

## 수정 (`src/kuaisar_xtr/data.py`, ParquetBatchDataset)

1. row group을 읽자마자 라벨이 NULL인 행을 먼저 뺀다.
2. 남은 행을 다음 row group으로 이어 붙여서 `batch_size`행이 찰 때마다 배치로 내보낸다.
3. 마지막 남은 행은 배치로 내보낸다. 단, 학습에서 1행만 남으면 버린다.
   평가(valid/test)에서는 모든 행을 사용한다.

셔플 방식(row group 순서와 row group 안의 행 순서)은 바뀌지 않았다.

## 영향

- END30 task: 배치가 설정한 `batch_size`대로 채워진다.
- 라벨에 NULL이 없는 행동 task(`like`, `forward` 등): 학습 데이터는 같고,
  row group 경계에서 배치가 나뉘는 위치만 바뀐다.
- valid/test 평가에 쓰는 행은 바뀌지 않는다.

## 검증

- 테스트 2개를 추가했다 (`tests/test_data.py`). 수정 전 코드에서는 둘 다 실패하고, 수정 후에는 전체 11개가 통과한다.
  - 라벨이 듬성듬성한 데이터에서도 row group을 넘어 꽉 찬 배치가 만들어지는지
  - 학습에서 1행만 남으면 버리는지
- 실제 데이터에서 모든 task의 학습 행 수, positive 수, valid/test 행 수가 manifest와 일치했다.
- 멈췄던 `deepfm`, `hyformer`의 `forward_end_30m` original 실험이 정상적으로 끝까지 돌아갔다.

## 참고

이제 END30은 학습 데이터가 작아서 epoch당 업데이트 횟수가 매우 적다.

| | epoch당 업데이트 |
|---|---|
| `forward_end_30m` original (batch 4096) | 9 step |
| `forward_end_30m` sampled (batch 4096) | 5 step |
| LightGCN (batch 65536) | 1 step |

최대 10 epoch이면 END30 모델은 많아야 수십 번밖에 업데이트되지 않아 학습이 덜 될 수 있다.
END30 전용 batch size나 epoch 수를 따로 둘지는 공통 hyperparameter라서 팀에서 함께 정해야 한다.

## 팀원 조치

- pull 받은 뒤 END30 실험을 다시 실행한다. 데이터는 다시 만들 필요 없다.
- 일관성을 위해 행동 task 실험도 수정된 코드로 다시 실행하는 것을 권장한다.
