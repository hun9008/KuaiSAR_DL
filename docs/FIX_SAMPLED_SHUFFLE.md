# sampled_1_10 학습 데이터 셔플 수정

## 문제

`sampled_1_10/{task}/train.parquet`가 라벨별로 몰려 저장되어 있었다.
로더는 row group 순서만 섞고 그룹 안의 행은 섞지 않았다. 그래서 학습 배치가
positive 0% 아니면 100%로 구성되었다. 이 문제는 모든 sampled task에 해당한다.

| 수정 전 예시 | row group별 positive 비율 |
|---|---|
| `forward` | 0%, 0%, 0%, 0%, 100% |
| `search` | 0%, 0%, 0%, 0%, 0%, 100% |
| `forward_end_30m` | positive가 파일 맨 앞에 모여 있음 (첫 배치만 37%, 나머지 0%) |

배치가 한 라벨로만 채워지면 학습이 한쪽으로 크게 쏠리고 BatchNorm 통계도
왜곡된다. 그러면 1:10 샘플링 자체의 효과를 비교할 수 없다.
`original` train은 사용자와 라벨이 이미 섞여 있어 이 문제가 없었다.

## 원인

- `prepare.py`: 1:10 샘플링을 라벨별 window(`PARTITION BY 라벨`)로 수행하는데,
  그 결과를 섞지 않고 그대로 저장했다.
- `data.py`: 학습할 때 row group 순서만 섞었다.

## 수정 후 데이터 구성 (prepare)

1. `rec_inter`에 user/item 피처를 join하고, 행동 라벨과 행동별 END30 라벨을 만든다.
   행동이 없거나 사용자별 마지막 이벤트인 행은 END30 라벨을 NULL로 둔다.
2. 전역 event time 기준으로 80/10/10 분할한다 → `original/{train,valid,test}.parquet`
3. task별로 train 중 라벨이 있는 행에서 negative를 positive의 최대 10배까지 고른다.
   → `sampled_1_10/{task}/train.parquet`
4. **[변경]** 저장하기 전에 행을 seed 고정 순서로 섞는다. 같은 seed면 항상 같은 결과가 나온다.

## 수정 후 학습 로드 (ParquetBatchDataset)

| | train | valid / test |
|---|---|---|
| 파일 | `original` → `original/train.parquet`<br>`sampled_1_10` → `sampled_1_10/{task}/train.parquet` | 두 regime 모두 `original/{valid,test}.parquet` |
| row group 순서 | epoch마다 섞음 | 고정 |
| row group 안의 행 | **[변경]** epoch마다 섞음 (seed 고정) | 고정 |
| NULL 라벨 | 제외 (END30은 해당 행동이 일어난 행만 남음) | 제외 |

## 검증

- 회귀 테스트 2개를 추가했다 (`tests/test_data.py`, `tests/test_prepare.py`).
  수정 전 코드에서는 둘 다 실패하고, 수정 후에는 전체 9개가 통과한다.
- 다시 전처리한 뒤 sampled task의 4096 배치당 positive 비율은 약 8~10%다.
  `search_end_30m`은 원래 비율이 1:3.15라서 샘플링하지 않으며 약 24%다.
- split, 라벨 통계, LightGCN 그래프는 바뀌지 않았다.

## 팀원 조치

1. pull 받은 뒤 데이터를 다시 만든다.
   ```bash
   ./run.sh prepare --raw-dir data/raw/KuaiSAR_v2 --output-dir data/processed --overwrite
   ```
2. 팀원끼리 `data/processed/manifest.json`이 같은지 확인한다.
3. 기존 `sampled_1_10` 결과는 무효이므로 다시 실행한다.
   `original` 결과도 학습 배치 순서가 바뀌었으니, 공정하게 비교하려면 함께 다시 실행한다.
