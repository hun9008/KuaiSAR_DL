# Experiment Protocol

## 고정 조건

1. 모든 모델은 같은 `data/processed/manifest.json`으로 생성된 split을 사용한다.
2. split은 random row split이 아니라 event time 80/10/10이다.
3. categorical 값은 DuckDB의 deterministic hash로 인코딩하며 0은 missing/padding이다.
4. `sampled_1_10`은 train negative만 task별 최대 positive의 10배로 제한한다.
5. validation/test는 항상 original distribution이다.
6. validation AUC로 early stopping하고 best checkpoint에서 test AUC/LogLoss를 한 번 계산한다.
7. 기본 seed 42 결과를 본 실험으로 삼고, 최종 표에는 42/43/44 3-seed 평균과 표준편차를 추가한다.

## 라벨 정의

| Task | Positive |
|---|---|
| like | `rec_inter.like > 0` |
| like_end_30m | `like=1` 이후 다음 추천 이벤트까지 30분 초과 |
| search | `rec_inter.search > 0` |
| search_end_30m | `search=1` 이후 다음 추천 이벤트까지 30분 초과 |
| follow | `rec_inter.follow > 0` |
| follow_end_30m | `follow=1` 이후 다음 추천 이벤트까지 30분 초과 |
| forward | `rec_inter.forward > 0` |
| forward_end_30m | `forward=1` 이후 다음 추천 이벤트까지 30분 초과 |

END30은 독립적인 전체 노출 task가 아니다. 각 행동이 발생한 이벤트를 모집단으로 삼는 조건부 예측 task이며, 행동 미발생 행은 label을 `NULL`로 두어 학습·평가에서 제외한다. 사용자의 데이터 구간 마지막 이벤트도 이후 관측이 없으므로 END로 간주하지 않고 `NULL`로 제외한다.

## Feature policy

허용: user/item ID, user profile categorical 4개, item category/type 4개, event hour/day-of-week.

금지: 현재 interaction의 `duration_ms`, `playing_time`, 모든 target action, 미래 interaction에서 파생한 feature. END label 계산을 위한 `next_event_time`도 feature로 저장하지 않는다.

## 결과 제출 체크리스트

- commit hash와 config 첨부
- `manifest.json` checksum 또는 동일 파일 확인
- 담당 task pair의 `metrics.json` 16개(2 tasks × 4 models × 2 regimes) 확인
- OOM이면 batch size만 낮추고 기록
- model architecture나 seed 변경 시 새 run name 사용
- 실패 run은 파일을 덮지 말고 원인 기록

## 4인 merge 규칙

- Member 1/2/3/4는 각각 Like/Search/Follow/Share(`forward`)와 해당 행동의 END30 task를 함께 담당한다.
- 각 담당자는 LightGCN, DeepFM, DCNv2, HyFormer를 모두 original/sampled 조건에서 실행한다.
- Member별 config 파일의 공통 hyperparameter를 임의 변경하지 않는다.
- 모델 변경은 `src/kuaisar_xtr/models.py`, 실행 변경은 담당 config에서만 한다.
- 각자 `assignment_{behavior}__{behavior_end_30m}.json`과 필요한 `metrics.json`만 공유한다.
- 집계 후 task × regime별 AUC 순위와 original 대비 sampled delta를 보고한다.
