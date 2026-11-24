# KuaiSAR XTR Prediction Framework

KuaiSAR의 추천 노출 로그에서 CTR을 넘어 8개 XTR 태스크를 동일 조건으로 비교하기 위한 재현 가능한 실험 프레임워크다. 데이터 준비, 시간순 분할, 1:10 negative sampling, 네 baseline 학습, AUC/LogLoss 평가, 체크포인트, 4인 분할 실행과 결과 집계를 한 흐름으로 제공한다.

- 기존 EDA: [reports/kuaisar_v2_eda.md](reports/kuaisar_v2_eda.md)
- XTR 후보 정리: [reports/XTR_candidate.md](reports/XTR_candidate.md)
- END 근거 자료: [reports/시간별 END 전이 분석.pdf](reports/시간별%20END%20전이%20분석.pdf)
- 상세 실험 규약: [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)

## 최종 Task 정의 및 담당

각 행동 task는 두 가지 예측으로 구성한다. 첫 번째는 해당 행동의 발생 확률이고,
두 번째는 **그 행동이 발생한 시점으로부터 30분 뒤 END일 확률**이다.

| 담당 | 행동 예측 | 행동별 END 30 예측 |
|---|---|---|
| Member 1 | Like (`like`) | Like END 30 (`like_end_30m`) |
| Member 2 | Search (`search`) | Search END 30 (`search_end_30m`) |
| Member 3 | Follow (`follow`) | Follow END 30 (`follow_end_30m`) |
| Member 4 | Share (`forward`) | Share END 30 (`forward_end_30m`) |

END 30의 구현 라벨은 해당 행동이 발생한 행을 기준으로 같은 사용자의 다음 추천
이벤트까지 간격이 1,800초를 초과하면 1, 1,800초 이하면 0이다. 행동이 발생하지
않은 행과 관측 구간상 다음 이벤트를 확인할 수 없는 사용자별 마지막 이벤트는
`NULL`로 두고 해당 END 30 task의 학습과 평가에서 제외한다. KuaiSAR 원본에서는
Share 컬럼명이 `forward`이므로 코드와 설정에서도 같은 이름을 사용한다.

## 실험 행렬

| 축 | 조건 |
|---|---|
| Task | `like`, `like_end_30m`, `search`, `search_end_30m`, `follow`, `follow_end_30m`, `forward`, `forward_end_30m` |
| Dataset regime | `original`, `sampled_1_10` |
| Model | LightGCN lane, DeepFM, DCNv2, HyFormer |
| Loss | BCEWithLogitsLoss |
| Metric | AUC (primary), LogLoss (secondary) |
| Split | 전역 event time 기준 80/10/10 |

행동별 END30 라벨은 해당 행동이 발생한 행에서만 정의한다. 예를 들어 `like_end_30m`은 `like=1` 이후 같은 사용자의 다음 추천 이벤트까지 30분을 초과하면 1이다. 행동이 발생하지 않은 행과 사용자별 마지막 이벤트는 정답 불명으로 END30 학습·평가에서 제외한다. `duration_ms`, `playing_time`은 노출 후 관측되는 값이라 입력 feature에서 제외한다.

## 1. 설치

Python 3.10 이상을 사용한다.

```bash
cd /home/hun/KuaiSAR_DL
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

CUDA 환경에서는 시스템 CUDA에 맞는 PyTorch를 먼저 설치한 뒤 위 명령을 실행하는 것을 권장한다.
`run.sh`는 저장소의 `.venv`가 있으면 자동 사용하고, 팀 공유를 위해 group-writable 결과를 생성한다.

## 2. 원본 데이터 배치

공식 Zenodo 배포본(압축 약 806MB, 해제 약 3.5GB)은 아래 명령으로 내려받는다. resume 가능한 다중 연결 다운로드와 공식 MD5 검증을 수행한다.

```bash
./run.sh download
```

그 결과 다음 세 핵심 파일을 포함한 전체 KuaiSAR v2가 배치된다. `src_inter.csv`, `social_network.csv`는 내려받지만 현재 공정 비교 입력에는 사용하지 않는다.

```text
data/raw/KuaiSAR_v2/
├── rec_inter.csv
├── user_features.csv
├── item_features.csv
├── src_inter.csv
└── social_network.csv
```

## 3. 데이터셋 생성

```bash
./run.sh prepare --raw-dir data/raw/KuaiSAR_v2 --output-dir data/processed
```

생성 결과:

```text
data/processed/
├── manifest.json                 # split, label 비율, vocab, sampling 기록
├── original/
│   ├── train.parquet
│   ├── valid.parquet
│   └── test.parquet
└── sampled_1_10/
    └── {task}/train.parquet      # task별 train negative만 최대 1:10
```

`valid`와 `test`는 두 regime 모두 원본 분포를 공유한다. 따라서 sampling 효과를 실제 분포에서 공정하게 비교할 수 있다. 같은 seed에서는 hash-rank 기반으로 정확히 같은 표본을 만든다.

기존 출력 덮어쓰기가 필요할 때만 `--overwrite`를 명시한다.

## 4. 단일 실험

```bash
python scripts/run_experiment.py \
  --config configs/member1_like.json \
  --model deepfm --task like --regime sampled_1_10 --device cuda:0
```

각 run은 `runs/{model}__{task}__{regime}__seed{seed}/`에 다음을 남긴다.

- `config.json`: 실제 사용 설정
- `history.json`: epoch별 train loss, validation AUC/LogLoss
- `best.pt`: validation AUC 기준 best checkpoint와 feature spec
- `metrics.json`: test 결과와 실행 시간

빠른 동작 확인은 `--max-train-steps 2`를 추가한다.

## 5. 4인 분할 실행

행동 예측과 그 행동 이후 END30 예측을 한 쌍으로 묶어 한 명이 전담한다. 각 담당자는 2 tasks × 4 baselines × 2 regimes = 16 runs를 동일 조건으로 실행한다. Share task의 원본 컬럼명과 내부 key는 `forward`다.

| 담당 | 담당 Task | 실행 모델 | 명령 |
|---|---|---|---|
| Member 1 | Like + Like END30 (`like`, `like_end_30m`) | LightGCN, DeepFM, DCNv2, HyFormer | `CUDA_VISIBLE_DEVICES=0 ./run.sh member1 --device cuda:0 --resume` |
| Member 2 | Search + Search END30 (`search`, `search_end_30m`) | LightGCN, DeepFM, DCNv2, HyFormer | `CUDA_VISIBLE_DEVICES=1 ./run.sh member2 --device cuda:0 --resume` |
| Member 3 | Follow + Follow END30 (`follow`, `follow_end_30m`) | LightGCN, DeepFM, DCNv2, HyFormer | `CUDA_VISIBLE_DEVICES=2 ./run.sh member3 --device cuda:0 --resume` |
| Member 4 | Share + Share END30 (`forward`, `forward_end_30m`) | LightGCN, DeepFM, DCNv2, HyFormer | `CUDA_VISIBLE_DEVICES=3 ./run.sh member4 --device cuda:0 --resume` |

`--resume`은 이미 `metrics.json`이 있는 run을 건너뛴다. 공유 스토리지에서는 각자 별도 branch/worktree를 쓰고 `runs/`는 Git에 올리지 않는다. 완료 후 결과만 집계한다.

```bash
./run.sh aggregate --runs-dir runs --output runs/summary.csv
```

이 명령은 개별 run 표 `summary.csv`와 seed 평균/표준편차 표 `summary_aggregate.csv`를 함께 만든다.

## 6. 테스트

```bash
python -m pytest -q
```

테스트는 네 모델의 forward/backward, nullable END 처리, sampled path 분리, synthetic KuaiSAR CSV → Parquet → 1 epoch 학습까지 확인한다.

## 구조

```text
configs/                 # 행동 + 행동별 END30 담당 고정 조건
docs/                    # 실험 규약과 협업 기준
reports/                 # 기존 EDA/분석 산출물
scripts/                 # prepare, single run, assignment, aggregate CLI
src/kuaisar_xtr/
├── prepare.py           # join, END, temporal split, exact sampling
├── data.py              # row-group streaming loader
├── models.py            # four shared-interface baselines
├── trainer.py           # early stop, checkpoint, AUC/LogLoss
└── config.py            # serializable experiment config
tests/
```

### LightGCN lane 해석

대규모 그래프 전체 전파를 mini-batch마다 반복하지 않도록 현재 lane은 LightGCN의 layer-0 ID embedding dot-product를 사용한다. 즉, 그래프 전파 전 collaborative-filtering lower bound다. 보고서에서는 `LightGCN-0`으로 표기해야 하며, full-graph LightGCN 결과와 혼동하지 않는다. 나머지 모델은 공통 categorical context 전체를 사용한다.
