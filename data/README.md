# Data layout

원본과 생성 Parquet은 Git에 포함하지 않는다. 전체 v2는 공식 Zenodo에서 받으며 압축본도 `data/downloads/`에 보관한다.

```bash
./run.sh download
./run.sh prepare --raw-dir data/raw/KuaiSAR_v2 --output-dir data/processed
```

생성된 `manifest.json`은 팀 전원이 동일한 split과 label 통계를 사용했는지 확인하는 기준 파일이다.
