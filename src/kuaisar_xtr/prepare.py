from __future__ import annotations

import shutil
from pathlib import Path

import duckdb

from .constants import (
    ACTION_END_TASKS,
    CATEGORICAL_FEATURES,
    DEFAULT_HASH_BUCKETS,
    END_THRESHOLD_SECONDS,
    GRAPH_ACTION_FOR_TASK,
    TASKS,
)
from .utils import atomic_json_dump


def _q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _hashed(expression: str, buckets: int) -> str:
    # Zero is reserved for missing/padding. DuckDB hash is deterministic.
    return f"CASE WHEN {expression} IS NULL THEN 0 ELSE 1 + hash(CAST({expression} AS VARCHAR)) % {buckets} END"


def prepare_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    negative_ratio: int = 10,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    """Build leakage-safe temporal splits and train-only 1:N samples.

    Expected raw files are ``rec_inter.csv``, ``user_features.csv`` and
    ``item_features.csv``. ``<action>_end_30m`` is defined only when that
    action occurred and is 1 when the user's next recommendation event is more
    than 30 minutes away. A user's final event has an unknown END label.
    """
    if not 0 < train_ratio < 1 or not 0 <= valid_ratio < 1 or train_ratio + valid_ratio >= 1:
        raise ValueError("Require 0 < train_ratio and train_ratio + valid_ratio < 1")
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    required = [raw_dir / name for name in ("rec_inter.csv", "user_features.csv", "item_features.csv")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing KuaiSAR files: {missing}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(output_dir / "prepare.duckdb"))
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA preserve_insertion_order=false")
    rec, users, items = map(_q, required)
    con.execute(f"CREATE OR REPLACE VIEW rec AS SELECT * FROM read_csv_auto('{rec}', header=true)")
    con.execute(f"CREATE OR REPLACE VIEW users AS SELECT * FROM read_csv_auto('{users}', header=true)")
    con.execute(f"CREATE OR REPLACE VIEW items AS SELECT * FROM read_csv_auto('{items}', header=true)")
    expected_columns = {
        "rec": {"user_id", "item_id", "time", "like", "search", "follow", "forward"},
        "users": {"user_id", "onehot_feat1", "onehot_feat2", "search_active_level", "rec_active_level"},
        "items": {"item_id", "first_level_category_name_en", "second_level_category_name_en", "item_type", "upload_type"},
    }
    for view, expected in expected_columns.items():
        actual = {row[0] for row in con.execute(f"DESCRIBE {view}").fetchall()}
        if missing_columns := expected - actual:
            con.close()
            raise ValueError(f"{view} is missing columns: {sorted(missing_columns)}")
    con.execute("""
        CREATE OR REPLACE TABLE timed AS
        SELECT r.*, CAST(r.time AS TIMESTAMP) AS event_time,
               lead(CAST(r.time AS TIMESTAMP)) OVER
                   (PARTITION BY r.user_id ORDER BY CAST(r.time AS TIMESTAMP), r.item_id) AS next_event_time
        FROM rec r
    """)
    # Collision-free, compact entity IDs are required for a real user-item graph.
    con.execute("""
        CREATE OR REPLACE TABLE user_id_map AS
        SELECT user_id, row_number() OVER (ORDER BY user_id) AS graph_id
        FROM (SELECT DISTINCT user_id FROM rec WHERE user_id IS NOT NULL)
    """)
    con.execute("""
        CREATE OR REPLACE TABLE item_id_map AS
        SELECT item_id, row_number() OVER (ORDER BY item_id) AS graph_id
        FROM (SELECT DISTINCT item_id FROM rec WHERE item_id IS NOT NULL)
    """)
    user_cols = ["onehot_feat1", "onehot_feat2", "search_active_level", "rec_active_level"]
    item_cols = ["first_level_category_name_en", "second_level_category_name_en", "item_type", "upload_type"]
    feature_exprs = [
        "CAST(um.graph_id AS BIGINT) AS user_id",
        "CAST(im.graph_id AS BIGINT) AS item_id",
    ]
    feature_exprs += [f"{_hashed('u.' + c, DEFAULT_HASH_BUCKETS[c])} AS {c}" for c in user_cols]
    feature_exprs += [f"{_hashed('i.' + c, DEFAULT_HASH_BUCKETS[c])} AS {c}" for c in item_cols]
    feature_exprs += [
        "1 + hour(t.event_time) AS event_hour",
        "1 + dayofweek(t.event_time) AS event_dow",
    ]
    end_exprs = [
        f"CASE WHEN COALESCE(t.{_ident(action)}, 0) <= 0 OR next_event_time IS NULL THEN NULL "
        f"ELSE CAST(date_diff('second', event_time, next_event_time) > {END_THRESHOLD_SECONDS} AS TINYINT) END AS {_ident(task)}"
        for task, action in ACTION_END_TASKS.items()
    ]
    action_exprs = [f"CAST(COALESCE(t.{_ident(task)}, 0) > 0 AS TINYINT) AS {_ident(task)}" for task in ("like", "search", "follow", "forward")]
    con.execute(f"""
        CREATE OR REPLACE TABLE examples AS
        SELECT {', '.join(feature_exprs + action_exprs + end_exprs)}, t.event_time
        FROM timed t
        JOIN user_id_map um ON t.user_id = um.user_id
        JOIN item_id_map im ON t.item_id = im.item_id
        LEFT JOIN users u ON t.user_id = u.user_id
        LEFT JOIN items i ON t.item_id = i.item_id
    """)
    train_cut, valid_cut = con.execute(
        "SELECT quantile_cont(event_time, ?), quantile_cont(event_time, ?) FROM examples",
        [train_ratio, train_ratio + valid_ratio],
    ).fetchone()
    split_predicates = {
        "train": "event_time < ?",
        "valid": "event_time >= ? AND event_time < ?",
        "test": "event_time >= ?",
    }
    split_args = {"train": [train_cut], "valid": [train_cut, valid_cut], "test": [valid_cut]}
    columns = CATEGORICAL_FEATURES + list(TASKS)
    selected_columns = ", ".join(_ident(column) for column in columns)
    for split, predicate in split_predicates.items():
        path = _q(output_dir / "original" / f"{split}.parquet")
        (output_dir / "original").mkdir(exist_ok=True)
        query = f"COPY (SELECT {selected_columns} FROM examples WHERE {predicate}) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        con.execute(query, split_args[split])
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(exist_ok=True)
    graph_stats = {}
    for action in ("like", "search", "follow", "forward"):
        dest = _q(graph_dir / f"{action}.parquet")
        action_id = _ident(action)
        con.execute(f"""
            COPY (
                SELECT DISTINCT user_id, item_id
                FROM examples
                WHERE event_time < ? AND {action_id} = 1
            ) TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """, [train_cut])
        graph_stats[action] = {
            "path": f"graphs/{action}.parquet",
            "edges": int(con.execute(
                f"SELECT count(*) FROM (SELECT DISTINCT user_id, item_id FROM examples WHERE event_time < ? AND {action_id}=1)",
                [train_cut],
            ).fetchone()[0]),
        }
    # Balanced variants reuse untouched validation/test sets; only training is sampled.
    for task in TASKS:
        task_dir = output_dir / "sampled_1_10" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        task_id = _ident(task)
        label_known = f"{task_id} IS NOT NULL"
        dest = _q(task_dir / "train.parquet")
        con.execute(f"""
            COPY (
                WITH eligible AS (
                    SELECT * FROM examples WHERE event_time < ? AND {label_known}
                ), ranked AS (
                    SELECT *,
                           row_number() OVER (PARTITION BY {task_id} ORDER BY hash(user_id, item_id, event_time, {seed})) AS class_rank,
                           sum({task_id}) OVER () AS positive_count
                    FROM eligible
                )
                SELECT {selected_columns} FROM ranked
                WHERE {task_id}=1 OR class_rank <= {negative_ratio} * positive_count
                -- The class-partitioned window emits rows grouped by label; without a
                -- deterministic shuffle, row groups (and batches) become single-class.
                ORDER BY hash(user_id, item_id, event_time, {seed}, 'row_order')
            ) TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """, [train_cut])
    stats = {}
    for task in TASKS:
        stats[task] = {}
        for split, predicate in split_predicates.items():
            task_id = _ident(task)
            n, pos = con.execute(f"SELECT count({task_id}), sum({task_id}) FROM examples WHERE {predicate}", split_args[split]).fetchone()
            stats[task][split] = {"rows": int(n or 0), "positive": int(pos or 0), "positive_rate": float(pos / n) if n else None}
    num_users = int(con.execute("SELECT count(*) FROM user_id_map").fetchone()[0])
    num_items = int(con.execute("SELECT count(*) FROM item_id_map").fetchone()[0])
    vocab_sizes = {**DEFAULT_HASH_BUCKETS, "user_id": num_users, "item_id": num_items}
    manifest = {
        "version": 2, "features": CATEGORICAL_FEATURES, "vocab_sizes": vocab_sizes,
        "tasks": list(TASKS),
        "end_definition": {
            "threshold_seconds": END_THRESHOLD_SECONDS,
            "conditioned_on_action": ACTION_END_TASKS,
            "terminal_event_label": "unknown",
        },
        "split": {"method": "global_time", "train_cut": str(train_cut), "valid_cut": str(valid_cut)},
        "sampling": {
            "method": "deterministic_hash_rank", "negative_ratio": negative_ratio, "train_only": True,
            "row_order": "deterministic_hash_shuffle",
        },
        "id_mapping": {"method": "contiguous_full_corpus_entities", "padding_id": 0, "hash_collisions": False},
        "lightgcn_graphs": {
            "definition": "distinct positive action edges from train split only",
            "task_to_action": GRAPH_ACTION_FOR_TASK,
            "actions": graph_stats,
        },
        "stats": stats,
    }
    atomic_json_dump(manifest, output_dir / "manifest.json")
    con.close()
    return manifest
