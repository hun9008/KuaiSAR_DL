from __future__ import annotations

import ast
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("/Users/jeong-yonghun/Desktop/kuaisar")
DATA_DIR = BASE_DIR / "KuaiSAR_v2"
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = REPORT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "kuaisar_v2_eda.md"

CHUNK_SIZE = 200_000
MAX_SAMPLE_ROWS = 250_000
RANDOM_SEED = 42


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_float(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def sample_chunk(chunk: pd.DataFrame, max_rows: int = 20_000) -> pd.DataFrame:
    if len(chunk) <= max_rows:
        return chunk.copy()
    return chunk.sample(n=max_rows, random_state=RANDOM_SEED)


def add_title(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def save_fig(fig: plt.Figure, name: str) -> str:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def parse_list_length(value: str) -> float:
    if pd.isna(value):
        return np.nan
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return float(len(parsed))
    except Exception:
        return np.nan
    return np.nan


def analyze_user_features(path: Path) -> dict:
    df = pd.read_csv(path)
    summary = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing": df.isna().sum().to_dict(),
        "unique_users": int(df["user_id"].nunique()),
        "value_counts": {},
        "figures": [],
    }

    for col in ["onehot_feat1", "onehot_feat2", "search_active_level", "rec_active_level"]:
        vc = df[col].value_counts().sort_index()
        summary["value_counts"][col] = vc.to_dict()

        fig, ax = plt.subplots(figsize=(7, 4))
        vc.plot(kind="bar", ax=ax, color="#3b82f6")
        add_title(ax, f"{col} Distribution", col, "Count")
        summary["figures"].append(save_fig(fig, f"user_features_{col}.png"))

    return summary


def analyze_social_network(path: Path) -> dict:
    df = pd.read_csv(path)
    out_degree = df.groupby("user_id").size().sort_values(ascending=False)
    in_degree = df.groupby("user_follow_id").size().sort_values(ascending=False)

    summary = {
        "rows": len(df),
        "cols": len(df.columns),
        "unique_users": int(pd.unique(pd.concat([df["user_id"], df["user_follow_id"]])).shape[0]),
        "unique_source_users": int(df["user_id"].nunique()),
        "unique_target_users": int(df["user_follow_id"].nunique()),
        "avg_out_degree": float(out_degree.mean()),
        "avg_in_degree": float(in_degree.mean()),
        "top_out_degree": out_degree.head(10).to_dict(),
        "top_in_degree": in_degree.head(10).to_dict(),
        "figures": [],
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log1p(out_degree.values), bins=20, color="#10b981", edgecolor="white")
    add_title(ax, "Social Network Out-Degree (log1p)", "log1p(out-degree)", "Users")
    summary["figures"].append(save_fig(fig, "social_network_out_degree_hist.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log1p(in_degree.values), bins=20, color="#f59e0b", edgecolor="white")
    add_title(ax, "Social Network In-Degree (log1p)", "log1p(in-degree)", "Users")
    summary["figures"].append(save_fig(fig, "social_network_in_degree_hist.png"))

    return summary


def analyze_item_features(path: Path) -> dict:
    row_count = 0
    missing = Counter()
    item_type_counter = Counter()
    upload_type_counter = Counter()
    first_cat_counter = Counter()
    second_cat_counter = Counter()
    upload_dates = Counter()
    caption_lengths = []
    sampled_frames = []

    usecols = [
        "item_id",
        "first_level_category_name_en",
        "second_level_category_name_en",
        "item_type",
        "upload_time",
        "upload_type",
        "caption",
    ]

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE):
        row_count += len(chunk)
        missing.update(chunk.isna().sum().to_dict())
        item_type_counter.update(chunk["item_type"].fillna("NA").astype(str))
        upload_type_counter.update(chunk["upload_type"].fillna("NA").astype(str))
        first_cat_counter.update(chunk["first_level_category_name_en"].fillna("NA").astype(str))
        second_cat_counter.update(chunk["second_level_category_name_en"].fillna("NA").astype(str))
        upload_dates.update(chunk["upload_time"].fillna("NA").astype(str))

        sampled = sample_chunk(chunk[["caption", "upload_time"]], max_rows=10_000)
        sampled["caption_token_count"] = sampled["caption"].map(parse_list_length)
        caption_lengths.extend(sampled["caption_token_count"].dropna().tolist())
        sampled_frames.append(sampled.drop(columns=["caption"]))
        if sum(len(frame) for frame in sampled_frames) > MAX_SAMPLE_ROWS:
            sampled_frames = [pd.concat(sampled_frames, ignore_index=True).sample(n=MAX_SAMPLE_ROWS, random_state=RANDOM_SEED)]

    sampled_df = pd.concat(sampled_frames, ignore_index=True) if sampled_frames else pd.DataFrame()
    sampled_df["upload_time"] = pd.to_datetime(sampled_df["upload_time"], errors="coerce")

    summary = {
        "rows": row_count,
        "cols": 18,
        "missing": dict(missing),
        "item_type": dict(item_type_counter),
        "upload_type_top10": dict(upload_type_counter.most_common(10)),
        "first_cat_top10": dict(first_cat_counter.most_common(10)),
        "second_cat_top10": dict(second_cat_counter.most_common(10)),
        "upload_date_min": min(k for k in upload_dates if k != "NA"),
        "upload_date_max": max(k for k in upload_dates if k != "NA"),
        "caption_token_mean": float(np.mean(caption_lengths)) if caption_lengths else np.nan,
        "caption_token_median": float(np.median(caption_lengths)) if caption_lengths else np.nan,
        "figures": [],
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pd.Series(dict(first_cat_counter.most_common(12))).sort_values().plot(kind="barh", ax=ax, color="#6366f1")
    add_title(ax, "Top First-Level Categories", "Items", "")
    summary["figures"].append(save_fig(fig, "item_features_first_category_top12.png"))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    pd.Series(item_type_counter).sort_values().plot(kind="bar", ax=ax, color="#ef4444")
    add_title(ax, "Item Type Distribution", "Item Type", "Count")
    summary["figures"].append(save_fig(fig, "item_features_item_type.png"))

    if not sampled_df.empty:
        daily = sampled_df["upload_time"].dt.to_period("D").value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        daily.plot(ax=ax, color="#14b8a6")
        add_title(ax, "Sampled Upload Date Trend", "Date", "Sample Count")
        summary["figures"].append(save_fig(fig, "item_features_upload_trend_sample.png"))

    if caption_lengths:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.clip(caption_lengths, 0, 200), bins=40, color="#8b5cf6", edgecolor="white")
        add_title(ax, "Caption Token Count Distribution", "Token Count (clipped at 200)", "Sample Count")
        summary["figures"].append(save_fig(fig, "item_features_caption_token_hist.png"))

    return summary


def analyze_rec_inter(path: Path) -> dict:
    row_count = 0
    missing = Counter()
    action_sums = Counter()
    time_counter = Counter()
    hour_counter = Counter()
    duration_sample = []
    playing_sample = []
    sampled_frames = []
    unique_user_sample = set()
    unique_item_sample = set()

    action_cols = ["forward", "like", "follow", "search_photo_related", "search", "click"]
    usecols = ["user_id", "item_id", "duration_ms", "playing_time", "time"] + action_cols

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE):
        row_count += len(chunk)
        missing.update(chunk.isna().sum().to_dict())

        for col in action_cols:
            action_sums[col] += int(chunk[col].fillna(0).sum())

        time_series = pd.to_datetime(chunk["time"], errors="coerce")
        time_counter.update(time_series.dt.to_period("D").astype(str).fillna("NA"))
        hour_counter.update(time_series.dt.hour.fillna(-1).astype(int))

        sampled = sample_chunk(chunk, max_rows=12_000)
        duration_sample.extend(sampled["duration_ms"].dropna().tolist())
        playing_sample.extend(sampled["playing_time"].dropna().tolist())
        sampled_frames.append(sampled[["user_id", "item_id"]])
        unique_user_sample.update(sampled["user_id"].dropna().astype(int).tolist())
        unique_item_sample.update(sampled["item_id"].dropna().astype(int).tolist())

        if len(duration_sample) > MAX_SAMPLE_ROWS:
            duration_sample = duration_sample[:MAX_SAMPLE_ROWS]
        if len(playing_sample) > MAX_SAMPLE_ROWS:
            playing_sample = playing_sample[:MAX_SAMPLE_ROWS]

    summary = {
        "rows": row_count,
        "cols": 12,
        "missing": dict(missing),
        "action_rates": {k: v / row_count for k, v in action_sums.items()},
        "date_min": min(k for k in time_counter if k != "NA"),
        "date_max": max(k for k in time_counter if k != "NA"),
        "duration_mean_sample": float(np.mean(duration_sample)) if duration_sample else np.nan,
        "duration_median_sample": float(np.median(duration_sample)) if duration_sample else np.nan,
        "playing_mean_sample": float(np.mean(playing_sample)) if playing_sample else np.nan,
        "playing_median_sample": float(np.median(playing_sample)) if playing_sample else np.nan,
        "sample_unique_users": len(unique_user_sample),
        "sample_unique_items": len(unique_item_sample),
        "figures": [],
    }

    daily = pd.Series({k: v for k, v in time_counter.items() if k != "NA"}).sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    daily.plot(ax=ax, marker="o", color="#2563eb")
    add_title(ax, "Recommendation Interactions by Day", "Date", "Rows")
    summary["figures"].append(save_fig(fig, "rec_inter_daily_counts.png"))

    hourly = pd.Series({k: v for k, v in hour_counter.items() if k >= 0}).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    hourly.plot(kind="bar", ax=ax, color="#f97316")
    add_title(ax, "Recommendation Interactions by Hour", "Hour", "Rows")
    summary["figures"].append(save_fig(fig, "rec_inter_hourly_counts.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(np.array(duration_sample), 0, 600_000), bins=50, color="#22c55e", edgecolor="white")
    add_title(ax, "Duration Distribution", "duration_ms (clipped at 600,000)", "Sample Count")
    summary["figures"].append(save_fig(fig, "rec_inter_duration_hist.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(np.array(playing_sample), 0, 600_000), bins=50, color="#a855f7", edgecolor="white")
    add_title(ax, "Playing Time Distribution", "playing_time (clipped at 600,000)", "Sample Count")
    summary["figures"].append(save_fig(fig, "rec_inter_playing_time_hist.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    pd.Series(summary["action_rates"]).sort_values().plot(kind="barh", ax=ax, color="#0ea5e9")
    add_title(ax, "Positive Action Rates", "Rate", "")
    summary["figures"].append(save_fig(fig, "rec_inter_action_rates.png"))

    return summary


def analyze_src_inter(path: Path) -> dict:
    row_count = 0
    missing = Counter()
    source_counter = Counter()
    item_type_counter = Counter()
    date_counter = Counter()
    click_values = []
    keyword_lengths = []
    sampled_session_pairs = set()

    usecols = [
        "user_id",
        "search_session_id",
        "search_session_source",
        "keyword",
        "item_id",
        "click_cnt",
        "item_type",
        "time",
    ]

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE):
        row_count += len(chunk)
        missing.update(chunk.isna().sum().to_dict())
        source_counter.update(chunk["search_session_source"].fillna("NA").astype(str))
        item_type_counter.update(chunk["item_type"].fillna("NA").astype(str))
        date_counter.update(pd.to_datetime(chunk["time"], errors="coerce").dt.to_period("D").astype(str).fillna("NA"))

        sampled = sample_chunk(chunk, max_rows=12_000)
        click_values.extend(sampled["click_cnt"].dropna().tolist())
        keyword_lengths.extend(sampled["keyword"].fillna("").astype(str).map(parse_list_length).dropna().tolist())
        sampled_session_pairs.update(
            zip(sampled["user_id"].fillna(-1).astype(int).tolist(), sampled["search_session_id"].fillna(-1).astype(int).tolist())
        )
        if len(click_values) > MAX_SAMPLE_ROWS:
            click_values = click_values[:MAX_SAMPLE_ROWS]
        if len(keyword_lengths) > MAX_SAMPLE_ROWS:
            keyword_lengths = keyword_lengths[:MAX_SAMPLE_ROWS]

    summary = {
        "rows": row_count,
        "cols": 9,
        "missing": dict(missing),
        "source_counts": dict(source_counter),
        "item_type_counts": dict(item_type_counter),
        "date_min": min(k for k in date_counter if k != "NA"),
        "date_max": max(k for k in date_counter if k != "NA"),
        "avg_click_cnt_sample": float(np.mean(click_values)) if click_values else np.nan,
        "median_click_cnt_sample": float(np.median(click_values)) if click_values else np.nan,
        "avg_keyword_token_len_sample": float(np.mean(keyword_lengths)) if keyword_lengths else np.nan,
        "median_keyword_token_len_sample": float(np.median(keyword_lengths)) if keyword_lengths else np.nan,
        "sample_unique_sessions": len(sampled_session_pairs),
        "figures": [],
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    pd.Series(source_counter).sort_values().plot(kind="barh", ax=ax, color="#14b8a6")
    add_title(ax, "Search Session Source Distribution", "Rows", "")
    summary["figures"].append(save_fig(fig, "src_inter_source_counts.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    pd.Series(item_type_counter).sort_values().plot(kind="barh", ax=ax, color="#fb7185")
    add_title(ax, "Search Item Type Distribution", "Rows", "")
    summary["figures"].append(save_fig(fig, "src_inter_item_type_counts.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    daily = pd.Series({k: v for k, v in date_counter.items() if k != "NA"}).sort_index()
    daily.plot(ax=ax, marker="o", color="#7c3aed")
    add_title(ax, "Search Interaction Rows by Day", "Date", "Rows")
    summary["figures"].append(save_fig(fig, "src_inter_daily_counts.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(np.array(click_values), 0, 10), bins=np.arange(0, 12) - 0.5, color="#f59e0b", edgecolor="white")
    add_title(ax, "click_cnt Distribution", "click_cnt (clipped at 10)", "Sample Count")
    summary["figures"].append(save_fig(fig, "src_inter_click_cnt_hist.png"))

    return summary


def md_table_from_dict(data: dict, key_name: str, value_name: str, limit: int | None = None) -> str:
    items = list(data.items())
    if limit is not None:
        items = items[:limit]
    lines = [f"| {key_name} | {value_name} |", "|---|---:|"]
    for key, value in items:
        display = fmt_float(value) if isinstance(value, float) else fmt_int(value) if isinstance(value, int) else str(value)
        lines.append(f"| {key} | {display} |")
    return "\n".join(lines)


def build_report(results: dict) -> str:
    lines = [
        "# KuaiSAR_v2 EDA Report",
        "",
        "## Overview",
        "",
        "- Analysis date: 2026-03-31",
        f"- Data directory: `{DATA_DIR}`",
        f"- Output figures directory: `{FIG_DIR}`",
        "- Large files were profiled with chunked aggregation and sampled distributions to avoid full-memory loading.",
        "",
    ]

    uf = results["user_features"]
    lines += [
        "## user_features.csv",
        "",
        f"- Rows: {fmt_int(uf['rows'])}",
        f"- Columns: {uf['cols']}",
        f"- Unique users: {fmt_int(uf['unique_users'])}",
        "",
        "### Key distributions",
        "",
        md_table_from_dict(uf["value_counts"]["search_active_level"], "search_active_level", "count"),
        "",
        md_table_from_dict(uf["value_counts"]["rec_active_level"], "rec_active_level", "count"),
        "",
    ]
    for fig in uf["figures"]:
        lines += [f"![{fig}](figures/{fig})", ""]

    sn = results["social_network"]
    lines += [
        "## social_network.csv",
        "",
        f"- Rows: {fmt_int(sn['rows'])}",
        f"- Columns: {sn['cols']}",
        f"- Unique users involved: {fmt_int(sn['unique_users'])}",
        f"- Unique source users: {fmt_int(sn['unique_source_users'])}",
        f"- Unique target users: {fmt_int(sn['unique_target_users'])}",
        f"- Average out-degree: {fmt_float(sn['avg_out_degree'])}",
        f"- Average in-degree: {fmt_float(sn['avg_in_degree'])}",
        "",
        "### Top out-degree users",
        "",
        md_table_from_dict(sn["top_out_degree"], "user_id", "follow_count", limit=10),
        "",
        "### Top in-degree users",
        "",
        md_table_from_dict(sn["top_in_degree"], "user_id", "follower_count", limit=10),
        "",
    ]
    for fig in sn["figures"]:
        lines += [f"![{fig}](figures/{fig})", ""]

    it = results["item_features"]
    lines += [
        "## item_features.csv",
        "",
        f"- Rows: {fmt_int(it['rows'])}",
        "- Columns: 18",
        f"- Upload date range: {it['upload_date_min']} to {it['upload_date_max']}",
        f"- Sample caption token mean: {fmt_float(it['caption_token_mean'])}",
        f"- Sample caption token median: {fmt_float(it['caption_token_median'])}",
        "",
        "### Item type distribution",
        "",
        md_table_from_dict(it["item_type"], "item_type", "count"),
        "",
        "### Top upload types",
        "",
        md_table_from_dict(it["upload_type_top10"], "upload_type", "count"),
        "",
        "### Top first-level categories",
        "",
        md_table_from_dict(it["first_cat_top10"], "category", "count"),
        "",
        "### Top second-level categories",
        "",
        md_table_from_dict(it["second_cat_top10"], "category", "count"),
        "",
    ]
    for fig in it["figures"]:
        lines += [f"![{fig}](figures/{fig})", ""]

    rec = results["rec_inter"]
    lines += [
        "## rec_inter.csv",
        "",
        f"- Rows: {fmt_int(rec['rows'])}",
        "- Columns: 12",
        f"- Time range: {rec['date_min']} to {rec['date_max']}",
        f"- Sample duration mean: {fmt_float(rec['duration_mean_sample'])} ms",
        f"- Sample duration median: {fmt_float(rec['duration_median_sample'])} ms",
        f"- Sample playing time mean: {fmt_float(rec['playing_mean_sample'])} ms",
        f"- Sample playing time median: {fmt_float(rec['playing_median_sample'])} ms",
        f"- Sample unique users: {fmt_int(rec['sample_unique_users'])}",
        f"- Sample unique items: {fmt_int(rec['sample_unique_items'])}",
        "",
        "### Positive action rates",
        "",
        md_table_from_dict(rec["action_rates"], "action", "rate"),
        "",
    ]
    for fig in rec["figures"]:
        lines += [f"![{fig}](figures/{fig})", ""]

    src = results["src_inter"]
    lines += [
        "## src_inter.csv",
        "",
        f"- Rows: {fmt_int(src['rows'])}",
        "- Columns: 9",
        f"- Time range: {src['date_min']} to {src['date_max']}",
        f"- Sample average click_cnt: {fmt_float(src['avg_click_cnt_sample'])}",
        f"- Sample median click_cnt: {fmt_float(src['median_click_cnt_sample'])}",
        f"- Sample average keyword token length: {fmt_float(src['avg_keyword_token_len_sample'])}",
        f"- Sample median keyword token length: {fmt_float(src['median_keyword_token_len_sample'])}",
        f"- Sample unique (user_id, session_id) pairs: {fmt_int(src['sample_unique_sessions'])}",
        "",
        "### Search session source counts",
        "",
        md_table_from_dict(src["source_counts"], "source", "count"),
        "",
        "### Search item type counts",
        "",
        md_table_from_dict(src["item_type_counts"], "item_type", "count"),
        "",
    ]
    for fig in src["figures"]:
        lines += [f"![{fig}](figures/{fig})", ""]

    lines += [
        "## Consolidated Findings",
        "",
        "- `item_features.csv` is the largest entity table and shows a heavy concentration in a few high-level categories, while captions are highly variable in token length.",
        "- `rec_inter.csv` is a dense interaction log with very low positive engagement rates relative to exposure volume, which is typical for implicit-feedback recommendation data.",
        "- `src_inter.csv` captures search-result impressions rather than only clicks, so `click_cnt` should be treated as a relevance/engagement label, not a row existence indicator.",
        "- `social_network.csv` is tiny compared with the interaction tables, suggesting it is likely an auxiliary graph feature rather than a core supervision source.",
        "- `user_features.csv` appears to contain compact categorical/ordinal profile features rather than rich demographics.",
        "",
        "## Output",
        "",
        f"- Markdown report: `{REPORT_PATH}`",
        f"- Figures: `{FIG_DIR}`",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    results = {
        "user_features": analyze_user_features(DATA_DIR / "user_features.csv"),
        "social_network": analyze_social_network(DATA_DIR / "social_network.csv"),
        "item_features": analyze_item_features(DATA_DIR / "item_features.csv"),
        "rec_inter": analyze_rec_inter(DATA_DIR / "rec_inter.csv"),
        "src_inter": analyze_src_inter(DATA_DIR / "src_inter.csv"),
    }

    report = build_report(results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
