from __future__ import annotations

TASKS = {
    "like": "like", "search": "search", "follow": "follow", "forward": "forward",
    "like_end_30m": "like_end_30m", "search_end_30m": "search_end_30m",
    "follow_end_30m": "follow_end_30m", "forward_end_30m": "forward_end_30m",
}

ACTION_END_TASKS = {
    "like_end_30m": "like", "search_end_30m": "search",
    "follow_end_30m": "follow", "forward_end_30m": "forward",
}
GRAPH_ACTION_FOR_TASK = {
    "like": "like", "search": "search", "follow": "follow", "forward": "forward",
    **ACTION_END_TASKS,
}
END_THRESHOLD_SECONDS = 1_800

# Post-interaction signals (playing_time, duration_ms) are intentionally excluded:
# they are unavailable before an impression and would leak behavior labels.
CATEGORICAL_FEATURES = [
    "user_id", "item_id", "onehot_feat1", "onehot_feat2", "search_active_level",
    "rec_active_level", "first_level_category_name_en", "second_level_category_name_en",
    "item_type", "upload_type", "event_hour", "event_dow",
]

DEFAULT_HASH_BUCKETS = {
    "user_id": 1_000_003, "item_id": 2_000_003,
    "first_level_category_name_en": 257, "second_level_category_name_en": 2_003,
    "item_type": 17, "upload_type": 67, "onehot_feat1": 67, "onehot_feat2": 67,
    "search_active_level": 67, "rec_active_level": 67, "event_hour": 25, "event_dow": 8,
}
