THRESHOLD_CANDIDATES = [0.40] # [0.35, 0.4, 0.45, 0.50, 0.55]

MODEL_SELECTION_CONFIG = {
    "min_precision": 0.35,
    "min_recall": 0.4,
    "primary_metric": "f1",
    "secondary_metric": "recall",
}

DATA_SPLIT_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.2,
    "random_state": 42,
    "stratify": True,
}

ONE_HOT_CONFIG = {
    "rare_category_columns": ["industry_category", "channel_scenario"],
    "min_frequency": 10,
    "drop_first": True,
}

LOGISTIC_SEARCH_CONFIG = {
    "params": {
        "class_weight": [
            {0: 1, 1: 5},
        ], # None, {0: 1, 1: 3}, {0: 1, 1: 5},
        "use_scaler": [True], # [False, True]
        "learning_rate": [0.03], # [0.01, 0.03, 0.05],
        "max_iter": [1000], # [500, 1000, 3000, 5000]
        "l2_lambda": [0.01], # [0.0, 0.01, 0.1]
        "tolerance": [1e-6],
        "merge_rare_categories": [True] # [False, True]
    }
}

BOOSTING_SEARCH_CONFIG = {
    "enabled": True,
    "class_weights": [
        [1, 5],
        # [1, 5],
        # [1, 7],
    ],
    "params": {
        "iterations": [500], # [500, 800, 1200],
        "learning_rate": [0.03], # 0.01
        "depth": [4], # 5
        "l2_leaf_reg": [3], # 5, 7
        "merge_rare_categories": [True], # False
    },
}

NUMERIC_FEATURES = [
    "tenure_days",
    "avg_order_value",
    "order_frequency",
    "total_orders",
    "recency_days",
    "repeat_purchase_propensity",
    "prev_response_rate",
    "prev_comm_count_channel",
    "prev_comm_response_rate_channel",
    "last_comm_days_channel",
    "prev_comm_count_scenario",
    "prev_comm_response_rate_scenario",
    "promo_sensitivity",
    "b2c_age",
    "time_trigger",
    "comm_time_sin",
    "comm_time_cos",
    "orders_per_30_days",
    "recency_to_tenure_ratio",
    "channel_fatigue_ratio",
    # "comm_weekday",
    # "is_weekend",
    # "avg_order_value_squared",
    # "random_noise_feature",
    # "recency_days_squared",
    # "order_frequency_squared",
    # "total_orders_squared",
    # "last_comm_days_channel_squared",
    # "promo_sensitivity_squared",
    # "recency_days_cubed",
    # "order_frequency_cubed",
    # "total_orders_cubed",
    # "last_comm_days_channel_cubed",
    # "promo_sensitivity_cubed",
]

CATEGORICAL_FEATURES = [
    "client_type",
    "industry_category",
    "channel_type",
    "scenario_type",
    "channel_scenario",
]

BOOSTING_FEATURES = [
    "client_type",
    "tenure_days",
    "avg_order_value",
    "order_frequency",
    "total_orders",
    "recency_days",
    "repeat_purchase_propensity",
    "industry_category",
    "channel_type",
    "scenario_type",
    "prev_response_rate",
    "time_trigger",
    "prev_comm_count_channel",
    "prev_comm_response_rate_channel",
    "last_comm_days_channel",
    "prev_comm_count_scenario",
    "prev_comm_response_rate_scenario",
    "promo_sensitivity",
    "b2c_age",
    "b2c_age_not_applicable",
    "comm_time_sin",
    "comm_time_cos",
    "orders_per_30_days",
    "channel_scenario",
    "comm_intensity",
    "channel_effectiveness",
    "is_morning",
    "is_evening",
]

NON_NEGATIVE_NUMERIC_COLUMNS = [
    "tenure_days",
    "avg_order_value",
    "order_frequency",
    "total_orders",
    "recency_days",
    "prev_comm_count_channel",
    "last_comm_days_channel",
    "prev_comm_count_scenario",
]

RATIO_COLUMNS = [
    "repeat_purchase_propensity",
    "prev_response_rate",
    "prev_comm_response_rate_channel",
    "prev_comm_response_rate_scenario",
    "promo_sensitivity",
]

TRAIN_BINARY_COLUMNS = [
    "time_trigger",
    "target",
]

PREDICT_BINARY_COLUMNS = [
    "time_trigger",
]
