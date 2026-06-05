import numpy as np
import pandas as pd

from app.ml.config import CATEGORICAL_FEATURES, ONE_HOT_CONFIG
from app.ml.encoding import (
    fit_transform_one_hot,
    transform_one_hot,
    align_feature_columns,
)


class CustomBoostingPipeline:
    def __init__(self, model, merge_rare_categories: bool = False):
        self.model = model
        self.merge_rare_categories = merge_rare_categories
        self.encoder_mapping = None
        self.feature_columns = None


    def fit(self, X: pd.DataFrame, y):
        X_transformed, self.encoder_mapping = fit_transform_one_hot(
            df=X,
            categorical_columns=CATEGORICAL_FEATURES,
            merge_rare_categories=self.merge_rare_categories,
            rare_category_columns=ONE_HOT_CONFIG["rare_category_columns"],
            min_frequency=ONE_HOT_CONFIG["min_frequency"],
            drop_first=ONE_HOT_CONFIG["drop_first"],
        )

        self.feature_columns = list(X_transformed.columns)

        X_transformed = np.asarray(X_transformed, dtype=float)
        self.model.fit(X_transformed, y)

        return self
    

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        if self.encoder_mapping is None:
            raise ValueError("Encoder mapping не найден. Pipeline не был обучен.")

        if self.feature_columns is None:
            raise ValueError("Список признаков не найден. Pipeline не был обучен.")

        X_transformed = transform_one_hot(
            df=X,
            encoder_mapping=self.encoder_mapping,
        )

        X_transformed = align_feature_columns(
            df=X_transformed,
            feature_columns=self.feature_columns,
        )

        return np.asarray(X_transformed, dtype=float)


    def predict_proba(self, X: pd.DataFrame):
        X_transformed = self.transform_features(X)
        return self.model.predict_proba(X_transformed)


    def predict(self, X: pd.DataFrame, threshold: float = 0.5):
        X_transformed = self.transform_features(X)
        return self.model.predict(X_transformed, threshold=threshold)
