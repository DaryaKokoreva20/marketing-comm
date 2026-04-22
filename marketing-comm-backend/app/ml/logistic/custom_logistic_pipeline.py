import numpy as np
import pandas as pd

from app.ml.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ONE_HOT_CONFIG
from app.ml.encoding import (
    fit_transform_one_hot,
    transform_one_hot,
    align_feature_columns,
)


class CustomLogisticPipeline:
    def __init__(
        self,
        model,
        use_scaler: bool = False,
        merge_rare_categories: bool = False,
    ):
        self.model = model
        self.use_scaler = use_scaler
        self.merge_rare_categories = merge_rare_categories

        self.encoder_mapping = None
        self.feature_columns = None

        self.numeric_feature_columns = None
        self.numeric_medians = None
        self.scaler_mean = None
        self.scaler_std = None

    def _fit_numeric_imputer(self, X: pd.DataFrame) -> pd.DataFrame:
        result_df = X.copy()

        self.numeric_feature_columns = [
            column for column in NUMERIC_FEATURES
            if column in result_df.columns
        ]

        if not self.numeric_feature_columns:
            self.numeric_medians = pd.Series(dtype=float)
            return result_df

        self.numeric_medians = result_df[self.numeric_feature_columns].median()
        result_df[self.numeric_feature_columns] = result_df[self.numeric_feature_columns].fillna(
            self.numeric_medians
        )

        return result_df

    def _transform_numeric_imputer(self, X: pd.DataFrame) -> pd.DataFrame:
        result_df = X.copy()

        if self.numeric_medians is None:
            raise ValueError("Numeric imputer не был обучен.")

        if not self.numeric_feature_columns:
            return result_df

        result_df[self.numeric_feature_columns] = result_df[self.numeric_feature_columns].fillna(
            self.numeric_medians
        )

        return result_df

    def _fit_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        result_df = X.copy()

        if not self.numeric_feature_columns:
            self.scaler_mean = pd.Series(dtype=float)
            self.scaler_std = pd.Series(dtype=float)
            return result_df

        self.scaler_mean = result_df[self.numeric_feature_columns].mean()
        self.scaler_std = (
            result_df[self.numeric_feature_columns]
            .std(ddof=0)
            .replace(0, 1.0)
            .fillna(1.0)
        )

        result_df[self.numeric_feature_columns] = (
            result_df[self.numeric_feature_columns] - self.scaler_mean
        ) / self.scaler_std

        return result_df

    def _transform_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        result_df = X.copy()

        if self.scaler_mean is None or self.scaler_std is None:
            raise ValueError("Scaler не был обучен.")

        if not self.numeric_feature_columns:
            return result_df

        result_df[self.numeric_feature_columns] = (
            result_df[self.numeric_feature_columns] - self.scaler_mean
        ) / self.scaler_std

        return result_df

    def fit(self, X: pd.DataFrame, y):
        X_transformed, self.encoder_mapping = fit_transform_one_hot(
            df=X,
            categorical_columns=CATEGORICAL_FEATURES,
            merge_rare_categories=self.merge_rare_categories,
            rare_category_columns=ONE_HOT_CONFIG["rare_category_columns"],
            min_frequency=ONE_HOT_CONFIG["min_frequency"],
        )

        X_transformed = self._fit_numeric_imputer(X_transformed)

        if self.use_scaler:
            X_transformed = self._fit_scaler(X_transformed)
        else:
            self.scaler_mean = None
            self.scaler_std = None

        self.feature_columns = list(X_transformed.columns)

        X_array = np.asarray(X_transformed, dtype=float)
        self.model.fit(X_array, y)

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

        X_transformed = self._transform_numeric_imputer(X_transformed)

        if self.use_scaler:
            X_transformed = self._transform_scaler(X_transformed)

        return np.asarray(X_transformed, dtype=float)

    def predict_proba(self, X: pd.DataFrame):
        X_transformed = self.transform_features(X)
        return self.model.predict_proba(X_transformed)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5):
        X_transformed = self.transform_features(X)
        return self.model.predict(X_transformed, threshold=threshold)
