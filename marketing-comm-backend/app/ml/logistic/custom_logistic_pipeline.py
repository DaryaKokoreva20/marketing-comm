import numpy as np
import pandas as pd

from app.ml.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ONE_HOT_CONFIG
from app.ml.encoding import (
    fit_transform_one_hot,
    transform_one_hot,
    align_feature_columns,
)


class CustomLogisticPipeline:
    """
    Пайплайн подготовки признаков для кастомной логистической регрессии.

    Выполняет:
    - удаление строк с недопустимыми пропусками;
    - One-Hot кодирование категориальных признаков;
    - объединение редких категорий, если включено;
    - стандартизацию числовых признаков, если use_scaler=True;
    - обучение и применение модели.
    """

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
        self.scaler_mean = None
        self.scaler_std = None

        # Индексы строк, удаленных из-за пропусков
        self.skipped_rows = []


    def fit(self, X: pd.DataFrame, y):
        """
        Обучает пайплайн и модель.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y_array = np.asarray(y)

        if len(X) != len(y_array):
            raise ValueError(
                "Размер X и y не совпадает: "
                f"len(X)={len(X)}, len(y)={len(y_array)}."
            )

        self.numeric_feature_columns = self._detect_numeric_feature_columns(X) # записываем только существующие числовые признаки

        valid_mask = self._build_valid_rows_mask(X) # создаем маску строк, которые можно использовать

        self.skipped_rows = X.index[~valid_mask].tolist() # сохраняем индексы удаленных строк (~ = наоборот)

        # оставляем только строчки, где в valid_mask стоит True
        X_clean = X.loc[valid_mask].copy()
        y_filtered = y_array[valid_mask.to_numpy()]

        if X_clean.empty:
            raise ValueError(
                "После удаления строк с пропусками не осталось данных для обучения."
            )

        # кодируем категориальные признаки и сразу применяем 
        X_transformed, self.encoder_mapping = fit_transform_one_hot(
            df=X_clean,
            categorical_columns=CATEGORICAL_FEATURES,
            merge_rare_categories=self.merge_rare_categories,
            rare_category_columns=ONE_HOT_CONFIG["rare_category_columns"],
            min_frequency=ONE_HOT_CONFIG["min_frequency"],
        )

        # масштабируем, если включено
        if self.use_scaler:
            X_transformed = self._fit_scaler(X_transformed)
        else:
            self.scaler_mean = None
            self.scaler_std = None

        self.feature_columns = list(X_transformed.columns) # сохраняем итоговый список колонок

        # начинаем обучение
        X_array = np.asarray(X_transformed, dtype=float)
        self.model.fit(X_array, y_filtered)

        return self
    

    def _detect_numeric_feature_columns(self, X: pd.DataFrame) -> list[str]:
        """
        Определяет числовые признаки, которые реально есть в DataFrame.
        """
        return [
            column for column in NUMERIC_FEATURES if column in X.columns
        ]


    def _build_valid_rows_mask(self, X: pd.DataFrame) -> pd.Series:
        """
        Формирует маску строк без недопустимых пропусков.
        """
        if self.numeric_feature_columns is None:
            self.numeric_feature_columns = self._detect_numeric_feature_columns(X)

        valid_mask = pd.Series(True, index=X.index)

        for column in self.numeric_feature_columns:
            if column == "b2c_age":
                continue

            valid_mask &= X[column].notna() # проверяет, не пустое ли значение (& - логическое И)

        if "b2c_age" in self.numeric_feature_columns:
            if "client_type" in X.columns:
                age_valid_mask = (
                    (X["client_type"] == "B2B")
                    | X["b2c_age"].notna()
                )
            else:
                age_valid_mask = X["b2c_age"].notna()

            valid_mask &= age_valid_mask

        return valid_mask
    

    def _fit_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Обучает StandardScaler на обучающей выборке и сразу применяет его.
        """
        result_df = X.copy()

        if not self.numeric_feature_columns:
            self.scaler_mean = pd.Series(dtype=float)
            self.scaler_std = pd.Series(dtype=float)
            return result_df

        # 
        existing_numeric_columns = [
            column for column in self.numeric_feature_columns
            if column in result_df.columns
        ]

        if not existing_numeric_columns:
            self.scaler_mean = pd.Series(dtype=float)
            self.scaler_std = pd.Series(dtype=float)
            return result_df

        self.scaler_mean = result_df[existing_numeric_columns].mean() # считаем среднее
        # считаем стандартное отклонение
        self.scaler_std = (
            result_df[existing_numeric_columns]
            .std(ddof=0) # делим на n
            .replace(0, 1.0) # чтобы потом не было деления на ноль
            .fillna(1.0) # заменяет NaN в стандартных отклонениях на 1.0
        )

        result_df[existing_numeric_columns] = (
            result_df[existing_numeric_columns] - self.scaler_mean
        ) / self.scaler_std

        return result_df
    

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Применяет предобработку к новым данным.
        """
        if self.encoder_mapping is None:
            raise ValueError("Encoder mapping не найден. Pipeline не был обучен.")

        if self.feature_columns is None:
            raise ValueError("Список признаков не найден. Pipeline не был обучен.")

        if self.numeric_feature_columns is None:
            raise ValueError("Список числовых признаков не найден. Pipeline не был обучен.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # убрать строки с недопустимыми пропусками
        valid_mask = self._build_valid_rows_mask(X)
        self.skipped_rows = X.index[~valid_mask].tolist()
        X_valid = X.loc[valid_mask].copy()

        if X_valid.empty:
            return np.empty((0, len(self.feature_columns)))

        # закодировать категории по старому encoder_mapping
        X_transformed = transform_one_hot(
            df=X_valid,
            encoder_mapping=self.encoder_mapping,
        )

        # выровнять колонки под обучающий набор feature_columns
        X_transformed = align_feature_columns(
            df=X_transformed,
            feature_columns=self.feature_columns,
        )

        # применить старый scaler, если он был включён
        if self.use_scaler:
            X_transformed = self._transform_scaler(X_transformed)

        return np.asarray(X_transformed, dtype=float)


    def _transform_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет scaler, обученный на обучающей выборке.
        """
        result_df = X.copy()

        if self.scaler_mean is None or self.scaler_std is None:
            raise ValueError("Scaler не был обучен.")

        if not self.numeric_feature_columns:
            return result_df

        existing_numeric_columns = [
            column for column in self.numeric_feature_columns
            if column in result_df.columns
        ]

        if not existing_numeric_columns:
            return result_df

        result_df[existing_numeric_columns] = (
            result_df[existing_numeric_columns] - self.scaler_mean
        ) / self.scaler_std

        return result_df


    def predict_proba(self, X: pd.DataFrame):
        """
        Предсказывает вероятности классов для строк без пропусков.
        """
        X_transformed = self.transform_features(X)

        if X_transformed.shape[0] == 0:
            return np.empty((0, 2))

        return self.model.predict_proba(X_transformed)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5):
        """
        Предсказывает классы 0 или 1 для строк без пропусков.
        """
        X_transformed = self.transform_features(X)

        if X_transformed.shape[0] == 0:
            return np.array([], dtype=int)

        return self.model.predict(X_transformed, threshold=threshold)

    def get_skipped_rows_message(self) -> str | None:
        """
        Возвращает сообщение о строках, для которых прогноз не был рассчитан.
        """
        if not self.skipped_rows:
            return None

        rows = ", ".join(str(row) for row in self.skipped_rows)
        return f"У объектов {rows} есть пустые значения."
