import numpy as np


class CustomLogisticRegression:
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        l2_lambda: float = 0.0,
        class_weight: dict | str | None = None,
        tolerance: float = 1e-6,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2_lambda = l2_lambda
        self.class_weight = class_weight
        self.tolerance = tolerance

        # Параметры модели (заполнятся при обучении)
        self.weights = None
        self.bias = 0.0
        self.loss_history = [] # история значений функции потерь


    def fit(self, X, y):
        """
        Обучает модель логистической регрессии градиентным спуском.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        # Инициализация весов нулями
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        sample_weights = self._get_sample_weights(y) # вычисляем веса объектов
        previous_loss = None # для проверки надобности ранней остановки

        # Основной цикл градиентного спуска
        for _ in range(self.max_iter):
            # 1. Линейная комбинация: z = X·w + b (@ - матричное умножение)
            linear_output = X @ self.weights + self.bias
            # 2. Предсказание вероятности: p = σ(z)
            y_pred_proba = self._sigmoid(linear_output)
            # 3. Взвешенная ошибка = разница между предсказанием и истиной
            errors = (y_pred_proba - y) * sample_weights

            # 4. Градиенты:
            #    dw = (1/n) * X.T @ errors
            #    db = (1/n) * Σ errors
            dw = (X.T @ errors) / n_samples # X.T - транспонированная матрица X
            db = np.sum(errors) / n_samples

            # 5. Добавляем градиент от L2-регуляризации: ∂(L2)/∂w = (λ/n) * w
            if self.l2_lambda > 0:
                dw += (self.l2_lambda / n_samples) * self.weights

            # 6. Обновляем веса и смещение
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 7. Вычисляем текущее значение функции потерь для мониторинга
            updated_linear_output = X @ self.weights + self.bias
            updated_y_pred_proba = self._sigmoid(updated_linear_output)

            current_loss = self._compute_loss(
                y_true=y,
                y_pred_proba=updated_y_pred_proba,
                sample_weights=sample_weights,
            )
            self.loss_history.append(current_loss)

            # 8. Ранняя остановка: если loss почти не меняется — выходим
            if previous_loss is not None and abs(previous_loss - current_loss) < self.tolerance:
                break

            previous_loss = current_loss

        return self
    

    def _get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones_like(y, dtype=float) # возвращает массив, заполненный единицами

        if isinstance(self.class_weight, dict):
            weight_0 = self.class_weight.get(0, 1.0)
            weight_1 = self.class_weight.get(1, 1.0)
            return np.where(y == 1, weight_1, weight_0).astype(float)

        if self.class_weight == "balanced":
            n_samples = len(y)
            n_positive = np.sum(y == 1)
            n_negative = np.sum(y == 0)

            weight_0 = n_samples / (2 * max(n_negative, 1))
            weight_1 = n_samples / (2 * max(n_positive, 1))

            return np.where(y == 1, weight_1, weight_0).astype(float)

        return np.ones_like(y, dtype=float)
    

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500) # np.clip ограничивает значения в массиве указанным диапазоном, чтобы не было переполнения памяти из-за огромных значений
        return 1.0 / (1.0 + np.exp(-z))
    

    def _compute_loss(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sample_weights: np.ndarray,
    ) -> float:
        eps = 1e-15
        y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps) # ограничиваем значения массива, чтобы не было log(0)

        weighted_log_loss = -np.mean( # среднее арифметическое 
            sample_weights
            * (
                y_true * np.log(y_pred_proba)
                + (1 - y_true) * np.log(1 - y_pred_proba)
            )
        )

        l2_penalty = (self.l2_lambda / (2 * len(y_true))) * np.sum(self.weights ** 2) # L_reg = (λ / (2n)) * Σ w_j²

        return float(weighted_log_loss + l2_penalty)


    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """
        Возвращает предсказанные классы (0 или 1).
        """
        probabilities = self.predict_proba(X)[:, 1] # взять все строки, но только столбец с индексом 1
        return (probabilities >= threshold).astype(int)


    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        # Линейная комбинация
        linear_output = X @ self.weights + self.bias
        # Вероятность класса 1 
        positive_class_proba = self._sigmoid(linear_output)
        # Вероятность класса 0
        negative_class_proba = 1.0 - positive_class_proba

        # Склеиваем в один массив: [P(класс=0), P(класс=1)]
        return np.column_stack([negative_class_proba, positive_class_proba])

    
    @property
    def coef_(self) -> np.ndarray:
        """
        Возвращает веса модели в формате, совместимом со sklearn.
        """
        return np.array([self.weights])

    @property
    def intercept_(self) -> np.ndarray:
        """
        Возвращает bias (смещение) модели.
        """
        return np.array([self.bias])
