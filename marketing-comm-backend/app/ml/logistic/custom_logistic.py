import numpy as np


class CustomLogisticRegression:
    # конструктор класса:
    def __init__(
        self,
        learning_rate: float = 0.01, # шаг градиентного спуска (насколько сильно мы двигаем веса на каждой итерации)
        max_iter: int = 1000,
        l2_lambda: float = 0.0, # коэффициент L2-регуляризации
        class_weight: dict | str | None = None,
        tolerance: float = 1e-6, # критерий ранней остановки
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2_lambda = l2_lambda
        self.class_weight = class_weight
        self.tolerance = tolerance

        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones_like(y, dtype=float)

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

    def _compute_loss(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sample_weights: np.ndarray,
    ) -> float:
        eps = 1e-15
        y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)

        weighted_log_loss = -np.mean(
            sample_weights
            * (
                y_true * np.log(y_pred_proba)
                + (1 - y_true) * np.log(1 - y_pred_proba)
            )
        )

        l2_penalty = (self.l2_lambda / (2 * len(y_true))) * np.sum(self.weights ** 2)

        return float(weighted_log_loss + l2_penalty)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        sample_weights = self._get_sample_weights(y)
        previous_loss = None

        for _ in range(self.max_iter):
            linear_output = X @ self.weights + self.bias
            y_pred_proba = self._sigmoid(linear_output)

            errors = (y_pred_proba - y) * sample_weights

            dw = (X.T @ errors) / n_samples
            db = np.sum(errors) / n_samples

            if self.l2_lambda > 0:
                dw += (self.l2_lambda / n_samples) * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            current_loss = self._compute_loss(
                y_true=y,
                y_pred_proba=y_pred_proba,
                sample_weights=sample_weights,
            )
            self.loss_history.append(current_loss)

            if previous_loss is not None and abs(previous_loss - current_loss) < self.tolerance:
                break

            previous_loss = current_loss

        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        linear_output = X @ self.weights + self.bias
        positive_class_proba = self._sigmoid(linear_output)
        negative_class_proba = 1.0 - positive_class_proba

        return np.column_stack([negative_class_proba, positive_class_proba])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)

    @property
    def coef_(self) -> np.ndarray:
        return np.array([self.weights])

    @property
    def intercept_(self) -> np.ndarray:
        return np.array([self.bias])
