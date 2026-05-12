import numpy as np

from .custom_tree import CustomRegressionTree


class CustomGradientBoostingClassifier:
    def __init__(
        self,
        iterations: int = 100,
        learning_rate: float = 0.05,
        depth: int = 3,
        l2_leaf_reg: float = 0.0,
        class_weights=None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.class_weights = class_weights
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.base_score = 0.0
        self.trees = []
        self.loss_history = []


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        sample_weights = self._get_sample_weights(y)

        positive_rate = np.average(y, weights=sample_weights) # взвешенное среднее
        positive_rate = np.clip(positive_rate, 1e-6, 1 - 1e-6)
        self.base_score = float(np.log(positive_rate / (1 - positive_rate)))

        raw_scores = np.full(len(y), self.base_score, dtype=float)

        self.trees = []
        self.loss_history = []

        for _ in range(self.iterations):
            probabilities = self._sigmoid(raw_scores)
            residuals = y - probabilities

            tree = CustomRegressionTree(
                max_depth=self.depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                l2_leaf_reg=self.l2_leaf_reg,
            )
            tree.fit(X, residuals, sample_weights=sample_weights)
            updates = tree.predict(X)

            raw_scores += self.learning_rate * updates
            self.trees.append(tree)

            current_loss = self._compute_loss(
                y_true=y,
                y_pred_proba=self._sigmoid(raw_scores),
                sample_weights=sample_weights,
            )
            self.loss_history.append(current_loss)

        return self


    def _get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weights is None:
            return np.ones_like(y, dtype=float)

        if isinstance(self.class_weights, (list, tuple)) and len(self.class_weights) == 2:
            weight_0 = float(self.class_weights[0])
            weight_1 = float(self.class_weights[1])
            return np.where(y == 1, weight_1, weight_0).astype(float)

        return np.ones_like(y, dtype=float)
    

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    

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

        return float(weighted_log_loss)

    
    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        raw_scores = np.full(X.shape[0], self.base_score, dtype=float)
        for tree in self.trees:
            raw_scores += self.learning_rate * tree.predict(X)

        positive_class_proba = self._sigmoid(raw_scores)
        negative_class_proba = 1.0 - positive_class_proba

        return np.column_stack([negative_class_proba, positive_class_proba])


    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)
