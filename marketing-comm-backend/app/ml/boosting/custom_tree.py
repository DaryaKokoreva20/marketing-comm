from __future__ import annotations

import numpy as np


class TreeNode:
    def __init__(
        self,
        feature_index: int | None = None,
        threshold: float | None = None,
        left: "TreeNode | None" = None,
        right: "TreeNode | None" = None,
        value: float | None = None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class CustomRegressionTree:
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        l2_leaf_reg: float = 0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.l2_leaf_reg = l2_leaf_reg
        self.root: TreeNode | None = None

    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if sample_weights is None:
            sample_weights = np.ones(len(y), dtype=float)
        else:
            sample_weights = np.asarray(sample_weights, dtype=float)

        self.root = self._build_tree(
            X=X,
            y=y,
            sample_weights=sample_weights,
            depth=0,
        )
        return self


    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        depth: int,
    ) -> TreeNode:
        n_samples = len(y)

        if n_samples == 0:
            return TreeNode(value=0.0)

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.allclose(y, y[0])
        ):
            return TreeNode(
                value=self._weighted_leaf_value(y, sample_weights)
            )
        
        best_feature, best_threshold = self._find_best_split(
            X=X,
            y=y,
            sample_weights=sample_weights,
        )

        if best_feature is None or best_threshold is None:
            return TreeNode(
                value=self._weighted_leaf_value(y, sample_weights)
            )

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_node = self._build_tree(
            X[left_mask],
            y[left_mask],
            sample_weights[left_mask],
            depth + 1,
        )
        right_node = self._build_tree(
            X[right_mask],
            y[right_mask],
            sample_weights[right_mask],
            depth + 1,
        )

        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
        )
    
    
    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
    ) -> tuple[int | None, float | None]:
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_score = np.inf

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            if len(unique_values) < 2:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_leaf:
                    continue
                if right_mask.sum() < self.min_samples_leaf:
                    continue

                left_score = self._weighted_sse(
                    y[left_mask],
                    sample_weights[left_mask],
                )
                right_score = self._weighted_sse(
                    y[right_mask],
                    sample_weights[right_mask],
                )

                total_score = left_score + right_score

                if total_score < best_score:
                    best_score = total_score
                    best_feature = feature_index
                    best_threshold = float(threshold)

        return best_feature, best_threshold
    
    
    def _weighted_sse(
        self,
        y: np.ndarray,
        sample_weights: np.ndarray,
    ) -> float:
        if len(y) == 0:
            return 0.0

        mean_value = self._weighted_leaf_value(y, sample_weights)
        return float(np.sum(sample_weights * (y - mean_value) ** 2))


    def _weighted_leaf_value(
        self,
        y: np.ndarray,
        sample_weights: np.ndarray,
    ) -> float:
        numerator = np.sum(sample_weights * y)
        denominator = np.sum(sample_weights) + self.l2_leaf_reg
        if denominator <= 0:
            return 0.0
        return float(numerator / denominator)


    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("Дерево не обучено.")

        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root) for row in X], dtype=float)
    

    def _predict_row(self, row: np.ndarray, node: TreeNode) -> float:
        if node.is_leaf:
            return float(node.value)

        if row[node.feature_index] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)
    