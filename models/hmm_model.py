import pickle
import numpy as np
from hmmlearn.hmm import GaussianHMM


class RegimeHMM:
    def __init__(self, n_regimes=4, covariance_type="full", n_iter=1000, random_state=42):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self._model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray) -> "RegimeHMM":
        self._model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, posteriors = self._model.score_samples(X)
        return posteriors

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "RegimeHMM":
        with open(path, "rb") as f:
            return pickle.load(f)
