import numpy as np


class RidgeClassifier:
    """
    Default LC-KSVD1 classifier: linear predictive classifier trained by
    ridge regression, per Jiang et al. 2011, Eq. (17)

    Any object exposing ``fit(Gamma, H)`` / ``predict(Gamma)`` with this
    signature can be passed as ``LCKSVD(..., classifier=...)`` in place of
    this default. Only meaningful for ``variant="lcksvd1"``.

    Parameters
    ----------
    lambda1 : float
        Ridge regularisation weight (default 1e-5, matching the paper's
        reported setting for the incremental-learning experiments, Sec. 5).

    Attributes
    ----------
    W_ : np.ndarray, shape (n_classes, n_components)
        Fitted classifier weights (set after ``fit()``).
    """

    def __init__(self, lambda1: float = 1e-5) -> None:
        self.lambda1 = lambda1
        self.W_: np.ndarray | None = None

    def fit(self, Gamma: np.ndarray, H: np.ndarray) -> "RidgeClassifier":
        n_components = Gamma.shape[0]
        gram = Gamma @ Gamma.T + self.lambda1 * np.eye(n_components)
        self.W_ = H @ Gamma.T @ np.linalg.inv(gram)
        return self

    def predict(self, Gamma: np.ndarray) -> np.ndarray:
        if self.W_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.argmax(self.W_ @ Gamma, axis=0)
