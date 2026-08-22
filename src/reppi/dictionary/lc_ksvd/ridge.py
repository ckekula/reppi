import numpy as np


class RidgeClassifier:
    """
    Default LC-KSVD1 classifier: linear predictive classifier trained by
    ridge regression, per Jiang et al. 2011, Eq. (17):

        W = H X^T (X X^T + lambda1 * I)^-1

    Any object exposing ``fit(Gamma, H)`` / ``predict(Gamma)`` with this
    signature can be passed as ``LCKSVD(..., classifier=...)`` in place of
    this default. Only meaningful for ``variant="lcksvd1"`` — LC-KSVD2's
    classifier is trained jointly with the dictionary (it's a block of the
    augmented system) and cannot be swapped out independently.

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
        """
        Parameters
        ----------
        Gamma : np.ndarray, shape (n_components, n_samples)
        H : np.ndarray, shape (n_classes, n_samples)  one-hot

        Returns
        -------
        self
        """
        n_components = Gamma.shape[0]
        gram = Gamma @ Gamma.T + self.lambda1 * np.eye(n_components)
        self.W_ = H @ Gamma.T @ np.linalg.inv(gram)
        return self

    def predict(self, Gamma: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        Gamma : np.ndarray, shape (n_components, n_samples)

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)
        """
        if self.W_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.argmax(self.W_ @ Gamma, axis=0)
