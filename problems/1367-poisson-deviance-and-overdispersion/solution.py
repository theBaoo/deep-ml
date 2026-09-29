import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    # Your code here
    term = np.where(
        y == 0,
        mu,
        y * np.log(y / mu) - y + mu
    )
    return 2 * term.sum(axis=-1)


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    # Your code here
    n = y.shape[-1]
    return (
        ((y - mu) ** 2 / mu).sum(axis=-1)
    ) / (n - n_params)
