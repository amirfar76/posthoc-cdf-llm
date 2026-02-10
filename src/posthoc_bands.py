from __future__ import annotations
import numpy as np
from typing import Tuple

def eps_sqrt_calibrator(n: int, K: int, S: int, delta: float) -> float:
    # Eq (width_used): eps = sqrt( (1/(2n)) * log( 8K^2/(delta^2 S^2) ) )
    return float(np.sqrt((1.0/(2*n)) * np.log((8*(K**2))/((delta**2)*(max(S,1)**2)))))

def cdf_band_from_samples(samples: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns (x_grid, L(x), U(x)) for empirical CDF band on sample points.
    x = np.sort(samples)
    n = len(x)
    # empirical CDF values at sample points (right-continuous ECDF)
    Fhat = np.arange(1, n+1) / n
    L = np.clip(Fhat - eps, 0.0, 1.0)
    U = np.clip(Fhat + eps, 0.0, 1.0)
    return x, L, U
