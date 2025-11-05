import numpy as np

def QIS(Sample, ddof=1):
    """
    Compute the Quadratic-Inverse Shrinkage (QIS) estimator for a given sample.
    Parameters:
    Sample (numpy.ndarray): A 2D array where rows represent the stocks and columns represent the time series.
    ddof (int, optional): Degrees of freedom correction for the sample mean. Default is 1: Mean is subtracted.
    Returns:
    numpy.ndarray: The QIS estimator of the covariance matrix.
    Notes:
    - The input sample is expected to be a NumPy array.
    - The function ensures the symmetry of the sample covariance matrix.
    - Eigenvalues are clipped to be non-negative.
    - The function uses a smoothing parameter for the shrinkage estimator.
    - The trace of the covariance matrix is preserved in the final estimator.
    """

    # Y is expected to be a NumPy array
    Sample = Sample.T
    N, p = Sample.shape  # Get dimensions of Y

    # Default setting: if k is None or NaN, set k = 1 (no de-mean)
    if ddof>=1:
        Sample=Sample -np.mean(Sample, axis=0)

    # Vars
    n = N - ddof   # Adjust effective sample size
    c = p / n   # Concentration ratio

    # Compute sample covariance matrix
    sample = np.matmul(Sample.T, Sample) / n
    sample = (sample + sample.T) / 2  # Ensure symmetry

    # Eigenvalue decomposition (use eigh for Hermitian matrix)
    lambda1, u = np.linalg.eigh(sample)
    lambda1 = np.clip(lambda1.real, a_min=0, a_max=None)  # Clip negative eigenvalues to 0

    # Compute Quadratic-Inverse Shrinkage estimator
    h = (min(c**2, 1/c**2)**0.35) / p**0.35  # Smoothing parameter

    # Inverse of (non-null) eigenvalues
    invlambda = 1 / lambda1[max(1, p - n + 1) - 1:p]

    # Calculate Lj and Lj_i (Differences of inverse eigenvalues)
    Lj = np.repeat(invlambda[:, np.newaxis], min(p, n), axis=1)
    Lj_i = Lj - Lj.T

    # Smoothed Stein shrinker (theta) and its conjugate (Htheta)
    Lj_squared = Lj * Lj
    theta = np.mean(Lj * Lj_i / (Lj_i * Lj_i + Lj_squared * h**2), axis=0)
    Htheta = np.mean(Lj * Lj * h / (Lj_i * Lj_i + Lj_squared * h**2), axis=0)
    Atheta2 = theta**2 + Htheta**2  # Squared amplitude

    # Shrink eigenvalues based on p and n
    if p <= n:
        delta = 1 / ((1 - c)**2 * invlambda + 2 * c * (1 - c) * invlambda * theta + c**2 * invlambda * Atheta2)
    else:
        delta0 = 1 / ((c - 1) * np.mean(invlambda))  # Shrinkage of null eigenvalues
        delta = np.concatenate((
            np.full(p - n, delta0), 
            1 / (invlambda * Atheta2)
        ))

    # Preserve the trace
    deltaQIS = delta * (np.sum(lambda1) / np.sum(delta))

    # Reconstruct covariance matrix
    sigmahat = np.matmul(u, np.matmul(np.diag(deltaQIS), u.T))

    return sigmahat
