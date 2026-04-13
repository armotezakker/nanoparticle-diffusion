"""
van_hove.py
-----------
Self part of the Van Hove correlation function and the non-Gaussian parameter
alpha_2. These quantities diagnose dynamic heterogeneity and non-Fickian
transport in nanoparticle systems.
"""

import numpy as np
import pandas as pd

from .trajectory_analysis import unwrap_trajectory

DT_SAVED = 0.02


def van_hove_self(traj_df: pd.DataFrame,
                  lag_steps: int,
                  n_bins: int = 80,
                  r_max: float = 8.0) -> tuple:
    """Compute the self part of the Van Hove correlation function.

    G_s(r, tau) = (1/N) * sum_t delta(r - |r(t+tau) - r(t)|)

    For free Brownian motion in 3D, G_s is a Maxwell-Boltzmann distribution
    (the distribution of displacement magnitudes of a 3D Gaussian process):
        G_s(r, tau) = 4 pi r^2 * (1 / (4 pi D tau))^(3/2)
                          * exp(-r^2 / (4 D tau))

    Deviations from this form signal non-Fickian dynamics:
      - A sharp central peak indicates a caged particle that rarely moves.
      - Heavy tails or a secondary peak at large r indicate hopping events
        where the particle occasionally makes a large displacement, a hallmark
        of discontinuous, activated transport in a porous medium.
      - A broadly elevated tail (elevated alpha_2) reflects dynamic heterogeneity:
        some time origins find the particle mobile, others find it trapped.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Single trajectory DataFrame.
    lag_steps : int
        Lag time in frames.
    n_bins : int
        Number of histogram bins. Default 80.
    r_max : float
        Maximum displacement to include (sigma units). Default 8.0.

    Returns
    -------
    bin_centres : np.ndarray, shape (n_bins,)
    density     : np.ndarray, shape (n_bins,)
        Probability density (integral over r = 1).
    """
    r = unwrap_trajectory(traj_df)
    n = len(r)
    if lag_steps >= n:
        raise ValueError(f'lag_steps={lag_steps} must be less than n_frames={n}.')

    displacements = np.sqrt(np.sum((r[lag_steps:] - r[:n - lag_steps])**2, axis=1))

    counts, edges = np.histogram(displacements, bins=n_bins, range=(0.0, r_max))
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    bin_width   = edges[1] - edges[0]

    # Normalise to probability density: integral rho dr = 1
    total = counts.sum() * bin_width
    density = counts / total if total > 0 else counts.astype(float)

    return bin_centres, density


def non_gaussian_parameter(traj_df: pd.DataFrame, lag_steps: int) -> float:
    """Compute the non-Gaussian parameter alpha_2 at a given lag time.

    alpha_2 = <r^4> / (3 * <r^2>^2) - 1

    where r is the displacement magnitude at lag tau. For a 3D Gaussian
    displacement distribution (free diffusion) alpha_2 = 0 exactly. Positive
    values signal dynamic heterogeneity: the distribution has heavier tails
    than a Gaussian, meaning a subset of time origins produce unusually large
    displacements. This is the signature of intermittent hopping or cage
    trapping with occasional escape.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Single trajectory DataFrame.
    lag_steps : int
        Lag time in frames.

    Returns
    -------
    float
        alpha_2 value. Returns 0.0 if there are fewer than 2 displacements.
    """
    r_unwrapped = unwrap_trajectory(traj_df)
    n = len(r_unwrapped)
    if lag_steps >= n:
        return 0.0

    displacements = r_unwrapped[lag_steps:] - r_unwrapped[:n - lag_steps]
    r2 = np.sum(displacements**2, axis=1)  # |r|^2 for each time origin

    mean_r2 = np.mean(r2)
    mean_r4 = np.mean(r2**2)

    if mean_r2 == 0:
        return 0.0

    return mean_r4 / (3.0 * mean_r2**2) - 1.0


def alpha2_vs_lag(traj_df: pd.DataFrame, lag_values: list) -> pd.Series:
    """Compute alpha_2 at each lag in lag_values.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Single trajectory DataFrame.
    lag_values : list of int
        Lag times in frames at which to evaluate alpha_2.

    Returns
    -------
    pd.Series
        alpha_2 values indexed by physical time (lag * DT_SAVED).
    """
    results = {}
    for lag in lag_values:
        tau = lag * DT_SAVED
        results[tau] = non_gaussian_parameter(traj_df, lag)

    return pd.Series(results, name='alpha_2')
