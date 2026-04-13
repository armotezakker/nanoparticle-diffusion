"""
msd.py
------
Mean squared displacement computation, power-law fitting, and ensemble
averaging. All positions must be PBC-unwrapped before computing MSD.
"""

import numpy as np
import pandas as pd

from .trajectory_analysis import unwrap_trajectory

# Physical time step between saved frames: dt_sim * save_interval = 0.002 * 10
DT_SAVED = 0.02


def compute_msd(traj_df: pd.DataFrame,
                max_lag_fraction: float = 0.4) -> pd.DataFrame:
    """Compute the time-averaged mean squared displacement.

    Estimator:
        MSD(tau) = (1 / (N - tau)) * sum_{t=0}^{N-tau-1} |r(t+tau) - r(t)|^2

    The sum runs over all available time origins, which maximises statistical
    efficiency for a single-particle trajectory (ergodicity is assumed).

    Lag range: tau = 1 to int(N * max_lag_fraction).
    Beyond 40% of total frames the number of independent time origins becomes
    small and the estimator variance grows rapidly; using lags beyond this
    point introduces systematic underestimation of the MSD variance.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Single trajectory DataFrame.
    max_lag_fraction : float
        Maximum lag as a fraction of total trajectory length. Default 0.4.

    Returns
    -------
    pd.DataFrame
        Columns: lag (frames), tau (LJ time units), msd (sigma^2).
    """
    r = unwrap_trajectory(traj_df)  # shape (N, 3)
    n = len(r)
    max_lag = max(1, int(n * max_lag_fraction))

    lags = np.arange(1, max_lag + 1, dtype=int)
    msd_vals = np.empty(len(lags), dtype=float)

    for i, lag in enumerate(lags):
        diff = r[lag:] - r[:n - lag]   # shape (N-lag, 3)
        msd_vals[i] = np.mean(np.sum(diff**2, axis=1))

    return pd.DataFrame({
        'lag': lags,
        'tau': lags * DT_SAVED,
        'msd': msd_vals,
    })


def fit_msd(msd_df: pd.DataFrame,
            fit_range: tuple = (0.05, 0.35)) -> dict:
    """Fit MSD = 4 * D * tau^alpha by linear regression in log-log space.

    The fit window avoids the ballistic short-time regime (where MSD ~ tau^2
    due to inertia) and the noisy long-time regime (few time origins, high
    variance). The default window 5%-35% of the total lag range is a
    reasonable compromise for the trajectory lengths in this study.

    Physical interpretation of alpha:
      alpha = 1   free Brownian diffusion
      alpha < 1   subdiffusion (trapping, viscoelastic medium)
      alpha > 1   superdiffusion (active, driven transport)

    Parameters
    ----------
    msd_df : pd.DataFrame
        Output of compute_msd.
    fit_range : tuple of float
        (low_fraction, high_fraction) defining which lags to include.

    Returns
    -------
    dict with keys:
        alpha, D, alpha_err, D_err, r_squared
    """
    n = len(msd_df)
    lo = max(0, int(fit_range[0] * n))
    hi = min(n, int(fit_range[1] * n))
    sub = msd_df.iloc[lo:hi]

    log_tau = np.log10(sub['tau'].to_numpy())
    log_msd = np.log10(sub['msd'].to_numpy())

    # Degree-1 polynomial: log_msd = alpha * log_tau + log(4D)
    coeffs, cov = np.polyfit(log_tau, log_msd, deg=1, cov=True)
    alpha = coeffs[0]
    intercept = coeffs[1]
    D = 10**intercept / 4.0

    # Standard errors from covariance matrix diagonal
    alpha_err = np.sqrt(cov[0, 0])
    D_err     = np.sqrt(cov[1, 1]) * D * np.log(10)  # propagated to D

    # R-squared in log space
    log_msd_pred = np.polyval(coeffs, log_tau)
    ss_res = np.sum((log_msd - log_msd_pred)**2)
    ss_tot = np.sum((log_msd - log_msd.mean())**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'alpha'    : alpha,
        'D'        : D,
        'alpha_err': alpha_err,
        'D_err'    : D_err,
        'r_squared': r_squared,
    }


def ensemble_msd(trajs: dict,
                 conc: float,
                 charge: float) -> tuple:
    """Compute ensemble-averaged MSD over all runs at (conc, charge).

    Parameters
    ----------
    trajs : dict
        Full trajectories dict, keys (conc, charge, run_id).
    conc : float
        Polymer concentration to select.
    charge : float
        NP charge to select.

    Returns
    -------
    mean_msd_df : pd.DataFrame
        Mean MSD across runs, indexed by lag.
    std_msd_df : pd.DataFrame
        Standard deviation across runs, indexed by lag.
    """
    run_msds = []
    for key, df in trajs.items():
        if key[0] == conc and key[1] == charge:
            msd_df = compute_msd(df)
            run_msds.append(msd_df.set_index('lag')['msd'])

    if not run_msds:
        raise ValueError(f'No trajectories found for conc={conc}, charge={charge}.')

    combined = pd.concat(run_msds, axis=1)
    mean_msd = combined.mean(axis=1).reset_index()
    std_msd  = combined.std(axis=1).reset_index()

    # Recover tau column from the first run_msd
    ref = compute_msd(next(
        df for key, df in trajs.items() if key[0] == conc and key[1] == charge
    ))
    lag_to_tau = ref.set_index('lag')['tau']

    mean_msd.columns = ['lag', 'msd']
    std_msd.columns  = ['lag', 'msd']
    mean_msd['tau']  = mean_msd['lag'].map(lag_to_tau)
    std_msd['tau']   = std_msd['lag'].map(lag_to_tau)

    return mean_msd, std_msd
