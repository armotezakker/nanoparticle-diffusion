"""
features.py
-----------
Feature extraction from single nanoparticle trajectories and regime
classification. Each feature is physically motivated and documented inline.
The feature vector is designed to distinguish three dynamical regimes:
free diffusion, subdiffusion, and hopping transport.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from tqdm import tqdm

from .msd import compute_msd, fit_msd
from .van_hove import non_gaussian_parameter
from .trajectory_analysis import displacement_magnitudes, detect_cage_escapes

DT_SAVED = 0.02


def extract_features(traj_df: pd.DataFrame) -> dict:
    """Extract physically meaningful features from a single trajectory.

    Every feature is annotated with its physical interpretation.
    The feature set covers MSD scaling, displacement statistics,
    velocity memory (VACF), non-Gaussian dynamics (Van Hove), and
    cage-trapping events.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Single trajectory DataFrame with columns x, y, z, vx, vy, vz.

    Returns
    -------
    dict
        Flat dictionary of scalar features.
    """
    features = {}
    n_frames = len(traj_df)

    # ------------------------------------------------------------------ MSD
    msd_df = compute_msd(traj_df, max_lag_fraction=0.4)
    n_lags  = len(msd_df)

    # Overall MSD fit in the central window (5%-35% of lag range)
    fit = fit_msd(msd_df, fit_range=(0.05, 0.35))
    features['msd_alpha'] = fit['alpha']   # anomalous diffusion exponent; 1=free
    features['msd_D']     = fit['D']       # diffusion coefficient (sigma^2 / LJ time)
    features['msd_r2']    = fit['r_squared']  # quality of power-law fit

    # Short-time alpha (first 5% of lag range): captures ballistic-to-diffusive crossover
    short_hi  = max(2, int(0.05 * n_lags))
    fit_short = fit_msd(msd_df, fit_range=(0.0, max(0.05, short_hi / n_lags)))
    features['msd_short_alpha'] = fit_short['alpha']

    # Long-time alpha (last 20% of lag range): captures slow relaxation or escape
    long_lo  = max(0, int(0.80 * n_lags))
    fit_long = fit_msd(msd_df, fit_range=(max(0.75, long_lo / n_lags), 1.0))
    features['msd_long_alpha'] = fit_long['alpha']

    # Ratio > 1 means the particle speeds up at long times, a sign of cage escape
    short_a = features['msd_short_alpha']
    features['msd_ratio'] = (features['msd_long_alpha'] / short_a
                              if abs(short_a) > 1e-6 else 0.0)

    # ------------------------------------------------- Displacement statistics
    disp = displacement_magnitudes(traj_df)
    features['disp_mean']     = float(np.mean(disp))    # typical step size
    features['disp_std']      = float(np.std(disp))     # variability of step size
    # Skewness should be near 0 for Maxwellian (free diffusion); positive skew
    # indicates occasional very large steps (hopping).
    features['disp_skewness'] = float(skew(disp))
    # Excess kurtosis: 0 for Gaussian; positive (leptokurtic) signals heavy tails
    # as expected for hopping transport where most steps are small but rare
    # steps are very large.
    features['disp_kurtosis'] = float(kurtosis(disp, fisher=True))
    # Coefficient of variation: normalized variability; high CV suggests two
    # distinct motion modes (rattling vs. hopping).
    features['disp_cv']       = (features['disp_std'] / features['disp_mean']
                                  if features['disp_mean'] > 0 else 0.0)

    # ---------------------------------------- Velocity autocorrelation (VACF)
    # VACF measures velocity memory. A negative dip at short lags indicates
    # backscattering: the particle bounces off a cage wall and reverses direction.
    # vacf_lag1 < 0 is a direct signature of caged motion.
    vx = traj_df['vx'].to_numpy(dtype=float)
    vy = traj_df['vy'].to_numpy(dtype=float)
    vz = traj_df['vz'].to_numpy(dtype=float)

    var_v = np.var(vx) + np.var(vy) + np.var(vz)
    if var_v > 0:
        # Compute VACF up to lag 20 for feature extraction
        max_vacf_lag = min(20, n_frames - 1)
        vacf = np.zeros(max_vacf_lag + 1)
        for lag in range(max_vacf_lag + 1):
            if n_frames - lag < 2:
                break
            vacf[lag] = (
                np.mean(vx[:n_frames - lag] * vx[lag:])
              + np.mean(vy[:n_frames - lag] * vy[lag:])
              + np.mean(vz[:n_frames - lag] * vz[lag:])
            ) / var_v
        features['vacf_lag1'] = float(vacf[1]) if len(vacf) > 1 else 0.0
        features['vacf_lag5'] = float(vacf[5]) if len(vacf) > 5 else 0.0
        # First lag where VACF crosses zero; -1 if it never does
        crossings = np.where(np.diff(np.sign(vacf)))[0]
        features['vacf_zero_crossing'] = int(crossings[0]) if len(crossings) > 0 else -1
    else:
        features['vacf_lag1'] = 0.0
        features['vacf_lag5'] = 0.0
        features['vacf_zero_crossing'] = -1

    # ---------------------------------------------- Van Hove (lag = 50 frames)
    # alpha_2 > 0 indicates displacement distribution is non-Gaussian;
    # large values (>0.25) identify strong dynamic heterogeneity.
    vh_lag = min(50, n_frames - 1)
    features['vh_alpha2'] = float(non_gaussian_parameter(traj_df, vh_lag))

    # --------------------------------------------------- Cage escape dynamics
    escapes = detect_cage_escapes(traj_df, threshold_factor=2.5)
    n_jumps = len(escapes)
    features['n_jumps']   = int(n_jumps)         # total number of cage escapes
    features['jump_rate'] = n_jumps / n_frames   # escape rate per frame

    if n_jumps >= 2:
        inter_jump_times = np.diff(escapes)      # frames between consecutive jumps
        features['cage_time'] = float(np.mean(inter_jump_times))  # mean cage lifetime
    else:
        features['cage_time'] = 0.0

    return features


def classify_regime(features: dict) -> str:
    """Assign a diffusion regime label based on trajectory features.

    Rules (applied in order of decreasing specificity):
      1. Free diffusion  : alpha near 1 and appreciable diffusion coefficient.
      2. Subdiffusion    : anomalous exponent well below 1 (strong trapping).
      3. Hopping         : non-Gaussian displacement and frequent cage escapes.
      4. Mixed           : does not meet any of the above criteria.

    Parameters
    ----------
    features : dict
        Output of extract_features.

    Returns
    -------
    str
        One of 'free', 'subdiff', 'hopping', 'mixed'.
    """
    alpha    = features['msd_alpha']
    D        = features['msd_D']
    vh_alpha2 = features['vh_alpha2']
    jump_rate = features['jump_rate']

    if alpha > 0.85 and D > 0.04:
        return 'free'
    elif alpha < 0.72:
        return 'subdiff'
    elif vh_alpha2 > 0.25 and jump_rate > 0.01:
        return 'hopping'
    else:
        return 'mixed'


def build_feature_matrix(trajs: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix with one row per trajectory.

    Loops over all trajectories, extracts features, and assigns a diffusion
    regime label. Physical parameters (polymer_conc, np_charge, run_id) are
    included as columns for downstream analysis.

    Parameters
    ----------
    trajs : dict
        Full trajectories dict, keys (conc, charge, run_id).
    meta : pd.DataFrame
        Metadata DataFrame (from metadata.csv).

    Returns
    -------
    pd.DataFrame
        Feature matrix. Each row is one trajectory.
        Columns: all features + polymer_conc, np_charge, run_id, diffusion_regime.
    """
    rows = []
    for (conc, charge, run_id), traj_df in tqdm(trajs.items(),
                                                  desc='Extracting features',
                                                  unit='traj'):
        feat = extract_features(traj_df)
        feat['polymer_conc']     = conc
        feat['np_charge']        = charge
        feat['run_id']           = run_id
        feat['diffusion_regime'] = classify_regime(feat)
        rows.append(feat)

    df = pd.DataFrame(rows)

    # Print class distribution
    counts = df['diffusion_regime'].value_counts()
    total  = len(df)
    print('\nDiffusion regime distribution:')
    print(f'{"Regime":<12} {"Count":>6} {"Percentage":>12}')
    print('-' * 32)
    for regime, count in counts.items():
        print(f'{regime:<12} {count:>6} {100*count/total:>11.1f}%')
    print()

    return df
