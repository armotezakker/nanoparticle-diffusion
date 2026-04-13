"""
trajectory_analysis.py
-----------------------
Core trajectory-level computations: PBC unwrapping, displacement statistics,
rolling radius of gyration, and cage-escape detection.
"""

import numpy as np
import pandas as pd

BOX_SIZE = 50.0


def unwrap_trajectory(traj_df: pd.DataFrame) -> np.ndarray:
    """Return cumulative displacements from the starting position, PBC-unwrapped.

    Without unwrapping, the mean squared displacement computed from wrapped
    coordinates is bounded by (box/2)^2 at long times regardless of how far
    the particle has actually travelled. Once a trajectory is longer than the
    cage relaxation time, the particle crosses box faces many times, and the
    wrapped position reverts near the starting coordinate, artificially
    suppressing the MSD. Unwrapping accumulates the true incremental
    displacements so the MSD grows correctly at all lag times.

    The minimum-image convention is applied to each frame-to-frame step:
    if |dx| > box/2 in any direction, the particle has crossed a periodic
    boundary and we correct by box * sign(dx).

    Parameters
    ----------
    traj_df : pd.DataFrame
        Single trajectory DataFrame with columns x, y, z.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Cumulative displacement from the first frame in each spatial dimension.
    """
    coords = traj_df[['x', 'y', 'z']].to_numpy(dtype=float)
    n = len(coords)
    unwrapped = np.zeros((n, 3), dtype=float)

    for i in range(1, n):
        delta = coords[i] - coords[i - 1]
        # Minimum-image convention: fold back displacements larger than half box
        delta[delta >  BOX_SIZE / 2] -= BOX_SIZE
        delta[delta < -BOX_SIZE / 2] += BOX_SIZE
        unwrapped[i] = unwrapped[i - 1] + delta

    return unwrapped


def displacement_magnitudes(traj_df: pd.DataFrame) -> np.ndarray:
    """Return frame-to-frame displacement magnitudes, PBC-corrected.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Single trajectory DataFrame with columns x, y, z.

    Returns
    -------
    np.ndarray, shape (N-1,)
        Displacement magnitude for each consecutive pair of frames.
    """
    coords = traj_df[['x', 'y', 'z']].to_numpy(dtype=float)
    delta = coords[1:] - coords[:-1]

    # Minimum-image correction for each component
    delta[delta >  BOX_SIZE / 2] -= BOX_SIZE
    delta[delta < -BOX_SIZE / 2] += BOX_SIZE

    return np.sqrt((delta**2).sum(axis=1))


def rolling_radius_of_gyration(traj_df: pd.DataFrame,
                                window: int = 500) -> pd.Series:
    """Compute the rolling radius of gyration over a sliding window.

    The radius of gyration within a window measures the spatial extent
    of the trajectory segment. In a caged environment Rg is small; after
    cage escape it rises sharply. The rolling Rg therefore tracks how
    localised the particle is as a function of time.

    Rg = sqrt( mean( |r_i - r_cm|^2 ) )

    where r_cm is the centre of mass of the positions within the window.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Single trajectory DataFrame with columns x, y, z.
    window : int
        Number of frames in the sliding window (default 500).

    Returns
    -------
    pd.Series
        Rolling Rg indexed by frame number. First (window-1) entries are NaN.
    """
    coords = traj_df[['x', 'y', 'z']].to_numpy(dtype=float)
    n = len(coords)
    rg_values = np.full(n, np.nan)

    for i in range(window - 1, n):
        segment = coords[i - window + 1: i + 1]
        r_cm = segment.mean(axis=0)
        rg_values[i] = np.sqrt(np.mean(np.sum((segment - r_cm)**2, axis=1)))

    return pd.Series(rg_values, index=traj_df.index if 't_index' not in traj_df.columns
                     else traj_df['t_index'].values)


def detect_cage_escapes(traj_df: pd.DataFrame,
                        threshold_factor: float = 2.5) -> np.ndarray:
    """Detect cage-escape events as anomalously large frame-to-frame displacements.

    In a polymer network, the nanoparticle is transiently trapped inside a
    cage formed by surrounding polymer strands. Most of the time the particle
    rattles within the cage with small displacements. Occasionally, thermally
    activated fluctuations allow the particle to squeeze through a pore and
    escape into a neighbouring cage. These cage-escape events appear as
    displacement magnitudes far exceeding the typical rattling amplitude.

    Detection criterion: a frame is flagged if its displacement magnitude
    exceeds threshold_factor * median(displacement). Using the median rather
    than the mean makes the threshold robust to outliers.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Single trajectory DataFrame with columns x, y, z.
    threshold_factor : float
        Multiplier applied to the median displacement to set the threshold.
        Default 2.5 captures rare, large hops without flagging thermal noise.

    Returns
    -------
    np.ndarray of int
        Frame indices (0-based, relative to traj_df row order) where
        cage-escape events are detected.
    """
    disp = displacement_magnitudes(traj_df)
    threshold = threshold_factor * np.median(disp)
    # disp has length N-1; the escape occurs at frame i+1 (the arrival frame)
    escape_indices = np.where(disp > threshold)[0] + 1
    return escape_indices
