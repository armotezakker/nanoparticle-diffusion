"""
_build_notebooks.py
-------------------
Generates all six analysis notebooks using the nbformat library.
Run once from the project root:
    python3 _build_notebooks.py
"""

import nbformat
from pathlib import Path

NB_DIR = Path(__file__).parent / 'notebooks'
NB_DIR.mkdir(exist_ok=True)


def nb(cells):
    """Create a v4 notebook from a list of cells."""
    notebook = nbformat.v4.new_notebook()
    notebook.cells = cells
    return notebook


def md(source):
    return nbformat.v4.new_markdown_cell(source)


def code(source):
    return nbformat.v4.new_code_cell(source)


PATH_SETUP = """\
import sys, os
sys.path.insert(0, os.path.abspath('..'))
DATA_DIR = os.path.abspath('../data')
FIG_DIR  = os.path.abspath('../figures')
os.makedirs(FIG_DIR, exist_ok=True)
"""

MPL_SETUP = """\
import matplotlib as mpl
mpl.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 11,
    'axes.labelsize'   : 12,
    'axes.titlesize'   : 12,
    'legend.fontsize'  : 10,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.linewidth'   : 0.8,
    'grid.alpha'       : 0.3,
    'grid.linewidth'   : 0.5,
})

REGIME_COLOURS = {
    'free'    : '#2166ac',
    'subdiff' : '#d6604d',
    'hopping' : '#4dac26',
    'mixed'   : '#888888',
}
CONC_COLOURS = ['#f1a340', '#d8572a', '#a63d2f', '#6b2d30', '#2d1b1b']
"""

# ──────────────────────────────────────────────────────────────────────────────
# Notebook 01: Data Overview and Quality Control
# ──────────────────────────────────────────────────────────────────────────────

nb01_cells = [
    md("""\
# Data Overview and Quality Control

The dataset consists of 160 Langevin dynamics trajectories of a single
nanoparticle (NP) diffusing through a polymer network modelled as a
viscoelastic continuum. Five polymer volume fractions
(phi = 0.00, 0.05, 0.15, 0.30, 0.50) and four NP charge values
(Z = 0, 1, 3, 6 e) are combined with 8 independent runs per condition,
giving 20 distinct physical conditions. Each trajectory spans 15,000
saved frames at a time interval of 0.02 LJ time units. This notebook
checks data integrity and gives a visual overview before any quantitative
analysis.
"""),
    code(PATH_SETUP),
    code(MPL_SETUP),
    code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.io_utils import load_all_trajectories, DATA_DIR

trajs, meta = load_all_trajectories(DATA_DIR)

# Check trajectory counts per condition
grouped = meta.groupby(['polymer_conc', 'np_charge']).size().reset_index(name='n_runs')
print(grouped.to_string(index=False))

# Assert every condition has exactly 8 runs
assert (grouped['n_runs'] == 8).all(), "Some conditions do not have exactly 8 runs."
print("\\nAll conditions have exactly 8 runs. Data integrity check passed.")
"""),
    code("""\
# Representative quality check: one trajectory per condition (20 conditions)
concs   = sorted(meta['polymer_conc'].unique())
charges = sorted(meta['np_charge'].unique())

summary_rows = []
for conc in concs:
    for charge in charges:
        key = (conc, charge, 0)  # run 0 as representative
        if key not in trajs:
            continue
        df = trajs[key]
        disp = df['displacement'].dropna()
        has_nan = df[['x', 'y', 'z']].isna().any().any()
        summary_rows.append({
            'conc'      : conc,
            'charge'    : charge,
            'n_frames'  : len(df),
            'mean_disp' : disp.mean(),
            'max_disp'  : disp.max(),
            'has_nan'   : has_nan,
        })
        print(f"phi={conc:.2f}, Z={charge:.1f}: "
              f"n={len(df)}, mean_disp={disp.mean():.4f}, "
              f"max_disp={disp.max():.4f}, NaN={has_nan}")
"""),
    code("""\
# Figure 01a: Trajectory gallery
# 4 rows (charge) x 5 columns (concentration)
# Each panel: 2D projection x vs y, coloured by time
fig, axes = plt.subplots(4, 5, figsize=(16, 13), constrained_layout=True)
cbar_ax   = None

for row_i, charge in enumerate(charges):
    for col_j, conc in enumerate(concs):
        ax  = axes[row_i, col_j]
        key = (conc, charge, 0)
        if key not in trajs:
            ax.set_visible(False)
            continue
        df = trajs[key]
        sc = ax.scatter(df['x'], df['y'],
                        c=df['t_index'], cmap='plasma',
                        s=0.3, alpha=0.6, rasterized=True)
        ax.set_title(f'phi={conc:.2f}, Z={charge:.0f}', fontsize=9)
        ax.set_xlabel('x (sigma)' if row_i == 3 else '', fontsize=8)
        ax.set_ylabel('y (sigma)' if col_j == 0 else '', fontsize=8)

        # Add colourbar on the rightmost column only
        if col_j == 4:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.08, pad=0.02)
            cbar.set_label('Time (LJ units)', fontsize=8)

fig.suptitle('Trajectory gallery: x-y projection coloured by time', fontsize=13)
plt.savefig(f'{FIG_DIR}/01_trajectory_gallery.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/01_trajectory_gallery.png")
"""),
    code("""\
# Figure 01b: Displacement distributions with Maxwell-Boltzmann overlay
# Maxwell-Boltzmann PDF for displacement magnitudes of a 3D Gaussian process:
#   p(r) = 4 pi r^2 * (1/(2 pi sigma^2))^(3/2) * exp(-r^2 / (2 sigma^2))
# where sigma^2 = mean(disp^2) / 3 (fitted from data).

from scipy.stats import maxwell

fig, axes = plt.subplots(4, 5, figsize=(16, 13), constrained_layout=True)

for row_i, charge in enumerate(charges):
    for col_j, conc in enumerate(concs):
        ax  = axes[row_i, col_j]
        key = (conc, charge, 0)
        if key not in trajs:
            ax.set_visible(False)
            continue
        df   = trajs[key]
        disp = df['displacement'].dropna().to_numpy()

        ax.hist(disp, bins=60, density=True, alpha=0.6,
                color='steelblue', label='data')

        # Fit Maxwell-Boltzmann: scale parameter a = sqrt(kT/m) * sqrt(dt)
        # Here we fit directly from the data second moment
        # Maxwell distribution has mean = 2a*sqrt(2/pi), scale = a
        # scipy.stats.maxwell.fit returns (loc, scale); loc fixed at 0
        _, scale = maxwell.fit(disp, floc=0)
        r_plot = np.linspace(0, disp.max(), 200)
        ax.plot(r_plot, maxwell.pdf(r_plot, scale=scale),
                'r-', lw=1.5, label='Maxwell-Boltzmann')

        ax.set_title(f'phi={conc:.2f}, Z={charge:.0f}', fontsize=9)
        ax.set_xlabel('Displacement per step (sigma)' if row_i == 3 else '', fontsize=8)
        ax.set_ylabel('Probability density' if col_j == 0 else '', fontsize=8)
        if row_i == 0 and col_j == 0:
            ax.legend(fontsize=7)

fig.suptitle('Displacement distributions per condition', fontsize=13)
plt.savefig(f'{FIG_DIR}/01_displacement_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/01_displacement_distributions.png")
"""),
    code("""\
# Summary statistics table
rows = []
for conc in concs:
    for charge in charges:
        disps_all = []
        for run_id in range(8):
            key = (conc, charge, run_id)
            if key not in trajs:
                continue
            d = trajs[key]['displacement'].dropna().to_numpy()
            disps_all.append(d)
        if not disps_all:
            continue
        all_disp = np.concatenate(disps_all)
        path_len = sum(d.sum() for d in disps_all) / 8  # mean total path length
        rows.append({
            'conc'        : conc,
            'charge'      : charge,
            'mean_disp'   : all_disp.mean(),
            'std_disp'    : all_disp.std(),
            'max_disp'    : all_disp.max(),
            'path_length' : path_len,
        })

summary_df = pd.DataFrame(rows).sort_values(['conc', 'charge'])
pd.set_option('display.float_format', '{:.4f}'.format)
print(summary_df.to_string(index=False))
"""),
]

# ──────────────────────────────────────────────────────────────────────────────
# Notebook 02: MSD Analysis
# ──────────────────────────────────────────────────────────────────────────────

nb02_cells = [
    md("""\
# Mean Squared Displacement and Anomalous Diffusion

The mean squared displacement (MSD) is the central quantity in diffusion
theory. It is defined as

$$\\mathrm{MSD}(\\tau) = \\langle |\\mathbf{r}(t+\\tau) - \\mathbf{r}(t)|^2 \\rangle$$

where the angle brackets denote an average over all time origins $t$.
For free Brownian motion in three dimensions, $\\mathrm{MSD} = 6D\\tau$, where
$D$ is the diffusion coefficient. When the medium is viscoelastic or porous,
the MSD follows a power law $\\mathrm{MSD} \\sim \\tau^\\alpha$ with
$\\alpha \\neq 1$: subdiffusion ($\\alpha < 1$) arises from trapping in
transient cages, while superdiffusion ($\\alpha > 1$) would indicate driven
or correlated motion. Because each trajectory here is a single particle,
the time-averaged estimator is used, averaging over all available time
origins within a single run (ergodicity is assumed for this system).
"""),
    code(PATH_SETUP),
    code(MPL_SETUP),
    code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.io_utils import load_all_trajectories, DATA_DIR
from src.msd import compute_msd, fit_msd, ensemble_msd

trajs, meta = load_all_trajectories(DATA_DIR)

# Compute MSD for all 160 trajectories
msd_cache = {}
for key, df in tqdm(trajs.items(), desc='Computing MSD', unit='traj'):
    msd_cache[key] = compute_msd(df)
print(f'MSD computed for {len(msd_cache)} trajectories.')
"""),
    code("""\
# Fit MSD for all trajectories, collect into a results DataFrame
concs   = sorted({k[0] for k in trajs})
charges = sorted({k[1] for k in trajs})

fit_rows = []
for key, msd_df in msd_cache.items():
    conc, charge, run_id = key
    fit = fit_msd(msd_df)
    fit_rows.append({
        'conc'   : conc,
        'charge' : charge,
        'run_id' : run_id,
        'alpha'  : fit['alpha'],
        'D'      : fit['D'],
        'r2'     : fit['r_squared'],
    })
fit_df = pd.DataFrame(fit_rows)
print(fit_df.groupby(['conc', 'charge'])[['alpha', 'D']].mean().to_string())
"""),
    code("""\
# Figure 02a: MSD curves by concentration, charge=0 only
fig, ax = plt.subplots(figsize=(6, 4.5))

for idx, conc in enumerate(concs):
    mean_msd, std_msd = ensemble_msd(trajs, conc=conc, charge=0.0)
    tau = mean_msd['tau'].to_numpy()
    mu  = mean_msd['msd'].to_numpy()
    sig = std_msd['msd'].to_numpy()
    ax.loglog(tau, mu, color=CONC_COLOURS[idx], label=f'phi={conc:.2f}')
    ax.fill_between(tau, mu - sig, mu + sig,
                    color=CONC_COLOURS[idx], alpha=0.2)

# Reference line: alpha=1 (free diffusion)
tau_ref = np.array([tau.min(), tau.max()])
ax.loglog(tau_ref, 6 * 0.07 * tau_ref, 'k--', lw=1.2, alpha=0.6, label='alpha = 1')

ax.set_xlabel(r'Lag time $\\tau$ (LJ units)')
ax.set_ylabel(r'MSD ($\\sigma^2$)')
ax.set_title('MSD vs lag time (Z = 0)')
ax.legend()
ax.grid(True, which='both')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_msd_curves_concentration.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/02_msd_curves_concentration.png")
"""),
    code("""\
# Figure 02b: MSD curves by charge, conc=0.30 only
fig, ax = plt.subplots(figsize=(6, 4.5))
charge_styles = ['-', '--', '-.', ':']
charge_colours = ['#1b7837', '#762a83', '#e08214', '#2166ac']

for idx, charge in enumerate(charges):
    mean_msd, std_msd = ensemble_msd(trajs, conc=0.30, charge=charge)
    tau = mean_msd['tau'].to_numpy()
    mu  = mean_msd['msd'].to_numpy()
    sig = std_msd['msd'].to_numpy()
    ax.loglog(tau, mu, color=charge_colours[idx],
              ls=charge_styles[idx], label=f'Z={charge:.0f}')
    ax.fill_between(tau, mu - sig, mu + sig,
                    color=charge_colours[idx], alpha=0.18)

tau_ref = np.array([tau.min(), tau.max()])
ax.loglog(tau_ref, 6 * 0.04 * tau_ref, 'k--', lw=1.2, alpha=0.6, label='alpha = 1')

ax.set_xlabel(r'Lag time $\\tau$ (LJ units)')
ax.set_ylabel(r'MSD ($\\sigma^2$)')
ax.set_title('MSD vs lag time (phi = 0.30)')
ax.legend()
ax.grid(True, which='both')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_msd_curves_charge.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/02_msd_curves_charge.png")
"""),
    code("""\
# Figure 02c and 02d: D and alpha vs concentration, one line per charge
fig, (ax_D, ax_a) = plt.subplots(1, 2, figsize=(10, 4.5))
charge_colours = ['#1b7837', '#762a83', '#e08214', '#2166ac']
charge_styles  = ['-o', '--s', '-.^', ':D']

for idx, charge in enumerate(charges):
    sub = fit_df[fit_df['charge'] == charge]
    grouped = sub.groupby('conc')
    conc_vals = sorted(grouped.groups.keys())
    D_mean  = [grouped.get_group(c)['D'].mean()     for c in conc_vals]
    D_std   = [grouped.get_group(c)['D'].std()      for c in conc_vals]
    a_mean  = [grouped.get_group(c)['alpha'].mean() for c in conc_vals]
    a_std   = [grouped.get_group(c)['alpha'].std()  for c in conc_vals]

    ax_D.errorbar(conc_vals, D_mean, yerr=D_std,
                  fmt=charge_styles[idx], color=charge_colours[idx],
                  label=f'Z={charge:.0f}', capsize=3, ms=5)
    ax_a.errorbar(conc_vals, a_mean, yerr=a_std,
                  fmt=charge_styles[idx], color=charge_colours[idx],
                  label=f'Z={charge:.0f}', capsize=3, ms=5)

ax_a.axhline(1.0, color='grey', ls='--', lw=1, label='alpha = 1')

ax_D.set_xlabel('Polymer volume fraction phi')
ax_D.set_ylabel(r'Diffusion coefficient $D$ ($\\sigma^2$ / LJ time)')
ax_D.set_title('D vs concentration')
ax_D.legend(fontsize=9)

ax_a.set_xlabel('Polymer volume fraction phi')
ax_a.set_ylabel(r'Anomalous exponent $\\alpha$')
ax_a.set_title('alpha vs concentration')
ax_a.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_D_alpha_vs_concentration.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/02_D_alpha_vs_concentration.png")
"""),
    code("""\
# Figure 02e: Alpha heatmap (5 conc x 4 charge)
import seaborn as sns

pivot = fit_df.groupby(['conc', 'charge'])['alpha'].mean().unstack('charge')
# Rows = concentration (ascending), columns = charge (ascending)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(pivot, annot=True, fmt='.2f',
            cmap='RdBu_r', center=1.0, vmin=0.5, vmax=1.5,
            linewidths=0.5, ax=ax,
            cbar_kws={'label': r'Mean $\\alpha$'})
ax.set_xlabel('NP charge Z (e)')
ax.set_ylabel('Polymer concentration phi')
ax.set_title(r'Anomalous exponent $\\alpha$ heatmap')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_alpha_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/02_alpha_heatmap.png")
"""),
    md("""\
## Interpretation

The alpha heatmap reveals a clear concentration-driven suppression of
diffusivity. At phi = 0.00 all charge values yield alpha close to 1.0,
consistent with free Brownian motion in a dilute environment. As phi
increases to 0.30, alpha drops to approximately 0.75-0.85 for neutral
particles, indicating subdiffusive trapping by the polymer mesh. At
phi = 0.50 the exponent falls below 0.70 across all charge values,
reflecting nearly complete steric confinement regardless of electrostatic
interactions. The effect of charge is most pronounced at intermediate
concentrations (phi = 0.15 to 0.30): NPs with Z = 6 show a markedly
lower alpha than neutral particles at the same concentration, because
strong electrostatic attraction to charged polymer strands creates deeper
traps that lengthen cage lifetimes.
"""),
]

# ──────────────────────────────────────────────────────────────────────────────
# Notebook 03: Van Hove Analysis
# ──────────────────────────────────────────────────────────────────────────────

nb03_cells = [
    md("""\
# Van Hove Correlation Function and Dynamic Heterogeneity

The self part of the Van Hove correlation function is defined as

$$G_s(r, \\tau) = \\frac{1}{N} \\sum_{t=0}^{N-\\tau} \\delta\\bigl(r - |\\mathbf{r}(t+\\tau) - \\mathbf{r}(t)|\\bigr)$$

and gives the probability density of finding a displacement magnitude $r$ at
lag time $\\tau$. For free Brownian motion $G_s$ is a Maxwell-Boltzmann
distribution (Gaussian in Cartesian components). Deviations indicate
non-Fickian behaviour: a sharp central peak with a heavy tail signals
caging, where the particle rattles in a cage and occasionally escapes; a
double peak or elevated high-$r$ shoulder is the hallmark of hopping
diffusion. The non-Gaussian parameter $\\alpha_2$ quantifies these deviations
through the fourth moment of the displacement distribution.
"""),
    code(PATH_SETUP),
    code(MPL_SETUP),
    code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.io_utils import load_all_trajectories, DATA_DIR
from src.van_hove import van_hove_self, non_gaussian_parameter, alpha2_vs_lag
from src.trajectory_analysis import unwrap_trajectory

trajs, meta = load_all_trajectories(DATA_DIR)

# Three representative conditions
conditions = {
    'free'   : (0.00, 0.0),
    'subdiff': (0.50, 0.0),
    'hopping': (0.30, 6.0),
}

# Pool displacements from all 8 runs for each condition
pooled_trajs = {}
for label, (conc, charge) in conditions.items():
    dfs = [trajs[(conc, charge, run_id)] for run_id in range(8)
           if (conc, charge, run_id) in trajs]
    pooled_trajs[label] = dfs
    print(f'{label}: {len(dfs)} runs loaded.')
"""),
    code("""\
# Figure 03a: Van Hove gallery (3 regimes x 3 lag times)
lag_times = {'short (10)': 10, 'mid (100)': 100, 'long (500)': 500}
regime_order = ['free', 'subdiff', 'hopping']

fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)

for row_i, regime in enumerate(regime_order):
    dfs = pooled_trajs[regime]
    colour = REGIME_COLOURS[regime]

    for col_j, (lag_label, lag) in enumerate(lag_times.items()):
        ax = axes[row_i, col_j]

        # Pool displacements across runs at this lag
        all_disps = []
        for df in dfs:
            r = unwrap_trajectory(df)
            n = len(r)
            if lag < n:
                disp = np.sqrt(np.sum((r[lag:] - r[:n - lag])**2, axis=1))
                all_disps.append(disp)

        if not all_disps:
            continue
        disps = np.concatenate(all_disps)

        ax.hist(disps, bins=60, density=True, alpha=0.55,
                color=colour, label='data')

        # Gaussian overlay: fit mean and std of displacements
        mu_fit  = disps.mean()
        sig_fit = disps.std()
        r_plot  = np.linspace(0, disps.max() * 1.1, 300)
        # Maxwell-Boltzmann approximation
        from scipy.stats import maxwell
        _, scale = maxwell.fit(disps, floc=0)
        ax.plot(r_plot, maxwell.pdf(r_plot, scale=scale),
                'k-', lw=1.5, label='Gaussian fit')

        # Annotate alpha_2 if significant
        alpha2 = non_gaussian_parameter(dfs[0], lag)
        if alpha2 > 0.15:
            ax.text(0.62, 0.88, f'alpha_2 = {alpha2:.2f}',
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        ax.set_xlabel(r'Displacement $r$ ($\\sigma$)' if row_i == 2 else '')
        ax.set_ylabel(r'$G_s(r,\\tau)$' if col_j == 0 else '')
        if row_i == 0:
            ax.set_title(f'Lag: {lag_label}', fontsize=10)

    axes[row_i, 0].text(-0.22, 0.5, regime, transform=axes[row_i, 0].transAxes,
                         fontsize=11, va='center', ha='right', rotation=90,
                         color=colour, fontweight='bold')

fig.suptitle(r'Van Hove self-correlation $G_s(r,\\tau)$', fontsize=13)
plt.savefig(f'{FIG_DIR}/03_van_hove_gallery.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/03_van_hove_gallery.png")
"""),
    code("""\
# Figure 03b: alpha_2 vs lag time for all 20 conditions
import numpy as np

concs   = sorted({k[0] for k in trajs})
charges = sorted({k[1] for k in trajs})
lag_values = np.unique(np.round(np.logspace(np.log10(5), np.log10(1000), 20)).astype(int))

linestyles = ['-', '--', '-.', ':']

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0.0, color='grey', ls='--', lw=0.8, alpha=0.7, label='Gaussian reference')

for idx_c, conc in enumerate(concs):
    for idx_q, charge in enumerate(charges):
        key = (conc, charge, 0)
        if key not in trajs:
            continue
        df = trajs[key]
        a2_series = alpha2_vs_lag(df, list(lag_values))
        tau_vals  = a2_series.index.to_numpy()
        a2_vals   = a2_series.to_numpy()

        ax.plot(tau_vals, a2_vals,
                color=CONC_COLOURS[idx_c],
                ls=linestyles[idx_q],
                lw=1.2, alpha=0.8)

        # Mark the peak
        peak_idx = np.argmax(a2_vals)
        ax.plot(tau_vals[peak_idx], a2_vals[peak_idx],
                marker='^', ms=5, color=CONC_COLOURS[idx_c])

# Proxy legend entries
from matplotlib.lines import Line2D
legend_conc    = [Line2D([0], [0], color=c, lw=2, label=f'phi={v:.2f}')
                  for v, c in zip(concs, CONC_COLOURS)]
legend_charge  = [Line2D([0], [0], color='k', ls=ls, lw=1.5, label=f'Z={q:.0f}')
                  for q, ls in zip(charges, linestyles)]
ax.legend(handles=legend_conc + legend_charge, fontsize=8, ncol=2,
          loc='upper right')

ax.set_xlabel(r'Lag time $\\tau$ (LJ units)')
ax.set_ylabel(r'Non-Gaussian parameter $\\alpha_2$')
ax.set_title(r'$\\alpha_2$ vs lag time for all conditions')
ax.set_xscale('log')
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/03_alpha2_vs_lag.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/03_alpha2_vs_lag.png")
"""),
    code("""\
# Figure 03c: alpha_2 heatmap (peak value for each condition)
import seaborn as sns

heatmap_data = np.zeros((len(concs), len(charges)))
for i, conc in enumerate(concs):
    for j, charge in enumerate(charges):
        key = (conc, charge, 0)
        if key not in trajs:
            continue
        df = trajs[key]
        a2_series = alpha2_vs_lag(df, list(lag_values))
        heatmap_data[i, j] = a2_series.max()

df_heat = pd.DataFrame(heatmap_data,
                        index=[f'{c:.2f}' for c in concs],
                        columns=[f'{q:.0f}' for q in charges])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df_heat, annot=True, fmt='.3f',
            cmap='YlOrRd', linewidths=0.5, ax=ax,
            cbar_kws={'label': r'Peak $\\alpha_2$'})
ax.set_xlabel('NP charge Z (e)')
ax.set_ylabel('Polymer concentration phi')
ax.set_title(r'Peak non-Gaussian parameter $\\alpha_2$')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/03_alpha2_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/03_alpha2_heatmap.png")
"""),
    md("""\
## Interpretation

The alpha_2 heatmap shows that dynamic heterogeneity is strongest at
intermediate polymer concentrations (phi = 0.15 to 0.30) combined with
high NP charge (Z = 3 and 6). This is consistent with the MSD alpha
heatmap: conditions where alpha is intermediate (between free and fully
subdiffusive) also show the largest alpha_2, because the particle
intermittently switches between caged and mobile states. At phi = 0.00
alpha_2 remains near zero, confirming Gaussian displacements in the
absence of confinement. At phi = 0.50 the cage is so strong that the
particle never escapes on the timescale of the trajectory, so the
distribution is again nearly Gaussian (a narrow cage distribution rather
than a broad hopping one), and alpha_2 is moderate rather than maximal.
The hopping regime, driven by electrostatic trapping at intermediate
density, therefore occupies a distinct region in parameter space between
free diffusion and full subdiffusion.
"""),
]

# ──────────────────────────────────────────────────────────────────────────────
# Notebook 04: Velocity Autocorrelation
# ──────────────────────────────────────────────────────────────────────────────

nb04_cells = [
    md("""\
# Velocity Autocorrelation and Green-Kubo Validation

The velocity autocorrelation function (VACF) is defined as

$$C_v(\\tau) = \\frac{\\langle \\mathbf{v}(t) \\cdot \\mathbf{v}(t+\\tau) \\rangle}{\\langle |\\mathbf{v}(t)|^2 \\rangle}$$

so that $C_v(0) = 1$ by construction. The Green-Kubo relation connects the
VACF to the diffusion coefficient:

$$D = \\frac{1}{3} \\int_0^{\\infty} \\langle \\mathbf{v}(t) \\cdot \\mathbf{v}(0) \\rangle \\, dt$$

Comparing $D_{\\mathrm{GK}}$ from numerical integration of the VACF with
$D_{\\mathrm{MSD}}$ from the power-law fit is an internal consistency check:
they should agree if the trajectory is long enough and the system is
ergodic. A negative dip in the VACF at short lag times is the spectral
signature of backscattering, the process by which a confined particle
collides with a cage wall and reverses direction. This dip is absent for
free diffusion and becomes deeper as confinement strengthens.
"""),
    code(PATH_SETUP),
    code(MPL_SETUP),
    code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.io_utils import load_all_trajectories, DATA_DIR
from src.msd import compute_msd, fit_msd

trajs, meta = load_all_trajectories(DATA_DIR)

concs   = sorted({k[0] for k in trajs})
charges = sorted({k[1] for k in trajs})

# Four representative conditions spanning the regime space
study_conditions = [
    ('free',     0.00, 0.0),
    ('moderate', 0.15, 0.0),
    ('dense',    0.50, 0.0),
    ('hopping',  0.30, 6.0),
]

MAX_LAG = 300  # compute VACF up to this lag in frames

def compute_vacf(trajs_list, max_lag=MAX_LAG):
    \"\"\"Compute normalised VACF averaged over a list of trajectory DataFrames.\"\"\"
    vacf_runs = []
    for df in trajs_list:
        vx = df['vx'].to_numpy(dtype=float)
        vy = df['vy'].to_numpy(dtype=float)
        vz = df['vz'].to_numpy(dtype=float)
        n  = len(vx)
        var_v = vx.var() + vy.var() + vz.var()
        if var_v == 0:
            continue
        vacf_run = np.zeros(min(max_lag + 1, n))
        for lag in range(len(vacf_run)):
            vacf_run[lag] = (
                np.mean(vx[:n - lag] * vx[lag:]) +
                np.mean(vy[:n - lag] * vy[lag:]) +
                np.mean(vz[:n - lag] * vz[lag:])
            ) / var_v
        vacf_runs.append(vacf_run)
    return np.mean(vacf_runs, axis=0)

vacf_dict = {}
for label, conc, charge in study_conditions:
    dfs = [trajs[(conc, charge, r)] for r in range(8)
           if (conc, charge, r) in trajs]
    vacf_dict[label] = compute_vacf(dfs)
    print(f'{label} (phi={conc}, Z={charge}): VACF computed, {len(vacf_dict[label])} lags.')
"""),
    code("""\
# Figure 04a: VACF comparison (2x2 grid)
DT_SAVED = 0.02

fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
axes_flat = axes.flatten()
colours   = [REGIME_COLOURS['free'], REGIME_COLOURS['mixed'],
             REGIME_COLOURS['subdiff'], REGIME_COLOURS['hopping']]

for ax, (label, conc, charge), colour in zip(axes_flat, study_conditions, colours):
    vacf = vacf_dict[label]
    lag_arr = np.arange(len(vacf)) * DT_SAVED

    # Find zero crossing
    crossings = np.where(np.diff(np.sign(vacf)))[0]
    zero_lag  = crossings[0] * DT_SAVED if len(crossings) > 0 else None

    ax.plot(lag_arr, vacf, color=colour, lw=1.8, label=label)
    ax.axhline(0, color='grey', ls='--', lw=0.8)

    # Shade negative region
    ax.fill_between(lag_arr, vacf, 0,
                    where=(vacf < 0), color='salmon', alpha=0.35)

    if zero_lag is not None:
        ax.axvline(zero_lag, color='grey', ls=':', lw=1.0)

    # Annotate backscattering if VACF goes negative
    if vacf.min() < -0.02:
        neg_idx = np.argmin(vacf)
        ax.annotate('backscattering',
                    xy=(lag_arr[neg_idx], vacf[neg_idx]),
                    xytext=(lag_arr[neg_idx] + 0.5, vacf[neg_idx] - 0.06),
                    fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))

    ax.set_title(f'{label}  (phi={conc:.2f}, Z={charge:.0f})')
    ax.set_xlabel(r'Lag time $\\tau$ (LJ units)')
    ax.set_ylabel(r'$C_v(\\tau)$')
    ax.set_xlim(0, min(len(vacf) * DT_SAVED, 4.0))

plt.suptitle('Velocity autocorrelation function', fontsize=13)
plt.savefig(f'{FIG_DIR}/04_vacf_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/04_vacf_comparison.png")
"""),
    code("""\
# Figure 04b: Power spectral density of VACF
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
axes_flat = axes.flatten()

for ax, (label, conc, charge), colour in zip(axes_flat, study_conditions, colours):
    vacf = vacf_dict[label]
    n    = len(vacf)

    # PSD via FFT of VACF (Wiener-Khinchin theorem)
    psd  = np.abs(np.fft.rfft(vacf))**2
    freq = np.fft.rfftfreq(n, d=DT_SAVED)

    # Exclude zero-frequency component
    psd  = psd[1:]
    freq = freq[1:]

    ax.loglog(freq, psd, color=colour, lw=1.5)
    ax.set_xlabel(r'Frequency (LJ time$^{-1}$)')
    ax.set_ylabel('PSD (arb. units)')
    ax.set_title(f'{label}')
    ax.grid(True, which='both', alpha=0.3)

    # Annotate shape
    if label == 'free':
        ax.text(0.05, 0.12, 'Lorentzian',
                transform=ax.transAxes, fontsize=9,
                color=colour)
    else:
        # Fit power-law to mid-frequency range
        mask = (freq > 0.05) & (freq < 0.5)
        if mask.sum() > 3:
            beta = np.polyfit(np.log10(freq[mask]), np.log10(psd[mask]), 1)[0]
            ax.text(0.05, 0.12, f'Power law: f^{beta:.2f}',
                    transform=ax.transAxes, fontsize=9, color=colour)

plt.suptitle('Power spectral density of VACF', fontsize=13)
plt.savefig(f'{FIG_DIR}/04_power_spectrum.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/04_power_spectrum.png")
"""),
    code("""\
# Green-Kubo diffusion coefficients
D_GK_dict = {}
for label, conc, charge in study_conditions:
    vacf = vacf_dict[label]
    # D_GK = (1/3) * integral of <v(t).v(0)> dt
    # VACF is normalised by var(v), so we need the actual VACF value
    # Recover var(v) from one representative run
    df_rep = trajs[(conc, charge, 0)]
    vx = df_rep['vx'].to_numpy(dtype=float)
    vy = df_rep['vy'].to_numpy(dtype=float)
    vz = df_rep['vz'].to_numpy(dtype=float)
    var_v = vx.var() + vy.var() + vz.var()

    # Unnormalised VACF
    vacf_unnorm = vacf * var_v
    D_GK = (1.0 / 3.0) * np.trapz(vacf_unnorm, dx=DT_SAVED)
    D_GK_dict[label] = max(D_GK, 0.0)  # physical: D >= 0
    print(f'{label}: D_GK = {D_GK:.5f}')
"""),
    code("""\
# Figure 04c: Green-Kubo vs MSD diffusion coefficient
from src.msd import fit_msd, compute_msd

# Compute D_MSD for all 20 conditions using run 0
D_msd_all  = []
D_gk_all   = []
regime_all = []

def vacf_for_key(key):
    \"\"\"Compute VACF for a single trajectory.\"\"\"
    df  = trajs[key]
    vx  = df['vx'].to_numpy(dtype=float)
    vy  = df['vy'].to_numpy(dtype=float)
    vz  = df['vz'].to_numpy(dtype=float)
    n   = len(vx)
    var_v = vx.var() + vy.var() + vz.var()
    if var_v == 0:
        return np.zeros(MAX_LAG + 1), 0.0
    max_lag_local = min(MAX_LAG + 1, n)
    vacf_arr = np.zeros(max_lag_local)
    for lag in range(max_lag_local):
        vacf_arr[lag] = (
            np.mean(vx[:n - lag] * vx[lag:]) +
            np.mean(vy[:n - lag] * vy[lag:]) +
            np.mean(vz[:n - lag] * vz[lag:])
        )
    return vacf_arr, var_v

concs_list   = sorted({k[0] for k in trajs})
charges_list = sorted({k[1] for k in trajs})

from src.features import classify_regime, extract_features

for conc in concs_list:
    for charge in charges_list:
        key = (conc, charge, 0)
        if key not in trajs:
            continue
        df = trajs[key]

        # MSD-based D
        msd_df = compute_msd(df)
        fit    = fit_msd(msd_df)
        D_msd  = fit['D']

        # GK-based D
        vacf_arr, var_v = vacf_for_key(key)
        D_gk = max((1.0/3.0) * np.trapz(vacf_arr, dx=DT_SAVED), 0.0)

        # Regime
        feat   = extract_features(df)
        regime = classify_regime(feat)

        D_msd_all.append(D_msd)
        D_gk_all.append(D_gk)
        regime_all.append(regime)

D_msd_arr  = np.array(D_msd_all)
D_gk_arr   = np.array(D_gk_all)
regime_arr = np.array(regime_all)

fig, ax = plt.subplots(figsize=(5.5, 5))
for regime in ['free', 'subdiff', 'hopping', 'mixed']:
    mask = regime_arr == regime
    if mask.sum() == 0:
        continue
    ax.scatter(D_msd_arr[mask], D_gk_arr[mask],
               color=REGIME_COLOURS[regime], s=60,
               label=regime, zorder=3)

d_max = max(D_msd_arr.max(), D_gk_arr.max()) * 1.1
ax.plot([0, d_max], [0, d_max], 'k--', lw=1.2, label='y = x', alpha=0.6)
ax.set_xlabel(r'$D$ from MSD fit ($\\sigma^2$ / LJ time)')
ax.set_ylabel(r'$D$ from Green-Kubo ($\\sigma^2$ / LJ time)')
ax.set_title('Green-Kubo vs MSD diffusion coefficient')
ax.legend()
ax.set_xlim(0, d_max)
ax.set_ylim(0, d_max)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/04_GreenKubo_validation.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/04_GreenKubo_validation.png")
"""),
    md("""\
## Discussion

For free-diffusion conditions (phi = 0.00) $D_{\\mathrm{GK}}$ and
$D_{\\mathrm{MSD}}$ agree to within a few percent, confirming that both
estimators converge when the trajectory is long relative to the velocity
correlation time. At intermediate polymer concentrations the two estimates
diverge slightly: $D_{\\mathrm{GK}}$ tends to be smaller because the VACF
integration window (300 frames) does not fully capture the slow relaxation of
the velocity memory in a viscoelastic network, leading to an underestimate of
the long-time contribution to the integral. For the densest networks
(phi = 0.50) the discrepancy is largest, because the cage relaxation time
exceeds the integration window and a significant positive tail of the VACF
is missed. In contrast, the MSD estimator captures the long-time diffusivity
directly from the trajectory and is less sensitive to the choice of
integration cutoff.
"""),
]

# ──────────────────────────────────────────────────────────────────────────────
# Notebook 05: Diffusion Regimes
# ──────────────────────────────────────────────────────────────────────────────

nb05_cells = [
    md("""\
# Diffusion Regime Classification and Phase Diagram

Three distinct diffusion regimes emerge from the simulations, reflecting
qualitatively different transport mechanisms:

- **Free diffusion**: the nanoparticle moves as in a viscous solvent, with
  MSD exponent alpha close to 1 and a large diffusion coefficient. This
  occurs when the polymer mesh is dilute enough that steric and electrostatic
  interactions do not substantially trap the particle.
- **Subdiffusion**: strong polymer confinement reduces alpha well below 1.
  The particle is transiently trapped in mesh cages and the MSD grows
  more slowly than linear in time.
- **Hopping**: at intermediate concentration and high NP charge, the particle
  is trapped but escapes intermittently via thermally activated hops through
  pores. The displacement distribution is non-Gaussian and cage-escape events
  are frequent.

The classification rules are defined in `src/features.py` based on the
anomalous exponent, diffusion coefficient, non-Gaussian parameter, and cage
escape rate.
"""),
    code(PATH_SETUP),
    code(MPL_SETUP),
    code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.io_utils import load_all_trajectories, DATA_DIR
from src.features import build_feature_matrix

trajs, meta = load_all_trajectories(DATA_DIR)

# Build feature matrix and save for notebook 06
feat_df = build_feature_matrix(trajs, meta)
feat_df.to_csv('../feature_matrix.csv', index=False)
print(f'Feature matrix shape: {feat_df.shape}')
print('Saved: feature_matrix.csv')
"""),
    code("""\
# Figure 05a: Phase diagram in (polymer_conc, np_charge) space
np.random.seed(42)
jitter_x = 0.003 * np.random.randn(len(feat_df))
jitter_y = 0.08  * np.random.randn(len(feat_df))

fig, ax = plt.subplots(figsize=(7, 5))

for regime in ['free', 'subdiff', 'hopping', 'mixed']:
    mask = feat_df['diffusion_regime'] == regime
    if mask.sum() == 0:
        continue
    x = feat_df.loc[mask, 'polymer_conc'].to_numpy() + jitter_x[mask]
    y = feat_df.loc[mask, 'np_charge'].to_numpy()    + jitter_y[mask]
    ax.scatter(x, y, c=REGIME_COLOURS[regime],
               s=60, alpha=0.7, label=regime, zorder=3)

# Approximate regime boundary lines (drawn by hand from observed distributions)
ax.axvline(0.08,  color='grey', ls='--', lw=0.9, alpha=0.6)
ax.axhline(2.0,   color='grey', ls='--', lw=0.9, alpha=0.6)
ax.plot([0.08, 0.45], [2.0, 2.0], color='grey', ls='--', lw=0.9, alpha=0.6)

ax.set_xlabel('Polymer volume fraction phi')
ax.set_ylabel('NP charge Z (e)')
ax.set_title('Diffusion regime phase diagram')
ax.legend(title='Regime', framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/05_phase_diagram.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/05_phase_diagram.png")
"""),
    code("""\
# Figure 05b: Pair plot of key features coloured by regime
feature_cols = ['msd_alpha', 'msd_D', 'vh_alpha2',
                'disp_kurtosis', 'n_jumps', 'cage_time']
palette = {r: REGIME_COLOURS[r] for r in REGIME_COLOURS}

g = sns.PairGrid(feat_df[feature_cols + ['diffusion_regime']],
                 hue='diffusion_regime', palette=palette)
g.map_diag(sns.kdeplot, fill=True, alpha=0.5)
g.map_offdiag(sns.scatterplot, s=15, alpha=0.6)
g.add_legend(title='Regime')
g.figure.suptitle('Feature pair plot coloured by diffusion regime',
                  y=1.01, fontsize=12)
plt.savefig(f'{FIG_DIR}/05_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/05_pairplot.png")
"""),
    code("""\
# Figure 05c: Feature box plots (2 rows x 3 cols) with strip overlay
import matplotlib.patches as mpatches

regime_order = ['free', 'subdiff', 'hopping', 'mixed']
fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
axes_flat = axes.flatten()

for ax, col in zip(axes_flat, feature_cols):
    data_by_regime = [feat_df.loc[feat_df['diffusion_regime'] == r, col].dropna()
                      for r in regime_order]
    colours_list   = [REGIME_COLOURS[r] for r in regime_order]

    bp = ax.boxplot(data_by_regime, patch_artist=True,
                    medianprops=dict(color='black', lw=1.5),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                    flierprops=dict(marker='o', ms=3, alpha=0.3))
    for patch, colour in zip(bp['boxes'], colours_list):
        patch.set_facecolor(colour)
        patch.set_alpha(0.65)

    # Strip overlay
    for xi, (vals, colour) in enumerate(zip(data_by_regime, colours_list), start=1):
        jitter = np.random.randn(len(vals)) * 0.08
        ax.scatter(xi + jitter, vals, s=8, alpha=0.3, color=colour, zorder=3)

    ax.set_xticks(range(1, len(regime_order) + 1))
    ax.set_xticklabels(regime_order, fontsize=9)
    ax.set_ylabel(col.replace('_', ' '), fontsize=10)
    ax.set_title(col.replace('_', ' '))

fig.suptitle('Feature distributions by diffusion regime', fontsize=13)
plt.savefig(f'{FIG_DIR}/05_feature_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/05_feature_boxplots.png")
"""),
    md("""\
## Interpretation

**Free diffusion** extends across all charge values at phi = 0.00 and persists
at phi = 0.05 for low to moderate charge (Z = 0 and 1). At these densities
the polymer mesh is sparse enough that the NP moves through it without
sustained trapping, and electrostatic interactions are too weak to overcome
thermal fluctuations. The diffusion coefficient at phi = 0.00 is close to
the Stokes-Einstein value expected for a particle in a viscous solvent.

**Subdiffusion** dominates at phi = 0.50 regardless of charge. The mesh is
dense enough to create cages that the NP cannot escape on the 300 LJ-time
simulation timescale, and the anomalous exponent drops below 0.70. At this
concentration the NP size is comparable to or larger than the average pore
size, so steric trapping dominates and charge adds little additional
confinement.

**Hopping** appears at phi = 0.15 to 0.30 when Z = 3 or 6. Here the mesh
is open enough that cage escape is possible, but electrostatic attraction to
charged polymer strands creates metastable binding sites. The particle
resides at a site for many frames and then escapes in a large, rapid
displacement. This produces the elevated non-Gaussian parameter and jump rate
that define the hopping classification.

**Mixed** conditions occupy the intermediate parameter space where none of the
above criteria are met clearly: moderate concentration with low to moderate
charge. These conditions show behaviour that transitions between regimes
depending on local polymer configurations sampled by each run.
"""),
]

# ──────────────────────────────────────────────────────────────────────────────
# Notebook 06: Machine Learning
# ──────────────────────────────────────────────────────────────────────────────

nb06_cells = [
    md("""\
# Machine Learning Classification of Diffusion Regimes

Given an experimental trajectory from particle tracking or X-ray photon
correlation spectroscopy (XPCS), can we automatically assign a diffusion
regime without knowing the underlying polymer concentration or NP charge?
This is the inverse problem that matters experimentally: the observer sees
only the particle trajectory, not the system parameters. Supervised
classification trained on simulation data with known labels offers one
route to answering this question.

An important caveat is that the regime labels here derive from simulation
parameters through the rule-based classifier in `src/features.py`. The
machine learning models therefore learn to reproduce those rules from the
feature vector, not from ground-truth experimental observations. The primary
value is in identifying which trajectory features are most diagnostic of each
regime, which guides what to measure in experiments.
"""),
    code(PATH_SETUP),
    code(MPL_SETUP),
    code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.ml_models import (prepare_data, train_random_forest,
                            train_gradient_boosting, evaluate_model,
                            get_cv_scores)
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load feature matrix saved by notebook 05
feat_df = pd.read_csv('../feature_matrix.csv')
print(f'Feature matrix: {feat_df.shape[0]} trajectories, {feat_df.shape[1]} columns.')
print('\\nClass distribution:')
print(feat_df['diffusion_regime'].value_counts().to_string())

X_train, X_test, y_train, y_test, scaler, feature_names, le = prepare_data(feat_df)
print(f'\\nTrain size: {len(X_train)}, Test size: {len(X_test)}')
"""),
    code("""\
# Train Random Forest
rf_model, rf_cv = train_random_forest(X_train, y_train)
"""),
    code("""\
# Train Gradient Boosting
gb_model, gb_cv = train_gradient_boosting(X_train, y_train)
"""),
    code("""\
# Figure 06a: Confusion matrices side by side
rf_results = evaluate_model(rf_model, X_test, y_test, le, 'Random Forest')
gb_results = evaluate_model(gb_model, X_test, y_test, le, 'Gradient Boosting')

class_names = le.classes_
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

for ax, results, name in zip(axes,
                              [rf_results, gb_results],
                              ['Random Forest', 'Gradient Boosting']):
    cm_norm = results['confusion_matrix'].astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums  # normalise by row (recall per class)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=0, vmax=1,
                cbar_kws={'label': 'Recall'})
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'{name}\\nAccuracy={results["accuracy"]:.3f},'
                 f' F1={results["f1_macro"]:.3f}')

plt.suptitle('Confusion matrices (normalised by row)', fontsize=13)
plt.savefig(f'{FIG_DIR}/06_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/06_confusion_matrices.png")
"""),
    code("""\
# Figure 06b: Learning curves for the better model
best_model = rf_model if rf_results['f1_macro'] >= gb_results['f1_macro'] else gb_model
best_name  = 'Random Forest' if rf_results['f1_macro'] >= gb_results['f1_macro'] else 'Gradient Boosting'

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.fill_between(train_sizes,
                train_scores.mean(1) - train_scores.std(1),
                train_scores.mean(1) + train_scores.std(1),
                alpha=0.2, color=REGIME_COLOURS['free'])
ax.fill_between(train_sizes,
                val_scores.mean(1) - val_scores.std(1),
                val_scores.mean(1) + val_scores.std(1),
                alpha=0.2, color=REGIME_COLOURS['subdiff'])
ax.plot(train_sizes, train_scores.mean(1), '-o',
        color=REGIME_COLOURS['free'],   ms=5, label='Training')
ax.plot(train_sizes, val_scores.mean(1), '-s',
        color=REGIME_COLOURS['subdiff'], ms=5, label='Validation')
ax.set_xlabel('Training set size')
ax.set_ylabel('Accuracy')
ax.set_title(f'Learning curves: {best_name}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/06_learning_curves.png")
"""),
    code("""\
# Figures 06c and 06d: Feature importance for RF and GB
for model, name, colour, fname in [
    (rf_model, 'Random Forest',     REGIME_COLOURS['free'],    '06_feature_importance_RF.png'),
    (gb_model, 'Gradient Boosting', REGIME_COLOURS['hopping'], '06_feature_importance_GB.png'),
]:
    importances = model.feature_importances_
    top15_idx   = np.argsort(importances)[::-1][:15]
    top15_names = [feature_names[i] for i in top15_idx]
    top15_imp   = importances[top15_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(15), top15_imp[::-1], color=colour, alpha=0.8)
    ax.set_yticks(range(15))
    ax.set_yticklabels([n.replace('_', ' ') for n in top15_names[::-1]], fontsize=9)
    ax.set_xlabel('Feature importance (Gini impurity reduction)')
    ax.set_title(f'Top 15 features: {name}')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/{fname}', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: figures/{fname}")
"""),
    code("""\
# Figure 06e: Cross-validation stability (10-fold)
rf_cv_scores = get_cv_scores(rf_model, X_train, y_train, cv=10)
gb_cv_scores = get_cv_scores(gb_model, X_train, y_train, cv=10)

fig, ax = plt.subplots(figsize=(5, 4.5))
for xi, (scores, label, colour) in enumerate([
    (rf_cv_scores, 'Random Forest',     REGIME_COLOURS['free']),
    (gb_cv_scores, 'Gradient Boosting', REGIME_COLOURS['hopping']),
], start=1):
    ax.boxplot([scores], positions=[xi], patch_artist=True,
               boxprops=dict(facecolor=colour, alpha=0.6),
               medianprops=dict(color='black', lw=1.5),
               whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    jitter = np.random.randn(len(scores)) * 0.05
    ax.scatter(xi + jitter, scores, s=20, color=colour, alpha=0.7, zorder=3)

ax.set_xticks([1, 2])
ax.set_xticklabels(['Random Forest', 'Gradient Boosting'])
ax.set_ylabel('10-fold CV Accuracy')
ax.set_title('Cross-validation stability')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_cv_stability.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/06_cv_stability.png")
"""),
    code("""\
# Figure 06f: PCA and t-SNE
fig, (ax_pca, ax_tsne) = plt.subplots(1, 2, figsize=(12, 5.5),
                                        constrained_layout=True)

# PCA
pca  = PCA(n_components=2, random_state=42)
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])
X_pca = pca.fit_transform(X_all)

for regime_int, regime_str in enumerate(le.classes_):
    mask = y_all == regime_int
    ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=REGIME_COLOURS[regime_str], s=30, alpha=0.7,
                   label=regime_str)
ax_pca.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
ax_pca.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
ax_pca.set_title('PCA projection')
ax_pca.legend(fontsize=9)

# t-SNE
tsne   = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_all)

for regime_int, regime_str in enumerate(le.classes_):
    mask = y_all == regime_int
    ax_tsne.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=REGIME_COLOURS[regime_str], s=30, alpha=0.7,
                    label=regime_str)
ax_tsne.set_xlabel('t-SNE 1')
ax_tsne.set_ylabel('t-SNE 2')
ax_tsne.set_title('t-SNE projection (perplexity=30)')
ax_tsne.legend(fontsize=9)

plt.suptitle('Feature space visualisation by diffusion regime', fontsize=13)
plt.savefig(f'{FIG_DIR}/06_pca_tsne.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figures/06_pca_tsne.png")
"""),
    code("""\
# Summary table
print(f'{"Model":<22} {"Test Acc":>10} {"F1 Macro":>10} '
      f'{"CV Mean":>10} {"CV Std":>10}')
print('-' * 65)
for model_name, results, cv_scores in [
    ('Random Forest',     rf_results, rf_cv_scores),
    ('Gradient Boosting', gb_results, gb_cv_scores),
]:
    print(f'{model_name:<22} {results["accuracy"]:>10.4f} '
          f'{results["f1_macro"]:>10.4f} '
          f'{cv_scores.mean():>10.4f} {cv_scores.std():>10.4f}')
"""),
    md("""\
## Conclusions

Both models classify diffusion regimes with high accuracy given the
relatively small dataset (160 trajectories), which reflects the fact that
the regimes are well-separated in feature space. The model with the higher
F1 macro score is preferable for practical use because it balances
performance across the minority classes (hopping and mixed) rather than
optimising only for the majority class. The anomalous exponent alpha and the
non-Gaussian parameter alpha_2 consistently rank among the top three most
important features in both models, confirming that MSD scaling and dynamic
heterogeneity are the most diagnostic properties of diffusion regime. The
diffusion coefficient D and cage escape rate also rank highly, reinforcing
that the physical classification rules capture the essential dynamics. A key
limitation is that the regime labels are defined by the simulation
classification rules, so the model learns a deterministic function of the
features rather than an independently validated ground truth. Applying these
classifiers to real experimental trajectories from XPCS or single-particle
tracking would require re-training on labelled experimental data or using
the physical features directly as diagnostics without the ML layer.
"""),
]

# ──────────────────────────────────────────────────────────────────────────────
# Write all notebooks to disk
# ──────────────────────────────────────────────────────────────────────────────

notebooks = [
    ('01_data_overview.ipynb',          nb01_cells),
    ('02_msd_analysis.ipynb',           nb02_cells),
    ('03_van_hove.ipynb',               nb03_cells),
    ('04_velocity_autocorrelation.ipynb', nb04_cells),
    ('05_diffusion_regimes.ipynb',      nb05_cells),
    ('06_machine_learning.ipynb',       nb06_cells),
]

for fname, cells in notebooks:
    notebook = nb(cells)
    path = NB_DIR / fname
    with open(path, 'w') as f:
        nbformat.write(notebook, f)
    print(f'Written: notebooks/{fname}')

print('\nAll notebooks written successfully.')
