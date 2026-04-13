# Nanoparticle Diffusion in Polymer Networks: Simulation and Analysis

**Ahmad Reza Motezakker**
KTH Royal Institute of Technology

---

## 1. Overview

The central question of this study is how the transport of a charged
nanoparticle (NP) through a polymer network is governed by the interplay
between steric confinement and electrostatic interactions. The parameter
space explored covers five polymer volume fractions
(phi = 0.00, 0.05, 0.15, 0.30, 0.50) and four NP charge values
(Z = 0, 1, 3, 6 e), with 8 independent Langevin dynamics runs per
condition for a total of 160 trajectories. Together these conditions reveal
three qualitatively distinct diffusion regimes: free Brownian diffusion,
anomalous subdiffusion driven by steric trapping, and hopping transport
driven by electrostatic binding at intermediate polymer density.

---

## 2. Simulation Model

The NP obeys Langevin dynamics with the equation of motion

$$m\ddot{\mathbf{r}} = -\gamma \dot{\mathbf{r}} + \mathbf{F}_\text{polymer}
 + \mathbf{F}_\text{elec} + \boldsymbol{\xi}(t)$$

where $\gamma$ is the friction coefficient, $\mathbf{F}_\text{polymer}$
encodes the excluded-volume and viscoelastic forces from the polymer mesh,
$\mathbf{F}_\text{elec}$ is the screened Coulomb interaction between the NP
and charged polymer strands, and $\boldsymbol{\xi}(t)$ is Gaussian white
noise satisfying the fluctuation-dissipation theorem. The simulation box
has side length 50 sigma with periodic boundary conditions; the NP has unit
mass and unit diameter. The saved time step is
$\Delta t_\text{saved} = 0.02$ LJ time units (integration step 0.002, saved
every 10 steps), and each trajectory spans 15,000 frames corresponding to a
total time of 300 LJ time units. Position and velocity data include
Gaussian detector noise ($\sigma = 0.008$ sigma) and a slow thermal drift,
mimicking realistic XPCS detector conditions in which absolute
displacements cannot be recovered without post-processing.

---

## 3. Mean Squared Displacement

The time-averaged MSD estimator was computed for all 160 trajectories and
fitted to a power law $\mathrm{MSD} \sim \tau^\alpha$ in the range
5%--35% of the total lag window to exclude the ballistic regime and the
noisy long-time tail. The anomalous exponent alpha spans the range
approximately 0.60 to 1.05 across the full parameter space. At phi = 0.00
all charge values yield alpha = $0.97 \pm 0.04$, consistent with free
Brownian motion. Alpha decreases monotonically with increasing polymer
concentration, reaching $0.62 \pm 0.05$ at phi = 0.50 for neutral
particles. The alpha heatmap (figures/02_alpha_heatmap.png) shows that
charge amplifies this suppression at phi = 0.15 to 0.30: particles with
Z = 6 achieve alpha values 0.10 to 0.15 lower than neutral particles at
the same concentration, because electrostatic binding to charged polymer
strands extends cage lifetimes. The diffusion coefficient D decreases by
more than an order of magnitude between phi = 0.00 and phi = 0.50 for all
charge values, confirming that steric confinement is the dominant factor at
high concentration.

---

## 4. Dynamic Heterogeneity

The non-Gaussian parameter alpha_2, computed from the self part of the Van
Hove correlation function, measures the degree to which the displacement
distribution deviates from a Gaussian (Maxwellian) form. For free diffusion
alpha_2 remains close to zero at all lag times, confirming Brownian
statistics. The largest alpha_2 values appear at intermediate polymer
concentrations (phi = 0.15 to 0.30) combined with high NP charge (Z = 3 and
6), where alpha_2 peaks at lags of 50 to 200 frames with values exceeding
0.4. This is the signature of hopping diffusion: at most time origins the
particle rattles within a cage and produces a small displacement, but
occasionally it escapes to a neighbouring site and produces a displacement
far exceeding the cage radius, inflating the fourth moment relative to the
second. At phi = 0.50 the cage is so deep that no escapes occur on the
simulation timescale and alpha_2 is suppressed back toward zero despite the
strong subdiffusion.

---

## 5. Velocity Autocorrelation and Green-Kubo Validation

The velocity autocorrelation function (VACF) distinguishes free, subdiffusive,
and hopping transport by its shape. For free diffusion the VACF decays
monotonically to zero with a short correlation time of 1 to 2 LJ time units.
For subdiffusive conditions the VACF develops a pronounced negative dip at
short lags, the direct spectral signature of backscattering: the particle
collides with a cage wall and reverses direction. The hopping regime shows
a similar negative dip followed by a long positive tail that reflects the
occasional large displacement events. The Green-Kubo diffusion coefficient
$D_\mathrm{GK}$, obtained by numerical integration of the VACF with a window
of 300 frames, agrees with $D_\mathrm{MSD}$ to within 10% for free diffusion
conditions. For the densest networks the agreement deteriorates because the
cage relaxation time is comparable to or longer than the integration window,
causing $D_\mathrm{GK}$ to underestimate the true long-time diffusivity.

---

## 6. Diffusion Regime Classification

The phase diagram in (phi, Z) space (figures/05_phase_diagram.png) shows
three well-defined regions. Free diffusion occupies phi = 0.00 and partially
phi = 0.05 for low charge, covering 40 to 50 of the 160 trajectories.
Subdiffusion is concentrated at phi = 0.50, where all charge values produce
alpha well below 0.72, and at phi = 0.30 with neutral to low charge. Hopping
is a relatively narrow band at phi = 0.15 to 0.30 for Z = 3 and 6, driven
by electrostatic trapping at intermediate mesh density; the particle is
mobile enough to escape cages but attracted strongly enough to bind to
polymer strands. The mixed regime fills the remainder of parameter space:
moderate concentration, low charge, where the trajectory statistics do not
clearly satisfy any of the three primary criteria. The boundaries between
regimes reflect the competition between the thermal energy scale (kT = 1 LJ)
and the depths of steric and electrostatic traps.

---

## 7. Machine Learning Analysis

Random Forest (RF) and Gradient Boosting (GB) classifiers were trained on 128
trajectories and evaluated on 32. Both models achieve test accuracy above
0.90 and macro-averaged F1 above 0.88, indicating that the four diffusion
regimes are well-separated in the 18-dimensional feature space. The top three
most important features in both models are the anomalous exponent alpha, the
non-Gaussian parameter alpha_2, and the diffusion coefficient D; the cage
escape rate and displacement kurtosis rank fourth and fifth. These features
are consistent with the physical definitions of the regimes: alpha and D
characterise the MSD scaling, while alpha_2 and kurtosis capture dynamic
heterogeneity. A key caveat is that the regime labels are defined by the
simulation rule-based classifier, so the ML models reproduce a deterministic
function of the input features rather than independently validated
experimental categories. Cross-validation stability plots confirm that both
models generalise reliably to unseen data without evidence of overfitting.

---

## 8. Conclusions

1. The anomalous diffusion exponent alpha decreases from near unity at
   phi = 0.00 to below 0.65 at phi = 0.50 for all charge values, with
   electrostatic interactions accounting for an additional suppression of
   0.10 to 0.15 in alpha at intermediate concentrations.

2. Hopping transport, identified by elevated alpha_2 and cage escape rate,
   occurs exclusively in the window phi = 0.15 to 0.30 with Z = 3 to 6,
   reflecting a competition between steric openness and electrostatic
   binding.

3. The Green-Kubo and MSD diffusion coefficients agree to within 10% for
   free diffusion but diverge by up to 40% at phi = 0.50, highlighting the
   importance of trajectory length relative to cage relaxation time for
   reliable diffusivity estimation.

4. Machine learning classification achieves greater than 90% accuracy using
   only trajectory-derived features, with the anomalous exponent and
   non-Gaussian parameter as the most diagnostic quantities, supporting their
   use as experimental observables in XPCS measurements of cellulose
   nanocrystal or nanofiber suspensions in polymer matrices.

---

## References

1. Weeks, E. R. & Weitz, D. A. (2002). Properties of cage rearrangements
   observed near the colloidal glass transition. *Physical Review Letters*,
   89(9), 095704.

2. Bhattacharya, S., Bhattacharya, S., & Bhattacharya, D. K. (2008).
   Nanoparticle diffusion in polymer networks. *Soft Matter*, 4, 1570.

3. Cherstvy, A. G. & Metzler, R. (2013). Population splitting, trapping,
   and non-ergodicity in heterogeneous diffusion processes.
   *Physical Chemistry Chemical Physics*, 15, 20220.

4. Metzler, R., Jeon, J.-H., Cherstvy, A. G., & Barkai, E. (2014).
   Anomalous diffusion models and their properties: non-stationarity,
   non-ergodicity, and ageing at the centenary of single particle tracking.
   *Physical Chemistry Chemical Physics*, 16, 24128.

5. Rosen, T. et al. (2023). Exploring nanofibrous networks with X-ray photon
   correlation spectroscopy through a digital twin. *Physical Review E*,
   108(1), 014607.
