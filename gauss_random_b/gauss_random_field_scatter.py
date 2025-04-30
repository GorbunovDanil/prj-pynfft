#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian-correlated random field on an arbitrary 2-D point cloud
generated via inverse NFFT (type-2).

Outputs
    • sample_points.png
    • empirical_covariance.png
Requires
    numpy · scipy · matplotlib · pyNFFT   (built against libnfft3)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynfft import NFFT
from scipy.spatial.distance import pdist, squareform
import logging, time, os

# ───────────── logging ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
# ───────────────────────────────────────────────────────────────────

# ───────────── parameters ─────────────────────────────────────────
Nx, Ny   = 256, 256      # spectral grid
Lx, Ly   = 1.0, 1.0      # domain size
lam_corr = 0.07          # Gaussian correlation length λ
M        = 10_000        # evaluation points
rng_seed = 42
# ───────────────────────────────────────────────────────────────────

rng = np.random.default_rng(rng_seed)
dx, dy = Lx/Nx, Ly/Ny

# 1) sample points (uniform random)  →  shift to (−0.5,0.5]
x_s = rng.uniform(0, Lx, M)
y_s = rng.uniform(0, Ly, M)
nodes = np.column_stack((x_s - 0.5*Lx,
                         y_s - 0.5*Ly))

log.info("Sample pts : %d  – node box [%+.2f,%+.2f]",
         M, nodes.min(), nodes.max())

# 2) k-space grid
kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
k_mag  = np.hypot(KX, KY)

S_k = np.exp(-0.5*(lam_corr*k_mag)**2)          # Gaussian spectrum

# 3) random complex coeffs in *natural* order
xi    = rng.normal(size=(Nx, Ny)) + 1j*rng.normal(size=(Nx, Ny))
f_hat = np.sqrt(S_k) * xi

# 3a) impose Hermitian symmetry  (still in natural indexing)
mirror = np.fft.ifftshift(np.conj(np.fft.ifftshift(f_hat))[::-1, ::-1])
f_hat  = 0.5*(f_hat + mirror)

herm_err = np.max(np.abs(f_hat - mirror))
log.info("Hermitian  : max|F − F*| = %.3e  (OK ≤1e-12)", herm_err)

# 3b) reorder for libNFFT:  fftshift  +  roll(-1) on each axis
f_hat_nfft = np.fft.fftshift(f_hat, axes=(0,1))
f_hat_nfft = np.roll(f_hat_nfft, -1, axis=0)
f_hat_nfft = np.roll(f_hat_nfft, -1, axis=1)

# 4) inverse NFFT
plan = NFFT(N=(Nx, Ny), M=M,
            n=(2*Nx, 2*Ny), m=8, dtype=np.complex128)
plan.x = nodes
plan.precompute()
plan.f_hat[:] = f_hat_nfft
plan.trafo()

field = plan.f.real
field -= field.mean()

log.info("Field      : mean %.3e, std %.3e", field.mean(), field.std())

# 5) empirical covariance vs theory (sub-sample to save RAM)
sub = 2000
idx = rng.choice(M, sub, replace=False)
coords = np.column_stack((x_s[idx], y_s[idx]))
d = squareform(pdist(coords))
prod = field[idx][:,None] * field[idx][None,:]

r_bins = np.linspace(0, 0.5*np.hypot(Lx, Ly), 40)
emp = np.zeros_like(r_bins)
for i in range(len(r_bins)-1):
    mask = (d>=r_bins[i]) & (d<r_bins[i+1])
    emp[i] = prod[mask].mean() if mask.any() else np.nan
emp[-1] = np.nan
rho = np.exp(-0.5*(r_bins/lam_corr)**2)

# 6) plots ----------------------------------------------------------
plt.figure(figsize=(5,4))
plt.scatter(x_s, y_s, c=field, s=8, cmap='viridis', linewidths=0)
plt.colorbar(label='f(x)')
plt.title(f'Gaussian field on {M} arbitrary points\nstd={field.std():.2e}')
plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
plt.tight_layout(); plt.savefig('sample_points.png', dpi=150)
log.info("Saved      : sample_points.png")

plt.figure(figsize=(5,4))
plt.semilogy(r_bins[:-1], emp[:-1], 'o', label='empirical')
plt.semilogy(r_bins, rho, 'k--', label='theoretical')
plt.xlabel('r'); plt.ylabel('⟨f(x) f(x+r)⟩')
plt.title('Covariance: empirical vs Gaussian')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout(); plt.savefig('empirical_covariance.png', dpi=150)
log.info("Saved      : empirical_covariance.png")

log.info("Done  (pid %d, cwd %s, time %.2fs)",
         os.getpid(), os.getcwd(), time.perf_counter())
