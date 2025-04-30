#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumPy-IFFT  ×  PyNFFT-iFFT  –  accuracy check & plot (1-D)

Produces      fft_vs_pynfft.{pdf,png}      and prints error metrics.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynfft import NFFT

# ───────────── parameters ─────────────────────────────────────────
L      = 1.0
N      = 1024
LAMBDAS = [0.20, 0.10]

OVERSAMPLING = 2            # n = OVERSAMPLING · N   (≥2)
WINDOW_WIDTH = 8            # Kaiser-Bessel window width (≥6)
# ──────────────────────────────────────────────────────────────────

# ---------- grid & frequency vector ------------------------------
dx       = L / N
x_grid   = np.arange(N) * dx
x_nodes  = x_grid - 0.5                       # (-0.5, 0.5]

k_full   = np.arange(N)
omega    = 2*np.pi * (k_full - N*(k_full > N//2)) / L

# ---------- random *Hermitian* spectrum --------------------------
np.random.seed(0)
c_pos = (np.random.normal(0, 1, N//2 + 1)
         + 1j*np.random.normal(0, 1, N//2 + 1))

c_pos[0]      = c_pos[0].real      + 0j       # k = 0      must be real
c_pos[-1]     = c_pos[-1].real     + 0j       # k = N/2    must be real

c_full              = np.zeros(N, dtype=np.complex128)
c_full[:N//2+1]     = c_pos
c_full[N//2+1:]     = np.conj(c_pos[1:-1][::-1])        # Hermitian

def sqrt_spectrum(lmb):
    return np.exp(-0.5*(lmb*omega)**2)

# ---------- inverse NFFT helper ----------------------------------
def iNFFT(f_hat, nodes):
    """Inverse NFFT with proper coefficient order & 1/N scaling."""
    f_hat_shifted = np.fft.fftshift(f_hat)              # k = –N/2 … N/2−1

    plan = NFFT(N=(len(f_hat),), M=len(nodes),
                n=(OVERSAMPLING*len(f_hat),),
                m=WINDOW_WIDTH,
                dtype=np.complex128)

    plan.x = nodes
    plan.precompute()
    plan.f_hat[:] = f_hat_shifted
    plan.trafo()

    return plan.f / len(f_hat)                          # normalise like NumPy

# ---------- comparison + plotting -------------------------------
rows = len(LAMBDAS)
fig, axes = plt.subplots(rows, 1, figsize=(10, 6), sharex=True)

for ax, lam in zip(axes, LAMBDAS):
    f_hat  = c_full * sqrt_spectrum(lam)

    f_np   = np.fft.ifft(f_hat).real
    f_nfft = iNFFT(f_hat, x_nodes).real

    delta   = f_np - f_nfft
    l2_rel  = np.linalg.norm(delta) / np.linalg.norm(f_np)
    max_abs = np.max(np.abs(delta))

    print(f"λ = {lam:4.2f} | L2-rel = {l2_rel:9.3e} | "
          f"max-abs = {max_abs:9.3e}")

    ax.plot(x_grid, f_np,   'b',  lw=1.0, label="NumPy (IFFT)")
    ax.plot(x_grid, f_nfft, 'r--',lw=0.9, label="PyNFFT (iNFFT)")
    ax.set_title(f"λ = {lam:.2f}")
    ax.set_ylabel("field value")
    ax.grid(True)
    ax.legend()

fig.tight_layout()
fig.savefig("fft_vs_pynfft.pdf")
fig.savefig("fft_vs_pynfft.png", dpi=150)

# ---------- self-test -------------------------------------------
assert max_abs < 1e-12, (
    "\n*** PyNFFT still disagrees with NumPy by > 1e-12 – "
    "check that this file replaced the old one in the container. ***")

print("\nGraphs saved as  fft_vs_pynfft.{pdf,png}")
