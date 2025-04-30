#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumPy-IFFT  ×  PyNFFT-iFFT   —   accuracy check & plot (1-D)

*  Computes the inverse Fourier transform of the **same spectrum**
   in two different ways:

     ①  NumPy’s uniform inverse FFT (`np.fft.ifft`)
     ②  PyNFFT’s inverse **non-uniform** FFT (type-2 “iNFFT”)

*  Verifies that the two results are identical (to ~10-15)
*  Saves comparison plots to   `fft_vs_pynfft.{pdf,png}`
*  Prints error metrics to stdout and aborts if the mismatch
   exceeds **1 × 10-12**.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # run head-less (no X-server needed)
import matplotlib.pyplot as plt
from pynfft import NFFT               # pip install pynfft

# ─────────────────────────── USER PARAMETERS ──────────────────────────────
L            = 1.0       # length of the spatial interval   x ∈ [0, L)
N            = 1024      # number of grid points (must be even)
LAMBDAS      = [0.20, 0.10]           # Gaussian smoothing widths

OVERSAMPLING = 2         # n = OVERSAMPLING · N   (≥ 2  ⇒  good accuracy)
WINDOW_WIDTH = 8         # Kaiser-Bessel window width (≥ 6)
# ───────────────────────────────────────────────────────────────────────────

# 1) ─────────── regular grid in x and corresponding frequency vector ──────
dx       = L / N
x_grid   = np.arange(N) * dx          # equidistant grid in ⟨0, L)
x_nodes  = x_grid - 0.5               # PyNFFT expects nodes in (-0.5, 0.5]

k_full   = np.arange(N)               # 0 … N-1
# Convert to *signed* angular frequencies  ω_k  in the range [-πN/L, πN/L)
omega    = 2*np.pi * (k_full - N*(k_full > N//2)) / L

# 2) ─────────── build a random *Hermitian* spectrum so that the inverse
#                 transform is *real* (NumPy convention) ───────────────────
np.random.seed(0)                     # reproducibility
c_pos = (np.random.normal(0, 1, N//2 + 1) +
         1j*np.random.normal(0, 1, N//2 + 1))

# Enforce the Hermitian constraint for the two “special” bins:
c_pos[0]  = c_pos[0].real + 0j        # k = 0     ⇒ purely real
c_pos[-1] = c_pos[-1].real + 0j       # k = N/2   ⇒ purely real

# Full spectrum of length N (Hermitian: F[–k] = conj(F[k]))
c_full            = np.zeros(N, dtype=np.complex128)
c_full[:N//2+1]   = c_pos
c_full[N//2+1:]   = np.conj(c_pos[1:-1][::-1])

def sqrt_spectrum(lmb: float) -> np.ndarray:
    """Gaussian envelope √P(ω) that damps high frequencies."""
    return np.exp(-0.5 * (lmb * omega) ** 2)

# 3) ─────────── helper: inverse NFFT (type-2) with NumPy normalisation ────
def iNFFT(f_hat: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """
    Evaluate ∑_k f_hat[k] · exp(+i 2π k · x) / N   at arbitrary nodes.
    The coefficient vector is supplied in the NumPy ‘fftshift’ order
    k = −N/2 … N/2−1 and the result is scaled by 1/N so that it matches
    `np.fft.ifft`.
    """
    f_hat_shifted = np.fft.fftshift(f_hat)      # reorder to (−N/2 … N/2−1)

    # Create the NFFT plan
    plan = NFFT(N=(len(f_hat),),                # tuple! length of spectrum
                M=len(nodes),                   # number of target points
                n=(OVERSAMPLING * len(f_hat),), # oversampled FFT grid
                m=WINDOW_WIDTH,                 # window width
                dtype=np.complex128)

    plan.x = nodes             # non-uniform nodes in (-0.5, 0.5]
    plan.precompute()          # costly ≈ O(N log N) step
    plan.f_hat[:] = f_hat_shifted
    plan.trafo()               # run the inverse transform

    return plan.f / len(f_hat) # divide by N to mimic NumPy’s IFFT

# 4) ─────────── compare NumPy and PyNFFT for each λ ───────────────────────
fig, axes = plt.subplots(len(LAMBDAS), 1, figsize=(10, 6), sharex=True)

for ax, lam in zip(axes, LAMBDAS):
    f_hat  = c_full * sqrt_spectrum(lam)   # apply Gaussian filter

    f_np   = np.fft.ifft(f_hat).real       # reference solution
    f_nfft = iNFFT(f_hat, x_nodes).real    # NFFT solution

    delta   = f_np - f_nfft
    l2_rel  = np.linalg.norm(delta) / np.linalg.norm(f_np)
    max_abs = np.max(np.abs(delta))

    print(f"λ = {lam:4.2f} | "
          f"L2-rel = {l2_rel:9.3e} | "
          f"max-abs = {max_abs:9.3e}")

    # Plot both curves
    ax.plot(x_grid, f_np,   'b',  lw=1.0, label="NumPy (IFFT)")
    ax.plot(x_grid, f_nfft, 'r--',lw=0.9, label="PyNFFT (iNFFT)")
    ax.set_title(f"λ = {lam:.2f}")
    ax.set_ylabel("field value")
    ax.grid(True)
    ax.legend()

fig.tight_layout()
fig.savefig("fft_vs_pynfft.pdf")
fig.savefig("fft_vs_pynfft.png", dpi=150)

# 5) ─────────── automatic pass/fail check ─────────────────────────
assert max_abs < 1e-12, (
    "\n*** PyNFFT still disagrees with NumPy by > 1e-12 ***\n"
    "   → check that this *commented* file is the one actually run "
    "inside the container.")

print("\nGraphs saved as  fft_vs_pynfft.{pdf,png}")
