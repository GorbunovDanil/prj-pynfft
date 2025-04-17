#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Porovnání NumPy‑IFFT vs. PyNFFT na TÉŽE pravidelné mřížce.
"""

import numpy as np
import matplotlib.pyplot as plt
from pynfft import NFFT

# ------------------------ mřížka a frekvence ------------------------
L = 1.0
N = 1024                  
dx = L / N
x_grid  = np.arange(N) * dx           # 0 … 1‑dx
x_shift = x_grid - 0.5                # interval (‑0.5,0.5]

k_full  = np.arange(N)                # 0…N‑1
omega   = 2*np.pi*(k_full - N*(k_full>N//2))/L   # správné kladné + záporné ω

# ------------------------ pomocné funkce ----------------------------
def spectrum(lambda_corr):
    """náhodné symetrické spektrum v NE‑shiftovaném pořadí (k=0..N-1)."""
    # náhodné gaussovské komplexy jen pro kladné k (0..N/2)
    np.random.seed(0)
    c_pos = np.random.normal(0,1,N//2+1) + 1j*np.random.normal(0,1,N//2+1)

    fhat = np.zeros(N, dtype=np.complex128)
    fhat[:N//2+1] = c_pos
    fhat[N//2+1:] = np.conj(c_pos[1:-1][::-1])   # FF[-k] = conj(FF[k])

    S = np.exp(-0.5 * (lambda_corr * omega)**2)
    return fhat * np.sqrt(S)

def slow_ifft(fhat, x_nodes):
    """Přímý součet 1/N Σ fhat_k e^{i 2π k x} se stejným pořadím k=0..N-1."""
    k = np.arange(N)
    phase = np.exp(1j * 2*np.pi * k[:,None] * x_nodes[None,:])
    return (fhat[:,None] * phase).sum(axis=0) / N

def field_pynfft(fhat, x_nodes):
    plan = NFFT(N, len(x_nodes))
    plan.x = x_nodes                    # (-0.5,0.5]
    plan.precompute()
    plan.f_hat[:] = fhat                # NE‑shiftované
    plan.trafo()
    return plan.f.copy()

# ------------------------ výpočet -----------------------------------
lambda_vals = [1/5, 1/10]
plt.figure(figsize=(10,6))

for i, lam in enumerate(lambda_vals, 1):
    f_hat = spectrum(lam)

    f_np  = slow_ifft(f_hat, x_shift).real
    f_nf  = field_pynfft(f_hat, x_shift).real

    plt.subplot(2,1,i)
    plt.title(f"λ = {lam:.2f}")
    plt.plot(x_grid, f_np, 'b',  lw=1, label='NumPy (exact)')
    plt.plot(x_grid, f_nf, 'r--',lw=1, label='PyNFFT')
    plt.ylabel("hodnota pole"); plt.grid(); plt.legend()

plt.xlabel("x")
plt.tight_layout()
plt.savefig("/data/fft_vs_pynfft.png", dpi=150)
plt.show()
