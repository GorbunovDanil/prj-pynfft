#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-D náhodné pole s Gaussovskou korelací (spectral synthesis, pravidelná mřížka).

* vygeneruje spektrum  S(k)=exp[-(λ|k|)²/2]  × bílý šum,
* zajistí Hermitovskou symetrii  ⇒  reálné pole,
* provede inverzní 2-D FFT,
* uloží obrázky realizace a empirické korelace,
* vypisuje detailní logy.

Výstupní soubory
----------------
    field_realisation_seed.png       –  konkrétní realizace (seed = 42)
    empirical_correlation.png        –  ρ(Δx) vs. teoretická Gaussova křivka
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")        
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import logging, time, os, sys

# ─────────────────────────────── LOGGING ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
# ────────────────────────────────────────────────────────────────────────

# ───────────────────────────── PARAMETRY ───────────────────────────────
Nx, Ny   = 256, 256      # počet bodů v osách x, y
Lx, Ly   = 1.0, 1.0      # fyzická délka domény
lam_corr = 0.07          # Gaussovská korelační délka λ  (v jednotkách L)
rng_seed = 42            # None ⇒ plně náhodné

dx, dy   = Lx / Nx, Ly / Ny
x        = np.arange(Nx) * dx
y        = np.arange(Ny) * dy

log.info("Grid       : %d × %d  (dx = %.4g, dy = %.4g)", Nx, Ny, dx, dy)
log.info("Domain     : Lx = %.3g, Ly = %.3g", Lx, Ly)
log.info("λ (corr)   : %.3g", lam_corr)
log.info("rng_seed   : %s", rng_seed)

t0 = time.perf_counter()

# ────────────────────── k-prostorová mřížka ────────────────────────────
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
k_mag  = np.hypot(KX, KY)

log.info("k-space    : |k| min %.3g, max %.3g", k_mag.min(), k_mag.max())

# ────────────────────── cílová spektrální hustota ──────────────────────
S_k = np.exp(-0.5 * (lam_corr * k_mag) ** 2)
log.info("S(k)       : min %.3e, max %.3e", S_k.min(), S_k.max())

# ────────────────────── náhodné komplexní spektrum ─────────────────────
np.random.seed(rng_seed)
xi = (np.random.normal(size=(Nx, Ny)) +
      1j * np.random.normal(size=(Nx, Ny)))          # bílý šum
f_hat = np.sqrt(S_k) * xi                            # amplituda × fáze

# ────────────────────── vynucená Hermitovská symetrie ───────────────────
mirror = np.fft.ifftshift(np.conj(np.fft.ifftshift(f_hat))[::-1, ::-1])
f_hat  = 0.5 * (f_hat + mirror)            

# kontrola – po symetrizaci
mirror = np.fft.ifftshift(np.conj(np.fft.ifftshift(f_hat))[::-1, ::-1])
herm_err = np.max(np.abs(f_hat - mirror))
log.info("Hermitian  : max|F − F*| = %.3e  (OK ≤ 1e-12)", herm_err)

# kolik energie zůstalo po Gaussově filtru
spec_rms = np.sqrt(np.mean(np.abs(f_hat) ** 2))
log.info("Spectrum σ : %.3e  ⇒  očekávaná σ(field) ≈ %.3e",
         spec_rms, spec_rms / np.sqrt(2))

# ────────────────────── inverzní FFT → pole v x-prostoru ───────────────
field = np.fft.ifft2(f_hat).real
field -= field.mean()
log.info("Field      : mean = %.3e, std = %.3e", field.mean(), field.std())

# ────────────────────── empirická autokorelace ─────────────────────────
corr = fftconvolve(field, field[::-1, ::-1], mode="same")
corr /= corr[Nx//2, Ny//2]
log.info("Corr check : ρ(0) = %.3f (should be 1.0)", corr[Nx//2, Ny//2])

# ────────────────────── vizualizace ────────────────────────────────────
real_png = f"field_realisation{'_seed' if rng_seed is not None else ''}.png"
plt.figure(figsize=(6, 5))
plt.imshow(field, extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis")
plt.colorbar(label="field value")
plt.title(f"2-D Gaussian-correlated random field\n{Nx}×{Ny},  λ={lam_corr}")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout(); plt.savefig(real_png, dpi=150)
log.info("Saved      : %s", real_png)

plt.figure(figsize=(5, 4))
rho_x = corr[Nx//2, :]
plt.semilogy(x, rho_x, label="empirical ρ")
plt.semilogy(x, np.exp(-0.5 * ((x - 0.5*dx) / lam_corr) ** 2),
             "k--", label="theoretical ρ")
plt.xlabel("Δx"); plt.ylabel("corr")
plt.title("Empirical vs theoretical Gaussian correlation")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout(); plt.savefig("empirical_correlation.png", dpi=150)
log.info("Saved      : empirical_correlation.png")

log.info("Done in %.2f s  (pid %d, cwd %s)",
         time.perf_counter() - t0, os.getpid(), os.getcwd())
