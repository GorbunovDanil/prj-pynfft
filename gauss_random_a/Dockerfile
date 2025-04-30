###############################################################################
# 2-D Gaussian-correlated random field (spectral synthesis)
#   • Base image        : slim Debian + Python 3.11
#   • System packages   : tool-chain + OpenBLAS / LAPACK (SciPy wheels use them)
#   • Python packages   : numpy, scipy, matplotlib
#   • Script copied to  : /opt/gauss_random_field_2d.py
###############################################################################

# ---------- base image -------------------------------------------------------
    FROM python:3.11-slim-bullseye

    # ---------- system packages --------------------------------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
            build-essential  gfortran  libopenblas-dev  liblapack-dev \
            ca-certificates  wget \
        && rm -rf /var/lib/apt/lists/*
    
    # ---------- working directory ------------------------------------------------
    WORKDIR /opt
    
    # ---------- Python dependencies ---------------------------------------------
    # Pre-built wheels exist for every platform → compile time is short.
    RUN pip install --no-cache-dir numpy scipy matplotlib
    
    # ---------- copy user script -------------------------------------------------
    COPY gauss_random_field_2d.py /opt/gauss_random_field_2d.py
    
    # ---------- headless backend for Matplotlib ----------------------------------
    ENV MPLBACKEND=Agg        
    
    # ---------- default command --------------------------------------------------
    CMD ["python", "/opt/gauss_random_field_2d.py"]
    