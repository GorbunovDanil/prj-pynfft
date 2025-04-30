###############################################################################
# 2-D Gaussian random field on arbitrary points  (inverse NFFT, type-2)
#
#  * Base image  : python:3.11-buster   (same Debian base as 3.7-buster)
#  * Builds      : libNFFT 3  (+ FFTW, OpenMP)  ➜  /usr/local
#  * Python pkgs : cython==0.29.36  numpy  scipy  matplotlib  pyNFFT
#  * Script      : /opt/gauss_random_field_scatter.py
###############################################################################

# ---------- base image -------------------------------------------------------
    FROM python:3.11-buster

    # ---------- system packages --------------------------------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
            build-essential  git  gfortran \
            autoconf  automake  libtool  pkg-config \
            fftw3-dev  wget  ca-certificates \
        && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /opt
    
    # ---------- build NFFT 3 -----------------------------------------------------
    RUN git clone https://github.com/NFFT/nfft.git && \
        cd nfft && \
        ./bootstrap.sh && \
        ./configure --enable-all --enable-openmp --enable-nfsft --enable-nfft-chdr \
                    --prefix=/usr/local && \
        make -j"$(nproc)" && \
        make install && \
        cd .. && rm -rf nfft
    
    # ---------- Python dependencies ---------------------------------------------
    #  • Pin Cython 0.29.36  (Cython 3 breaks pyNFFT)
    RUN pip install --no-cache-dir \
            cython==0.29.36 \
            numpy  scipy  matplotlib
    
    # ---------- build & install pyNFFT ------------------------------------------
    RUN git clone https://github.com/ghisvail/pyNFFT.git && \
        cd pyNFFT && \
        python setup.py build_ext  \
            -I/usr/local/include  -L/usr/local/lib  -R/usr/local/lib && \
        python setup.py install && \
        cd .. && rm -rf pyNFFT
    
    # ---------- copy user script -------------------------------------------------
    COPY gauss_random_field_scatter.py /opt/gauss_random_field_scatter.py
    
    # ---------- head-less backend for Matplotlib ---------------------------------
    ENV MPLBACKEND=Agg
    
    # ---------- default command --------------------------------------------------
    CMD ["python", "/opt/gauss_random_field_scatter.py"]
    