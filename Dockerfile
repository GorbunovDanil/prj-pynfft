# ---------- base image ---------------------------------------------------------
    FROM python:3.7-buster

    # ---------- system packages ----------------------------------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
            build-essential git autoconf automake libtool \
            fftw3-dev pkg-config wget ca-certificates \
        && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /opt
    
    # ---------- build NFFT ---------------------------------------------------------
    RUN git clone https://github.com/NFFT/nfft.git && \
        cd nfft && \
        ./bootstrap.sh && \
        ./configure --enable-all --enable-openmp --enable-nfsft --enable-nfft-chdr --prefix=/usr/local && \
        make -j$(nproc) && \
        make install
    
    # ---------- install PyNFFT + plotting -----------------------------------------
    RUN git clone https://github.com/ghisvail/pyNFFT.git && \
        cd pyNFFT && \
        pip install --no-cache-dir cython==0.29.36 matplotlib && \
        python setup.py build_ext -I /usr/local/include -L /usr/local/lib -R /usr/local/lib && \
        python setup.py install
    
    # ---------- copy user script ---------------------------------------------------
    #  ❱❱❱ název souboru změněn na compare.py
    COPY compare.py /opt/compare.py
    
    # ---------- headless backend pro Matplotlib ------------------------------------
    ENV MPLBACKEND=Agg           
    
    # ---------- default command ----------------------------------------------------
    #  ❱❱❱ spouštíme nový skript
    CMD ["python", "/opt/compare.py"]
    
