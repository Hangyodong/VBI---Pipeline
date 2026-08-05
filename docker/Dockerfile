# VBI-SBI HCP pipeline — RWWEIB_2CPL on cuBNM.
#
# Build with docker/build.sh, not `docker build` directly: the cuBNM fork lives
# outside this repo and the script stages both source trees via `git archive`
# into a small context. See docker/build.sh for the one command.
#
# Two stages. The build stage needs the CUDA toolkit (nvcc) but NOT a GPU —
# cuBNM's setup.py gates the GPU build on `find_executable('nvcc')` alone, so
# this builds fine on a CPU-only machine and runs later on GPUs.

ARG CUDA_VER=12.4.1
ARG UBUNTU_VER=22.04

# ---------------------------------------------------------------------------
# Stage 1 — compile the cuBNM fork into a wheel
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu${UBUNTU_VER} AS build

ARG PYTHON_VER=3.13
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Miniconda gives us both the Python version and GSL, which cuBNM links
# statically (setup.py looks for libgsl.a / libgslcblas.a).
RUN curl -sS -o /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda install -y python=${PYTHON_VER} \
    && conda install -y --no-deps -c conda-forge gsl=2.7 \
    && conda clean -afy
RUN python -m pip install --no-cache-dir build

# cubnm-src.tar is `git archive HEAD` of the fork, so the build sees a clean
# tree with no .git. setuptools_scm has no repo to read, hence the fallback
# version — the wheel is identified by CUBNM_COMMIT below, not by this string.
ARG CUBNM_COMMIT=unknown
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+fork.${CUBNM_COMMIT}
WORKDIR /src/cubnm
ADD cubnm-src.tar /src/cubnm/

# nvcc is invoked without -arch (cuBNM's setup.py does not pass one), so the
# result carries PTX and JIT-compiles on first run for the target GPU.
RUN python -m build --wheel . && ls -la dist/

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VER}-runtime-ubuntu${UBUNTU_VER} AS runtime

ARG PYTHON_VER=3.13
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS -o /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda install -y python=${PYTHON_VER} \
    && conda install -y --no-deps -c conda-forge gsl=2.7 \
    && conda clean -afy

# Versions pinned to the environment that produced the recorded results.
# cupy, brainspace and tvb are deliberately absent: every import of them in
# this repo is inside a function or a try block, and dropping them was verified
# by importing all 50 library modules with those three names blocked. cupy
# alone would add ~1.5 GB and only serves the Wilson-Cowan path, which
# INFERENCE_MODEL=rwweib2 never takes. `pip install cupy-cuda12x brainspace`
# in a derived image if you need extract_gradient_cortex.py or the WC engine.
RUN python -m pip install --no-cache-dir \
        torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN python -m pip install --no-cache-dir \
        "numpy==2.3.5" \
        "scipy==1.17.1" \
        "pandas==2.3.3" \
        "matplotlib==3.10.6" \
        "scikit-learn==1.7.2" \
        "sbi==0.26.1" \
        "h5py==3.15.1" \
        "tqdm==4.67.1"

COPY --from=build /src/cubnm/dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Mirrors the LD_LIBRARY_PATH fix in cuBNM's own stable Dockerfile: without it
# the nvrtc / cublas shared objects shipped inside the pip nvidia-* packages
# are not found at runtime.
ENV LD_LIBRARY_PATH=/opt/conda/lib/python${PYTHON_VER}/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/lib/python${PYTHON_VER}/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}

# vbi-src.tar is `git archive HEAD` of this repo: the 60 tracked files only, so
# output_hcp/ and the untracked local-only scripts never enter the image.
WORKDIR /app
ADD vbi-src.tar /app/

# HCP_FC.mat is not in the repo (1.1 GB, over GitHub's limit) and is not baked
# in either — mount the real HCP_Data over this one at run time:
#   docker run --gpus all -v /path/to/HCP_Data:/app/HCP_Data ...
# Results land in /app/output_hcp, so mount that too if you want to keep them.
VOLUME ["/app/HCP_Data", "/app/output_hcp"]

ARG VBI_COMMIT=unknown
ARG CUBNM_COMMIT=unknown
LABEL org.opencontainers.image.title="vbi-hcp" \
      org.opencontainers.image.description="VBI-SBI HCP pipeline, RWWEIB_2CPL on cuBNM" \
      vbi.commit="${VBI_COMMIT}" \
      vbi.cubnm.commit="${CUBNM_COMMIT}"

# SMOKE defaults to 1 inside main_HCP.py, so a bare `docker run` is a tiny toy
# run. Override with -e SMOKE=0 for the real thing.
ENTRYPOINT ["python"]
CMD ["main_HCP.py"]
