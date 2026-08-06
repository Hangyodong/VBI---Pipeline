# Building the image without a Docker daemon

`docker/build.sh` is the normal path. Use this one only when you cannot reach a
Docker daemon — on a shared HPC node, for instance, where the account is not in
the `docker` group and `/var/run/docker.sock` is `root:docker 0660`.

Rootless Docker is not a way out there either: `rootlesskit` refuses to start
without subuid/subgid ranges, and writing `/etc/subuid` needs root, which is
the thing you do not have.

So this path never runs a container at all. It builds the environment natively,
then writes the image as tar and JSON by hand.

```bash
./mkenv.sh                                    # ~30 min: conda env + cuBNM wheel
python ocipull.py library/ubuntu 22.04 /var/tmp/vbi-img-build/base
python mkimage.py /var/tmp/vbi-img-build/base /var/tmp/vbi-img-build/vbi-hcp.tar
docker load -i /var/tmp/vbi-img-build/vbi-hcp.tar     # on any machine with docker
```

| Script | Does |
|---|---|
| `mkenv.sh` | conda env (python 3.13 + gsl), pinned python stack, compiles the cuBNM fork with nvcc, verifies |
| `ocipull.py` | pulls base image layers straight from a v2 registry over HTTPS |
| `mkimage.py` | packs the env at `/opt/conda` and the repo at `/app`, emits a `docker load`-able archive |

## Why the base image is just `ubuntu:22.04`

`ldd` on the built `cubnm/core*.so` resolves nothing beyond glibc and the
conda-provided `libstdc++` / `libgomp` / `libgcc_s`. cuBNM links CUDA and GSL
**statically** (`-lcudart_static`, `libgsl.a`), so no CUDA base image is
needed — several GB saved. The only CUDA piece required at run time is the
driver, which `--gpus all` injects.

## Traps this hit, so you do not have to

- **The host's `~/.local` shadows the new env.** Both are python 3.13, so pip
  called every dependency "already satisfied" and installed nothing, and the
  interpreter then imported the host's torch. `PYTHONNOUSERSITE=1` everywhere.
- **Installing torch separately from `sbi` loses the pin.** pip re-resolves
  torch off PyPI to satisfy sbi and lands a cu13 build, which needs a much
  newer driver than the cu124 stack this pipeline was validated against. One
  resolve, torch pinned, pytorch.org as an *extra* index.
- **Python's `tarfile` silently truncates long paths.** Prefixing `opt/conda/`
  pushed 2,719 paths past the 100-character ustar name limit and they were
  written back un-prefixed — scipy's `_arpacklib` .so among them, so the image
  imported numpy happily and then died inside sklearn. `mkimage.py` shells out
  to GNU tar for the layer tars.
- **`conda-pack --dest-prefix /opt/conda`** rewrites the build path into the
  final one, so no `conda-unpack` step is needed at run time.

## Verifying without a daemon

`unshare -U -r -m` plus `chroot` runs the extracted image on the host. Bind
mounts need `mount --make-rprivate /` first, and only the NVIDIA driver libs
should be bound in — binding all of `/lib64` overwrites the container's glibc
and nothing starts.

```bash
cd /var/tmp/vbi-img-build && mkdir rootfs
for l in base-0.tar layer-env.tar layer-app.tar; do tar -xf stage/$l -C rootfs; done
mkdir -p nvlib && cp -L /lib64/libcuda.so.1 /lib64/libnvidia-ml.so.1 nvlib/

unshare -U -r -m bash -c '
  mount --make-rprivate /
  for d in dev proc sys; do mount --rbind /$d rootfs/$d; done
  mount --rbind $PWD/nvlib rootfs/nvlib
  chroot rootfs env LD_LIBRARY_PATH=/nvlib:/opt/conda/lib /opt/conda/bin/python -c "
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
```

Check the layers too. Each layer's sha256 — after gunzip, if it is compressed —
must equal the matching entry in the config's `rootfs.diff_ids`, or `docker load`
rejects the archive. That catches a truncated transfer or a mis-assembled
manifest before it reaches the machine that has a daemon.

### Result for the archive built on 2026-08-05

| Check | Outcome |
|---|---|
| `sha256sum -c vbi-hcp.tar.sha256` | OK |
| Layer `diff_id`s (3/3) | match |
| chroot run | exit 0 |
| GPU | `torch.cuda.is_available()` True, `NVIDIA A10` |
| Fork kernel | `from cubnm.sim import RWWEIB_2CPLSimGroup` imports; `RWWEIB_2CPLModel::run_simulations_gpu` present in `core*.so` |

Versions seen inside: torch 2.6.0+cu124, numpy 2.3.5, scipy 1.17.1,
scikit-learn 1.7.2, sbi 0.26.1, h5py 3.15.1.

Two things to know before running it elsewhere:

- **`cubnm.__version__` reports `0+unknown`.** The Dockerfile sets
  `SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+fork.<commit>`, but this path does not go
  through it. Identify the fork by the image label `vbi.cubnm.commit` instead.
- **`LD_LIBRARY_PATH` is `/opt/conda/lib` only.** The Dockerfile's second stage
  also prepends the pip `nvidia-*` nvrtc and cublas directories; this image does
  not. torch uses its own bundled libraries and ran fine, but if an
  `undefined symbol` shows up, pass them at run time:
  ```
  -e LD_LIBRARY_PATH=/opt/conda/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/lib/python3.13/site-packages/nvidia/cublas/lib:/opt/conda/lib
  ```

`HCP_FC.mat` is not in the image — `/app/HCP_Data` holds only
`HCP_CABNP381_SC_first100.mat`, `basis_cortex.npy`,
`gradient_subjects_cortex.npy` and `myelin_subjects.npy`. Mount the real
`HCP_Data` over it or the FC load fails.
