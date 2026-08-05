"""Stage B — assemble a `docker load`-able image tarball without a daemon.

Takes the ubuntu base layers pulled by ocipull.py, the conda env built by
mkenv.sh, and this repo, and writes a docker-archive tarball:

    manifest.json           [{Config, RepoTags, Layers}]
    <cfg-sha>.json          image config; rootfs.diff_ids = sha256 of each
                            UNCOMPRESSED layer tar, in order
    layer-N.tar             uncompressed layer tars

`docker load -i <out>` accepts exactly this. Nothing here needs root, a daemon,
or a container runtime — it is all tar and JSON.
"""
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

BUILD = "/var/tmp/vbi-img-build"
ENVP = f"{BUILD}/env"
REPO = "/scratch/home/wog3597/vbi"
CUBNM_SRC = "/scratch/home/wog3597/cubnm_build"
BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{BUILD}/base"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{BUILD}/vbi-hcp.tar"
TAG = os.environ.get("TAG", "vbi-hcp:latest")
STAGE = f"{BUILD}/stage"


def sh(cmd, **kw):
    print(f"  $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def sha256_file(path, bufsize=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def gunzip_to(src, dst):
    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, 8 << 20)


def main():
    os.makedirs(STAGE, exist_ok=True)
    layers = []            # (path_to_uncompressed_tar, description)

    # ---- base layers: ungzip the registry blobs, keep them byte-identical ----
    pull = json.load(open(f"{BASE_DIR}/pull.json"))
    for i, blob in enumerate(pull["layers"]):
        out = f"{STAGE}/base-{i}.tar"
        if not os.path.exists(out):
            print(f"[base] gunzip layer {i}", flush=True)
            gunzip_to(blob, out)
        layers.append((out, f"base {pull['repo']}:{pull['tag']} layer {i}"))

    # ---- conda env -> /opt/conda ----
    # --dest-prefix rewrites every hardcoded build path to the final location,
    # so the env works on extraction with no conda-unpack step at runtime.
    env_tar = f"{STAGE}/env.tar"
    if not os.path.exists(env_tar):
        print("[env] conda-pack -> /opt/conda", flush=True)
        sh(f"conda-pack -p {ENVP} --dest-prefix /opt/conda "
           f"--format tar -o {env_tar} --n-threads -1 --ignore-missing-files")
    # conda-pack emits paths relative to the env root; re-root them at opt/conda.
    #
    # Extract and re-tar with GNU tar rather than rewriting TarInfo names in
    # place. Prefixing "opt/conda/" pushes 2,719 of these paths past the
    # 100-character ustar name limit, and Python's tarfile silently wrote them
    # back under their un-prefixed name — scipy's _arpacklib .so among them, so
    # the image imported numpy happily and then died inside sklearn. GNU tar
    # emits long names correctly, so let it own the format.
    env_layer = f"{STAGE}/layer-env.tar"
    env_root = f"{STAGE}/envroot"
    if not os.path.exists(env_layer):
        print("[env] extract + re-tar under opt/conda", flush=True)
        shutil.rmtree(env_root, ignore_errors=True)
        os.makedirs(f"{env_root}/opt/conda")
        sh(f"tar -xf {env_tar} -C {env_root}/opt/conda")
        sh(f"tar -cf {env_layer} --owner=0 --group=0 --numeric-owner "
           f"-C {env_root} opt")
    layers.append((env_layer, "conda env at /opt/conda"))

    # ---- repo -> /app (tracked files only) ----
    app_layer = f"{STAGE}/layer-app.tar"
    app_root = f"{STAGE}/approot"
    if not os.path.exists(app_layer):
        print("[app] git archive HEAD -> /app", flush=True)
        shutil.rmtree(app_root, ignore_errors=True)
        os.makedirs(f"{app_root}/app")
        sh(f"git -C {REPO} archive HEAD | tar -x -C {app_root}/app")
        sh(f"tar -cf {app_layer} --owner=0 --group=0 --numeric-owner "
           f"-C {app_root} app")
    layers.append((app_layer, "repo at /app"))

    # ---- config ----
    diff_ids = []
    for path, desc in layers:
        diff_ids.append("sha256:" + sha256_file(path))
        print(f"  layer {os.path.basename(path):22s} "
              f"{os.path.getsize(path)/1e6:8.1f} MB  {desc}", flush=True)

    vbi_commit = subprocess.check_output(
        f"git -C {REPO} rev-parse --short HEAD", shell=True, text=True).strip()
    cubnm_commit = subprocess.check_output(
        f"git -C {CUBNM_SRC} rev-parse --short HEAD", shell=True, text=True).strip()
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    env_path = ("PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin")
    cfg = {
        "architecture": "amd64",
        "os": "linux",
        "created": created,
        "config": {
            "Env": [
                env_path,
                # cuBNM links CUDA and GSL statically, so the only CUDA piece
                # needed at run time is the driver, injected by --gpus all.
                "LD_LIBRARY_PATH=/opt/conda/lib",
                "PYTHONUNBUFFERED=1",
            ],
            "Entrypoint": ["/opt/conda/bin/python"],
            "Cmd": ["main_HCP.py"],
            "WorkingDir": "/app",
            "Volumes": {"/app/HCP_Data": {}, "/app/output_hcp": {}},
            "Labels": {
                "org.opencontainers.image.title": "vbi-hcp",
                "org.opencontainers.image.description":
                    "VBI-SBI HCP pipeline, RWWEIB_2CPL on cuBNM",
                "vbi.commit": vbi_commit,
                "vbi.cubnm.commit": cubnm_commit,
            },
        },
        "rootfs": {"type": "layers", "diff_ids": diff_ids},
        "history": [{"created": created, "created_by": d}
                    for _, d in layers],
    }
    cfg_bytes = json.dumps(cfg, indent=1).encode()
    cfg_name = hashlib.sha256(cfg_bytes).hexdigest() + ".json"
    open(f"{STAGE}/{cfg_name}", "wb").write(cfg_bytes)

    manifest = [{
        "Config": cfg_name,
        "RepoTags": [TAG],
        "Layers": [os.path.basename(p) for p, _ in layers],
    }]
    open(f"{STAGE}/manifest.json", "w").write(json.dumps(manifest, indent=1))

    # ---- final archive ----
    print(f"[out] writing {OUT}", flush=True)
    with tarfile.open(OUT, "w") as tar:
        tar.add(f"{STAGE}/manifest.json", arcname="manifest.json")
        tar.add(f"{STAGE}/{cfg_name}", arcname=cfg_name)
        for p, _ in layers:
            tar.add(p, arcname=os.path.basename(p))
    print(f"done: {OUT}  {os.path.getsize(OUT)/1e9:.2f} GB")
    print(f"      vbi={vbi_commit} cubnm={cubnm_commit} tag={TAG}")
    print(f"load: docker load -i {OUT}")


if __name__ == "__main__":
    main()
