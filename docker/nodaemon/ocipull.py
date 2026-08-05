"""Pull an image's layers from a v2 registry over plain HTTPS — no daemon.

Usage: python ocipull.py <repo> <tag> <destdir>
       python ocipull.py library/ubuntu 22.04 /tmp/ub
"""
import hashlib
import json
import os
import sys
import urllib.request

REG = "https://registry-1.docker.io"
AUTH = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{}:pull"
MANIFEST_ACCEPT = ",".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])


def get(url, token=None, accept=None, binary=False):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read() if binary else json.loads(r.read())


def pull(repo, tag, dest, arch="amd64"):
    os.makedirs(dest, exist_ok=True)
    token = get(AUTH.format(repo))["token"]
    man = get(f"{REG}/v2/{repo}/manifests/{tag}", token, MANIFEST_ACCEPT)

    if man.get("manifests"):                       # multi-arch index
        for m in man["manifests"]:
            p = m.get("platform", {})
            if p.get("architecture") == arch and p.get("os") == "linux":
                man = get(f"{REG}/v2/{repo}/manifests/{m['digest']}",
                          token, MANIFEST_ACCEPT)
                break
        else:
            raise SystemExit(f"no linux/{arch} in index")

    cfg_d = man["config"]["digest"]
    cfg = get(f"{REG}/v2/{repo}/blobs/{cfg_d}", token, binary=True)
    open(os.path.join(dest, "config.json"), "wb").write(cfg)

    layers = []
    for i, l in enumerate(man["layers"]):
        d = l["digest"]
        out = os.path.join(dest, d.replace(":", "_") + ".tar.gz")
        if not os.path.exists(out) or \
           hashlib.sha256(open(out, "rb").read()).hexdigest() != d.split(":")[1]:
            print(f"  layer {i+1}/{len(man['layers'])} "
                  f"{l['size']/1e6:.0f} MB {d[:19]}", flush=True)
            blob = get(f"{REG}/v2/{repo}/blobs/{d}", token, binary=True)
            got = "sha256:" + hashlib.sha256(blob).hexdigest()
            if got != d:
                raise SystemExit(f"digest mismatch: {got} != {d}")
            open(out, "wb").write(blob)
        layers.append(out)
    json.dump({"repo": repo, "tag": tag, "layers": layers,
               "config": cfg_d, "manifest": man},
              open(os.path.join(dest, "pull.json"), "w"), indent=1)
    print(f"  pulled {len(layers)} layers -> {dest}")
    return layers


if __name__ == "__main__":
    pull(sys.argv[1], sys.argv[2], sys.argv[3])
