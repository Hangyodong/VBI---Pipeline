#!/usr/bin/env bash
# Build the VBI-SBI image.
#
#   ./docker/build.sh                      # uses ../cubnm_build for the fork
#   CUBNM_SRC=/path/to/cubnm ./docker/build.sh
#   IMAGE=myname:tag ./docker/build.sh
#
# The cuBNM fork lives outside this repo, and Docker cannot COPY above its
# build context. Rather than build from a parent directory (which would drag
# in 18 GB of run artifacts), this stages both trees as `git archive` tarballs
# into a throwaway context, so the context is ~150 MB of clean source with no
# .git and no output_hcp/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUBNM_SRC="${CUBNM_SRC:-$(dirname "$REPO_ROOT")/cubnm_build}"
IMAGE="${IMAGE:-vbi-hcp:latest}"
CTX="$REPO_ROOT/.docker-ctx"

command -v docker >/dev/null || { echo "error: docker not on PATH" >&2; exit 1; }
docker info >/dev/null 2>&1 || {
    echo "error: cannot reach the Docker daemon." >&2
    echo "  You are probably not in the 'docker' group. Check with: id" >&2
    echo "  Fix: sudo usermod -aG docker \$USER   (then log out and back in)" >&2
    exit 1
}
[ -d "$CUBNM_SRC/.git" ] || {
    echo "error: no git repo at CUBNM_SRC=$CUBNM_SRC" >&2
    echo "  The RWWEIB_2CPL model only exists in that fork; it is not on PyPI." >&2
    exit 1
}

# The fork's kernel changes are local-only commits. If its tree is dirty the
# image would silently not match the source, so refuse rather than guess.
if [ -n "$(git -C "$CUBNM_SRC" status --porcelain)" ]; then
    echo "error: cuBNM fork at $CUBNM_SRC has uncommitted changes." >&2
    echo "  git archive only sees committed state, so the image would not" >&2
    echo "  match your working tree. Commit or stash first." >&2
    exit 1
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
    echo "warning: this repo has uncommitted changes; they will NOT be in the image." >&2
fi

VBI_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
CUBNM_COMMIT="$(git -C "$CUBNM_SRC" rev-parse --short HEAD)"

rm -rf "$CTX"; mkdir -p "$CTX"
trap 'rm -rf "$CTX"' EXIT
git -C "$REPO_ROOT"  archive --format=tar HEAD -o "$CTX/vbi-src.tar"
git -C "$CUBNM_SRC"  archive --format=tar HEAD -o "$CTX/cubnm-src.tar"

echo "vbi    $VBI_COMMIT   ($(du -h "$CTX/vbi-src.tar"   | cut -f1))"
echo "cubnm  $CUBNM_COMMIT ($(du -h "$CTX/cubnm-src.tar" | cut -f1))"
echo "building $IMAGE — first build pulls ~3 GB of CUDA base images and"
echo "compiles cuBNM from source; expect 30-60 min."

docker build \
    -f "$REPO_ROOT/docker/Dockerfile" \
    -t "$IMAGE" \
    --build-arg "VBI_COMMIT=$VBI_COMMIT" \
    --build-arg "CUBNM_COMMIT=$CUBNM_COMMIT" \
    "$CTX"

echo
echo "built: $IMAGE   $(docker image inspect "$IMAGE" --format '{{.Size}}' | numfmt --to=iec)"
echo
echo "smoke test (needs a GPU node):"
echo "  docker run --rm --gpus all -v \"\$PWD/HCP_Data:/app/HCP_Data\" $IMAGE"
echo "real run:"
echo "  docker run --rm --gpus all -e SMOKE=0 \\"
echo "    -v \"\$PWD/HCP_Data:/app/HCP_Data\" -v \"\$PWD/output_hcp:/app/output_hcp\" $IMAGE"
