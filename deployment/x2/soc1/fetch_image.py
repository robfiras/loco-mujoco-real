"""Fetch one platform of a Docker Hub image as a docker-archive tarball, without docker or podman.

usage: python3 fetch_image.py <repo> <tag> <arch> <out.tar>     e.g. library/ros humble-ros-base-jammy arm64 ros.tar
Every blob is checked against its digest, and every layer against the config's diff_id.
"""
import gzip, hashlib, io, json, shutil, sys, tarfile, tempfile, urllib.request
from pathlib import Path

repo, tag, arch, out = sys.argv[1:5]
REG = "https://registry-1.docker.io/v2"
tok = json.load(urllib.request.urlopen(
    f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"))["token"]

def req(url, accept="*/*"):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "Accept": accept}))

idx = json.load(req(f"{REG}/{repo}/manifests/{tag}",
                    "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json"))
digest = next(m["digest"] for m in idx["manifests"]
              if m["platform"]["os"] == "linux" and m["platform"]["architecture"] == arch)
man = json.load(req(f"{REG}/{repo}/manifests/{digest}",
                    "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"))
cfg_raw = req(f"{REG}/{repo}/blobs/{man['config']['digest']}").read()
assert "sha256:" + hashlib.sha256(cfg_raw).hexdigest() == man["config"]["digest"]
diff_ids = json.loads(cfg_raw)["rootfs"]["diff_ids"]

work = Path(tempfile.mkdtemp(dir=Path(out).resolve().parent))
layers = []
for i, (layer, diff_id) in enumerate(zip(man["layers"], diff_ids)):
    gz = work / "blob.gz"
    h = hashlib.sha256()
    with req(f"{REG}/{repo}/blobs/{layer['digest']}") as r, open(gz, "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk); f.write(chunk)
    assert "sha256:" + h.hexdigest() == layer["digest"], f"layer {i} digest mismatch"
    name = diff_id.split(":")[1] + ".tar"
    h2 = hashlib.sha256()
    with gzip.open(gz, "rb") as src, open(work / name, "wb") as dst:
        while chunk := src.read(1 << 20):
            h2.update(chunk); dst.write(chunk)
    assert "sha256:" + h2.hexdigest() == diff_id, f"layer {i} diff_id mismatch"
    gz.unlink()
    layers.append(name)
    print(f"layer {i + 1}/{len(man['layers'])}: {layer['size'] / 1e6:6.1f} MB ok", flush=True)

cfg_name = man["config"]["digest"].split(":")[1] + ".json"
(work / cfg_name).write_bytes(cfg_raw)
(work / "manifest.json").write_text(json.dumps(
    [{"Config": cfg_name, "RepoTags": [f"docker.io/{repo}:{tag}"], "Layers": layers}]))
with tarfile.open(out, "w") as t:
    for p in sorted(work.iterdir()):
        t.add(p, arcname=p.name)
shutil.rmtree(work)
print(f"{out}: {Path(out).stat().st_size / 1e6:.0f} MB, platform linux/{arch}, from {digest}")
