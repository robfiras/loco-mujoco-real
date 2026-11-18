"""Builds a robot bundle whose clips all start and end standing.

::

    python deployment/x2/robot_clips.py windows <source> [--match <regex>] [--kind ...]
    python deployment/x2/robot_clips.py check <source> [<clip>:<start>:<steps> ...]
    python deployment/x2/robot_clips.py build --out <dir> --policy-from <bundle> \\
        <name>=<source>:<clip>[:<start>:<steps>] ...

On the robot, ``deploy_x2.py`` switches clips (n / p, LB / RB), plays them one after another
(``--end next``) and holds the last frame of each. None of that may ask for a pose in
mid-stride, so every clip it steps through starts and ends standing. LAFAN never stands still
in the middle of a walk, so walks from standing to standing come from BONES-SEED.

A source is a bundle directory or its ``motions.lmj.npz`` (any loco_mujoco Trajectory npz with
``clip_names`` in its metadata), a BONES-SEED chunk npz with its csv (``name,first_frame,
frames``) beside it, or a directory of such chunks, whose clips are found by name.

Standing, at a frame, means that for 0.2 s either side the root moves slower than 0.1 m/s, no
joint turns faster than 1.0 rad/s, both ankle roll links move slower than 0.05 m/s and are
within 3 cm of each other in height, the pelvis is above 0.6 m, the tilt is under 12 degrees,
and both shoulder rolls are within 0.6 rad of zero, which LAFAN's opening T-pose is not.

``windows`` lists, calmest first (by top joint speed), windows of three kinds:

=====  =====================================================================================
stand  from the first frame of a standing run to the last frame of the same or a later one,
       5 to 30 s, the root slower than 0.2 m/s throughout
fwd    from the end of a standing run to the start of a later one, each widened by
       ``--margin`` frames into its run, so the clip stands still that long at both ends;
       at least 1 m forward along the starting heading, never more than 0.3 m behind the
       start, turning less than 30 degrees, slower than 1.5 m/s
back   the same, walking backward
=====  =====================================================================================

Every window keeps the pelvis above 0.55 m and the tilt under 15 degrees, and has no yaw rate
above 10 rad/s, which marks a quaternion flip in the data.

``check`` says whether each window's first and last frame are standing and, if not, what
fails. Without windows it checks each clip whole.

``build`` writes ``<out>/motions.lmj.npz``, ``<out>/calm_windows.txt`` and copies
``policy.json`` and ``policy.npz`` from ``--policy-from``. Each clip is named ``<name>``, in
the order given. ``<source>:<clip>`` crops the clip from the end of its first standing run to
the start of its last, widened by ``--margin`` as above; ``<source>:<clip>:<start>:<steps>``
takes that window as it is (steps 0: to the end of the clip), as ``windows`` prints it.
Every source must describe the same model. ``calm_windows.txt``, the playlist of
``deploy_x2.py``, lists the clips that start and end standing; any other stays in the npz,
reachable with ``--motion`` only, and is reported.

A chunk npz is several GB, and loading one of its arrays whole can exhaust a machine's
memory. Its members are stored uncompressed, so they are memory-mapped here and only the
frames of the clips asked for are read, and copied. Scanning many chunks still reads a lot
from disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

MOTIONS_FILE = "motions.lmj.npz"
POLICY_FILES = ("policy.json", "policy.npz")
PER_FRAME = ("qpos", "qvel", "xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat")
MODEL_KEYS = ("joint_names", "body_names", "site_names", "jnt_type", "njnt", "nbody", "nsite",
              "body_rootid", "body_weldid", "body_mocapid", "body_pos", "body_quat", "body_ipos",
              "body_iquat", "site_bodyid", "site_pos", "site_quat", "frequency")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
ARMS = ("left_shoulder_roll_joint", "right_shoulder_roll_joint")

# standing, at a frame: all of these hold for STILL_S either side
STILL_S = 0.2
MAX_ROOT_SPEED = 0.1      # m/s, horizontal
MAX_JOINT_SPEED = 1.0     # rad/s
MAX_FOOT_SPEED = 0.05     # m/s, horizontal
MAX_FOOT_DZ = 0.03        # m between the two feet
MIN_PELVIS_Z = 0.6        # m
MAX_TILT = 12.0           # deg
MAX_SHOULDER_ROLL = 0.6   # rad; LAFAN's T-pose holds about 1.4
# every window
MAX_WINDOW_S = 30.0
MIN_WINDOW_Z = 0.55       # m
MAX_WINDOW_TILT = 15.0    # deg
GLITCH_YAW_RATE = 10.0    # rad/s; faster is a quaternion flip in the data
# stand
MIN_STAND_S = 5.0
MAX_STAND_SPEED = 0.2     # m/s
# fwd / back
MIN_WALK_M = 1.0
MAX_BACKTRACK_M = 0.3
MAX_TURN = 30.0           # deg from the starting heading
MAX_WALK_SPEED = 1.5      # m/s

KINDS = {"stand": "standing throughout", "fwd": "walk forward", "back": "walk backward"}


class Npz:
    """An npz whose uncompressed members are memory-mapped, not read, until sliced."""

    def __init__(self, path: Path):
        self.path = path
        self._zip = zipfile.ZipFile(path)
        self.files = [n[:-4] for n in self._zip.namelist() if n.endswith(".npy")]
        self._arrays: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._arrays:
            self._arrays[key] = self._open(key)
        return self._arrays[key]

    def _open(self, key: str) -> np.ndarray:
        info = self._zip.getinfo(key + ".npy")
        if info.compress_type == zipfile.ZIP_STORED and info.file_size > 1 << 20:
            with open(self.path, "rb") as f:
                f.seek(info.header_offset)
                name_len, extra_len = struct.unpack("<HH", f.read(30)[26:30])
                f.seek(info.header_offset + 30 + name_len + extra_len)
                version = np.lib.format.read_magic(f)
                read_header = {(1, 0): np.lib.format.read_array_header_1_0,
                               (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if read_header is not None:
                    shape, fortran, dtype = read_header(f)
                    if not dtype.hasobject:
                        return np.memmap(self.path, dtype=dtype, mode="r", offset=f.tell(),
                                         shape=shape, order="F" if fortran else "C")
        with self._zip.open(info) as f:
            # allow_pickle: `metadata` is a pickled dict, in our own exports
            return np.lib.format.read_array(f, allow_pickle=True)


class Source:
    """Clips by name, in one Trajectory npz or across a directory of BONES-SEED chunks."""

    def __init__(self, arg: str):
        path = Path(arg).expanduser()
        if path.is_dir() and (path / MOTIONS_FILE).exists():
            path = path / MOTIONS_FILE
        if not path.exists():
            raise SystemExit(f"{path} does not exist")
        self.path = path
        self._npz: dict[Path, Npz] = {}
        self.clips: dict[str, tuple[Path, int, int]] = {}   # name -> (npz, first frame, frames)
        if path.is_dir():
            chunks = sorted(path.glob("chunk_*.csv"))
            if not chunks:
                raise SystemExit(f"{path}: neither a bundle nor a directory of chunk_*.csv")
            for c in chunks:
                self._add_csv(c)
            return
        d = self.npz(path)
        meta = (d["metadata"].item() if "metadata" in d.files else None) or {}
        names = list(meta.get("clip_names") or [])
        split = np.asarray(d["split_points"])
        split = split[split >= 0]
        if len(names) == len(split) - 1:
            for i, n in enumerate(names):
                self.clips[n] = (path, int(split[i]), int(split[i + 1] - split[i]))
        elif path.with_suffix(".csv").exists():
            self._add_csv(path.with_suffix(".csv"))
        else:
            raise SystemExit(f"{path}: no clip_names in its metadata and no csv beside it")

    def _add_csv(self, path: Path) -> None:
        npz = path.with_suffix(".npz")
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["name"] in self.clips:
                    raise SystemExit(f"{r['name']} is in both {self.clips[r['name']][0].name} "
                                     f"and {npz.name}")
                self.clips[r["name"]] = (npz, int(r["first_frame"]), int(r["frames"]))

    def npz(self, path: Path) -> Npz:
        if path not in self._npz:
            self._npz[path] = Npz(path)
        return self._npz[path]

    def locate(self, clip: str) -> tuple[Npz, int, int]:
        if clip not in self.clips:
            raise SystemExit(f"{clip} is not a clip of {self.path}")
        path, lo, n = self.clips[clip]
        return self.npz(path), lo, n

    def read(self, clip: str, keys=PER_FRAME, first: int = 0, last: int | None = None) -> dict:
        """Copies of frames ``first..last`` (both kept) of the clip, never views."""
        d, lo, n = self.locate(clip)
        last = n - 1 if last is None else last
        return {k: np.array(d[k][lo + first:lo + last + 1]) for k in keys}

    def motion(self, clip: str) -> "Motion":
        d, _, _ = self.locate(clip)
        return Motion(self.read(clip, ("qpos", "qvel", "xpos")), d)


class Motion:
    """One clip's standing test, and the measures its windows are judged by."""

    def __init__(self, frames: dict, d: Npz):
        p, v, x = frames["qpos"], frames["qvel"], frames["xpos"]
        self.freq = freq = float(d["frequency"])
        bodies = [str(b) for b in d["body_names"]]
        joints = [str(j) for j in d["joint_names"]]
        free = np.asarray(d["jnt_type"]) == 0  # mjJNT_FREE
        qpos_adr = np.concatenate([[0], np.cumsum(np.where(free, 7, 1))])
        feet = [bodies.index(b) for b in FEET]
        arms = [int(qpos_adr[joints.index(j)]) for j in ARMS]

        self.qpos, self.qvel = p, v
        w, qx, qy, qz = p[:, 3:7].T.astype(np.float64)
        self.tilt = np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1)))
        self.yaw = np.unwrap(np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        self.speed = np.hypot(v[:, 0], v[:, 1])
        self.joint_speed = np.abs(v[:, 6:]).max(axis=1)
        self.glitch = np.abs(v[:, 5]) > GLITCH_YAW_RATE
        foot_speed = np.zeros(len(p))
        step = np.linalg.norm(np.diff(x[:, feet, :2], axis=0), axis=2)
        foot_speed[1:] = step.max(axis=1) * freq
        self.checks = {  # per frame, where each part of standing holds
            "root moving": self.speed < MAX_ROOT_SPEED,
            "joints moving": self.joint_speed < MAX_JOINT_SPEED,
            "feet moving": foot_speed < MAX_FOOT_SPEED,
            "feet not level": np.abs(x[:, feet[0], 2] - x[:, feet[1], 2]) < MAX_FOOT_DZ,
            "pelvis low": p[:, 2] > MIN_PELVIS_Z,
            "tilted": self.tilt < MAX_TILT,
            "arms out (T-pose)": np.abs(p[:, arms]).max(axis=1) < MAX_SHOULDER_ROLL,
        }
        self.half = int(STILL_S * freq)
        self.standing = self._held(np.logical_and.reduce(list(self.checks.values())))
        idx = np.flatnonzero(self.standing)
        starts = idx[np.r_[True, np.diff(idx) > 1]] if len(idx) else idx
        ends = idx[np.r_[np.diff(idx) > 1, True]] if len(idx) else idx
        self.runs = [(int(s), int(e)) for s, e in zip(starts, ends)]

    def __len__(self) -> int:
        return len(self.qpos)

    def _held(self, ok: np.ndarray) -> np.ndarray:
        """Where ``ok`` holds at every frame within ``half`` either side (cut at the ends)."""
        bad = np.concatenate([[0], np.cumsum(~ok)])
        t = np.arange(len(ok))
        return bad[np.minimum(t + self.half + 1, len(ok))] == bad[np.maximum(t - self.half, 0)]

    def why_not(self, t: int) -> str:
        if self.standing[t]:
            return "standing"
        lo, hi = max(0, t - self.half), t + self.half + 1
        failing = [k for k, ok in self.checks.items() if not ok[lo:hi].all()]
        return "NOT standing: " + ", ".join(failing)

    def measure(self, a: int, b: int) -> dict | None:
        """Frames ``a..b``, or None where they fail what every window must keep."""
        seg = slice(a, b + 1)
        if (self.qpos[seg, 2].min() < MIN_WINDOW_Z or self.tilt[seg].max() > MAX_WINDOW_TILT
                or self.glitch[seg].any()):
            return None
        h = self.yaw[a]
        along = (self.qpos[seg, :2] - self.qpos[a, :2]) @ np.array([np.cos(h), np.sin(h)])
        return {"start": a, "steps": b - a + 1, "seconds": (b - a + 1) / self.freq,
                "along": float(along[-1]), "ahead": float(along.max()),
                "behind": float(-along.min()),
                "dyaw": float(np.degrees(self.yaw[b] - h)),
                "turn": float(np.degrees(np.abs(self.yaw[seg] - h).max())),
                "vmax": float(self.speed[seg].max()), "jvmax": float(self.joint_speed[seg].max()),
                "tilt": float(self.tilt[seg].max())}

    def walk(self, i: int, j: int, margin: int) -> tuple[int, int]:
        """From the end of standing run ``i`` to the start of run ``j``, widened by ``margin``."""
        (s0, e0), (s1, e1) = self.runs[i], self.runs[j]
        return max(s0, e0 - margin), min(e1, s1 + margin)

    def windows(self, margin: int) -> list[tuple[str, dict]]:
        out = []
        longest = int(MAX_WINDOW_S * self.freq)
        for i, (s0, e0) in enumerate(self.runs):
            for j in range(i, len(self.runs)):
                s1, e1 = self.runs[j]
                if e1 - s0 >= int(MIN_STAND_S * self.freq) and e1 - s0 <= longest:
                    m = self.measure(s0, e1)
                    if m and m["vmax"] < MAX_STAND_SPEED:
                        out.append(("stand", m))
                if j == i:
                    continue
                a, b = self.walk(i, j, margin)
                if b - a > longest:
                    break
                m = self.measure(a, b)
                if not m or m["turn"] > MAX_TURN or m["vmax"] > MAX_WALK_SPEED:
                    continue
                if m["along"] >= MIN_WALK_M and m["behind"] <= MAX_BACKTRACK_M:
                    out.append(("fwd", m))
                elif -m["along"] >= MIN_WALK_M and m["ahead"] <= MAX_BACKTRACK_M:
                    out.append(("back", m))
        return out


def cmd_windows(args) -> None:
    src = Source(args.source)
    pattern = re.compile(args.match) if args.match else None
    names = [n for n in src.clips if not pattern or pattern.search(n)]
    print(f"{len(names)} clips of {src.path}")
    found = {k: [] for k in args.kind}
    for name in names:
        m = src.motion(name)
        if args.verbose:
            print(f"  {name}  {len(m) / m.freq:.1f} s, standing runs "
                  + (" ".join(f"{s}..{e}" for s, e in m.runs) or "none"))
        for kind, w in m.windows(args.margin):
            if kind in found:
                found[kind].append((name, w))
    print("\n<clip>:<start>:<steps>, length, distance along the starting heading, final heading "
          "change, top root speed, top joint speed, top tilt")
    for kind, rows in found.items():
        print(f"\n== {KINDS[kind]}")
        taken: dict[str, list[tuple[int, int]]] = {}
        n = 0
        for name, w in sorted(rows, key=lambda r: (r[1]["jvmax"], r[1]["steps"])):
            a, b = w["start"], w["start"] + w["steps"]
            if any(a < b2 and a2 < b for a2, b2 in taken.get(name, [])):
                continue
            taken.setdefault(name, []).append((a, b))
            print(f"  {name}:{w['start']}:{w['steps']}  {w['seconds']:5.1f} s  "
                  f"{w['along']:+5.2f} m  {w['dyaw']:+4.0f} deg  {w['vmax']:4.2f} m/s  "
                  f"{w['jvmax']:5.2f} rad/s  {w['tilt']:4.1f} deg")
            n += 1
            if n >= args.top:
                break


def parse_window(text: str) -> tuple[str, int, int]:
    clip, start, steps = text.rsplit(":", 2)
    return clip, int(start), int(steps)


def cmd_check(args) -> None:
    src = Source(args.source)
    windows = [parse_window(w) for w in args.windows] or [(n, 0, 0) for n in src.clips]
    motions: dict[str, Motion] = {}
    bad = 0
    for clip, start, steps in windows:
        if clip not in motions:
            motions[clip] = src.motion(clip)
        m = motions[clip]
        if not 0 <= start < len(m):
            raise SystemExit(f"{clip} has {len(m)} frames, none at {start}")
        last = min(start + steps, len(m)) - 1 if steps else len(m) - 1
        first_says, last_says = m.why_not(start), m.why_not(last)
        bad += not (m.standing[start] and m.standing[last])
        print(f"{clip}:{start}:{steps}\n  start ({start}) {first_says}\n"
              f"  end   ({last}) {last_says}")
    print(f"{len(windows) - bad} of {len(windows)} start and end standing")


def parse_spec(text: str) -> tuple[str, str, str, int | None, int | None]:
    """``<name>=<source>:<clip>[:<start>:<steps>]``; a clip name may not hold a colon."""
    name, rest = text.split("=", 1)
    parts = rest.split(":")
    if len(parts) >= 4 and parts[-1].isdigit() and parts[-2].isdigit():
        return name, ":".join(parts[:-3]), parts[-3], int(parts[-2]), int(parts[-1])
    return name, ":".join(parts[:-1]), parts[-1], None, None


def same_model(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind == "f":  # float noise of ~1e-8 between exports is the same model
        return a.shape == b.shape and np.allclose(a, b, atol=1e-6)
    return np.array_equal(a, b)


def cmd_build(args) -> None:
    out = Path(args.out).expanduser()
    if (out / MOTIONS_FILE).exists() and not args.force:
        raise SystemExit(f"{out / MOTIONS_FILE} exists; --force replaces it")
    policy_dir = Path(args.policy_from).expanduser()
    policy = json.loads((policy_dir / "policy.json").read_text())

    sources: dict[str, Source] = {}
    model, model_from = None, None
    pieces = {k: [] for k in PER_FRAME}
    names, origins = [], []
    for spec in args.clips:
        name, src_arg, clip, start, steps = parse_spec(spec)
        if name in names:
            raise SystemExit(f"two clips are named {name}")
        if src_arg not in sources:
            sources[src_arg] = Source(src_arg)
        src = sources[src_arg]
        d, _, n = src.locate(clip)
        if model is None:
            model = {k: np.asarray(d[k]) for k in MODEL_KEYS if k in d.files}
            model_from = d.path
        for k, v in model.items():
            if k not in d.files or not same_model(v, np.asarray(d[k])):
                raise SystemExit(f"{d.path} describes another model than {model_from}: {k}")
        if start is None:
            m = src.motion(clip)
            if len(m.runs) < 2:
                stands = (f"stands only at {m.runs[0][0]}..{m.runs[0][1]}" if m.runs
                          else "never stands")
                raise SystemExit(f"{clip} {stands}; give it a window, <clip>:<start>:<steps>")
            first, last = m.walk(0, len(m.runs) - 1, args.margin)
        else:
            first = max(0, start)
            last = min(start + steps, n) - 1 if steps else n - 1
        if first >= last:
            raise SystemExit(f"{clip}: frames {first}..{last} leave nothing of its {n}")
        frames = src.read(clip, PER_FRAME, first, last)
        for k in PER_FRAME:
            pieces[k].append(frames[k])
        names.append(name)
        origins.append(f"{d.path.name}:{clip}:{first}..{last}")
        print(f"{name:16s} <- {origins[-1]}  ({last - first + 1} frames)")

    joints = [str(j) for j in model["joint_names"]]
    bodies = [str(b) for b in model["body_names"]]
    missing = ([j for j in policy["joint_names"] if j not in joints]
               + [b for b in policy["tracked_body_names"] if b not in bodies])
    if missing:
        raise SystemExit(f"the clips lack what {policy_dir / 'policy.json'} reads: {missing}")

    lengths = [len(p) for p in pieces["qpos"]]
    arrays = {k: np.concatenate(v) for k, v in pieces.items()}
    arrays["split_points"] = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    arrays["metadata"] = np.array({"clip_names": names, "clip_sources": origins,
                                   "exported_by": Path(__file__).name}, dtype=object)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / MOTIONS_FILE, **arrays, **model)
    print(f"{len(names)} clips, {sum(lengths)} frames -> {out / MOTIONS_FILE}")

    built = Source(str(out / MOTIONS_FILE))
    calm = []
    for name in names:
        m = built.motion(name)
        if m.standing[0] and m.standing[-1]:
            calm.append(f"{name}:0:0")
        else:
            print(f"  {name} left out of calm_windows.txt: start {m.why_not(0)}, "
                  f"end {m.why_not(len(m) - 1)}")
    (out / "calm_windows.txt").write_text("".join(w + "\n" for w in calm))
    print(f"{len(calm)} clips start and end standing -> {out / 'calm_windows.txt'}")
    for f in POLICY_FILES:
        if (policy_dir / f).resolve() != (out / f).resolve():
            shutil.copy2(policy_dir / f, out / f)
    print(f"{', '.join(POLICY_FILES)} <- {policy_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    margin = argparse.ArgumentParser(add_help=False)
    margin.add_argument("--margin", type=int, default=10,
                        help="frames of standing kept at each end of a walk (default 10)")

    w = sub.add_parser("windows", parents=[margin],
                       help="list windows that start and end standing, by kind")
    w.add_argument("source")
    w.add_argument("--match", help="only clips whose name this regex finds")
    w.add_argument("--kind", nargs="+", choices=list(KINDS), default=list(KINDS))
    w.add_argument("--top", type=int, default=20, help="windows per kind (default 20)")
    w.add_argument("-v", "--verbose", action="store_true", help="print each clip's standing runs")
    w.set_defaults(run=cmd_windows)

    c = sub.add_parser("check", help="do windows start and end standing?")
    c.add_argument("source")
    c.add_argument("windows", nargs="*", metavar="clip:start:steps")
    c.set_defaults(run=cmd_check)

    b = sub.add_parser("build", parents=[margin], help="write a bundle from cropped clips")
    b.add_argument("--out", required=True, help="bundle directory to write")
    b.add_argument("--policy-from", required=True,
                   help="bundle whose policy.json and policy.npz to copy")
    b.add_argument("--force", action="store_true", help="replace an existing motions.lmj.npz")
    b.add_argument("clips", nargs="+", metavar="name=source:clip[:start:steps]")
    b.set_defaults(run=cmd_build)

    args = ap.parse_args()
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())
