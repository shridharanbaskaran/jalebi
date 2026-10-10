"""Stage checkpoints and resume (0.22, fit.resume).

Every stage of the per-disk pipeline writes its result into the disk's output folder as soon as it is done:

    detection.json          auto-detect: the candidate table, the suggested components, windows and ordering
    grid.json               the parameter vector after the grid stage (+ the best cell per component)
    de_pass1.json, ...      each optimiser pass: theta, chi2, ln P, the resolved parameters (areas), runtime,
                            every start of a multi-start run (0.22, fit.optimise.n_starts)
    continuum_refined.npz   the model-aware refined continuum of pass 1 (continuum_refined2.npz, ...) and
                            whether it was accepted
    chain.h5                the emcee HDF backend (0.16; groups "mcmc" or "mcmc_block{i}"), continued from its
                            last stored step; dynesty writes dynesty.save and restores from it

Each file carries a key: a hash of the data file(s) (names, sizes, modification times -- not the bytes, cubes
are large), the config with the run-only keys removed (output, workers / processes, checkpoint and resume
switches, cache locations, the number of MCMC steps so that a longer run continues the chain) and the model
version (the shared-table model version and LSF version from the emulator modules plus a stage-format version
here).  A file whose key differs, that is partial or that cannot be parsed is ignored and the stage runs again.
Files are written to a temporary name and renamed into place, so a crash never leaves a half-written
checkpoint under the final name.  Log lines are prefixed with "[stage]" so batch monitors can pick them out.

fit.resume: auto (default; continue from the last valid stage) | off (ignore and overwrite checkpoints).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any

import numpy as np

STAGE_FORMAT_VERSION = 1

# config keys that do not change the result and therefore do not take part in the key
RUN_ONLY_KEYS = {
    ("output",),
    ("fit", "resume"),
    ("fit", "mcmc", "processes"), ("fit", "mcmc", "checkpoint"), ("fit", "mcmc", "nsteps"),
    ("fit", "optimise", "workers"),
    ("fit", "grid", "n_jobs"),
    ("fit", "dynesty", "processes"), ("fit", "dynesty", "checkpoint"),
    ("fit", "emulator", "cache_dir"), ("fit", "emulator", "read_only"), ("fit", "emulator", "rebuild"),
    ("fit", "emulator", "spot_check"),
    ("linedata", "data_dir"),
    ("report",),
    ("fit", "shift_null"),          # 0.23: post-processing only
}


def _strip(d: dict, path: tuple = ()) -> dict:
    out = {}
    for k, v in d.items():
        p = path + (k,)
        if p in RUN_ONLY_KEYS:
            continue
        out[k] = _strip(v, p) if isinstance(v, dict) else v
    return out


def data_signature(path: str) -> str:
    """Cheap signature of the data: file names, sizes and mtimes (ns) of a file or of every file in a folder."""
    try:
        from .examples import resolve_path
        path = resolve_path(path)
    except Exception:
        pass
    h = hashlib.sha256()
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                p = os.path.join(root, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                h.update(f"{os.path.relpath(p, path)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    elif os.path.exists(path):
        st = os.stat(path)
        h.update(f"{os.path.basename(path)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    else:
        h.update(f"missing:{path}".encode())
    return h.hexdigest()[:16]


def model_version() -> str:
    from .emulator import LSF_VERSION
    from .emulator_shared import EMULATOR_MODEL_VERSION
    from .model import OpacityBasis
    return f"stage{STAGE_FORMAT_VERSION}-model{EMULATOR_MODEL_VERSION}-lsf{LSF_VERSION}-trunc{OpacityBasis.LINE_TRUNCATE:g}"


def stage_key(cfg, data_path: str | None = None) -> str:
    """The key every checkpoint of this run carries."""
    d = _strip(cfg.model_dump(mode="json"))
    blob = json.dumps(d, sort_keys=True, default=str)
    h = hashlib.sha256()
    h.update(blob.encode()); h.update(model_version().encode())
    h.update(data_signature(data_path if data_path is not None else cfg.target.path).encode())
    return h.hexdigest()[:20]


def atomic_write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=1, default=_json_default)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_npz(path: str, **arrays):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp.npz", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


class Checkpoint:
    """Reads and writes the stage files of one disk folder under one key."""

    def __init__(self, outdir: str | None, key: str, enabled: bool = True, say=None):
        self.outdir = outdir
        self.key = key
        self.enabled = bool(enabled and outdir)
        self.say = say or (lambda m: None)
        self.resumed: list[str] = []
        self.written: list[str] = []

    def path(self, name: str) -> str:
        return os.path.join(self.outdir, name)

    # ---- json stages -------------------------------------------------------------------------
    def load(self, name: str) -> dict | None:
        """The payload of a valid checkpoint, else None (missing, other key, partial or unreadable)."""
        if not self.enabled:
            return None
        p = self.path(name)
        if not os.path.exists(p):
            return None
        try:
            with open(p) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            self.say(f"[stage] {name}: unreadable, ignored")
            return None
        if not isinstance(d, dict) or d.get("key") != self.key or not d.get("complete", False):
            self.say(f"[stage] {name}: " + ("different data/config/model version" if d.get("key") != self.key else "incomplete") + ", ignored")
            return None
        self.resumed.append(name)
        return d.get("payload", {})

    def save(self, name: str, payload: dict):
        if not self.enabled:
            return
        atomic_write_json(self.path(name), {"key": self.key, "stage": name, "complete": True,
                                            "model_version": model_version(), "payload": payload})
        self.written.append(name)

    # ---- array stages (the refined continuum) ------------------------------------------------
    def load_arrays(self, name: str) -> dict | None:
        if not self.enabled:
            return None
        p = self.path(name)
        if not os.path.exists(p):
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                if str(z["key"]) != self.key:
                    self.say(f"[stage] {name}: different data/config/model version, ignored")
                    return None
                out = {k: z[k] for k in z.files if k != "key"}
        except Exception:
            self.say(f"[stage] {name}: unreadable, ignored")
            return None
        self.resumed.append(name)
        return out

    def save_arrays(self, name: str, **arrays):
        if not self.enabled:
            return
        atomic_write_npz(self.path(name), key=np.array(self.key), **arrays)
        self.written.append(name)

    # ---- chain.h5 (emcee) --------------------------------------------------------------------
    def chain_key_path(self) -> str:
        return self.path("chain.key.json")

    def chain_resumable(self, name: str = "chain.h5") -> bool:
        """True when the sampler file (chain.h5 or dynesty.save) exists and the sidecar carries this run's key."""
        if not self.enabled or not os.path.exists(self.path(name)):
            return False
        try:
            with open(self.chain_key_path()) as fh:
                d = json.load(fh)
                return d.get("key") == self.key and d.get("file", "chain.h5") == name
        except (OSError, ValueError):
            return False

    def mark_chain(self, name: str = "chain.h5"):
        if self.enabled:
            atomic_write_json(self.chain_key_path(), {"key": self.key, "file": name, "model_version": model_version()})

    def discard_chain(self, name: str = "chain.h5"):
        for n in (name, "chain.key.json"):
            try:
                os.unlink(self.path(n))
            except OSError:
                pass


def chain_progress(path: str) -> dict[str, int]:
    """{group: stored iterations} of an emcee HDF backend file (empty when the file cannot be read)."""
    out = {}
    try:
        import h5py
        with h5py.File(path, "r") as f:
            for g in f:
                if g == "mcmc" or g.startswith("mcmc_block"):
                    try:
                        out[g] = int(f[g].attrs.get("iteration", 0))
                    except Exception:
                        out[g] = 0
    except Exception:
        pass
    return out


def opt_payload(opt, problem, starts: list | None = None) -> dict:
    """What an optimiser pass stores."""
    P, log_s = problem.params_from_theta(opt.theta)
    return {"theta": np.asarray(opt.theta, float).tolist(), "chi2": float(opt.chi2), "log_prob": float(opt.log_prob),
            "npix": int(opt.npix), "ndim": int(opt.ndim), "runtime_s": float(opt.runtime_s),
            "tau_max": {k: float(v) for k, v in (opt.tau_max or {}).items()},
            "params": problem.model.resolve_params(P), "log_s": float(log_s),
            "free": [p.key for p in problem.free], "starts": starts or []}


def opt_from_payload(d: dict, problem):
    from .fit import OptResult
    th = np.asarray(d["theta"], float)
    if len(th) != problem.ndim or d.get("free") != [p.key for p in problem.free]:
        return None
    o = OptResult(th, float(d["chi2"]), float(d["log_prob"]), int(d["npix"]), int(d["ndim"]), float(d["runtime_s"]),
                  {k: float(v) for k, v in d.get("tau_max", {}).items()})
    o.starts = d.get("starts", [])
    return o


def detection_payload(det) -> dict:
    return {"table": det.table.to_dict(orient="list"),
            "components": [c.model_dump(mode="json") for c in det.components],
            "windows": [list(w) for w in det.windows], "ordering": [list(o) for o in det.ordering],
            "n_pixels": int(det.n_pixels), "chi2_all": float(det.chi2_all), "chi2_red": float(det.chi2_red)}


def detection_from_payload(d: dict):
    import pandas as pd
    from .config import ComponentConfig
    from .detect import DetectionResult
    return DetectionResult(table=pd.DataFrame(d["table"]), components=[ComponentConfig(**c) for c in d["components"]],
                           windows=[tuple(w) for w in d["windows"]], ordering=[list(o) for o in d.get("ordering", [])],
                           n_pixels=int(d.get("n_pixels", 0)), chi2_all=float(d.get("chi2_all", np.nan)),
                           chi2_red=float(d.get("chi2_red", np.nan)))


def status(outdir: str, key: str | None = None) -> dict[str, Any]:
    """Which stage files a folder holds and whether they match `key` (jalebi resume-status)."""
    out = {}
    for name in ("detection.json", "grid.json") + tuple(f"de_pass{i}.json" for i in range(1, 6)):
        p = os.path.join(outdir, name)
        if os.path.exists(p):
            try:
                with open(p) as fh:
                    d = json.load(fh)
                out[name] = "valid" if (key is None or d.get("key") == key) and d.get("complete") else "stale"
            except Exception:
                out[name] = "unreadable"
    for name in ("continuum_refined.npz",) + tuple(f"continuum_refined{i}.npz" for i in range(2, 6)):
        p = os.path.join(outdir, name)
        if os.path.exists(p):
            try:
                with np.load(p, allow_pickle=False) as z:
                    out[name] = "valid" if key is None or str(z["key"]) == key else "stale"
            except Exception:
                out[name] = "unreadable"
    ch = os.path.join(outdir, "chain.h5")
    if os.path.exists(ch):
        prog = chain_progress(ch)
        ok = True
        try:
            with open(os.path.join(outdir, "chain.key.json")) as fh:
                ok = key is None or json.load(fh).get("key") == key
        except Exception:
            ok = False
        out["chain.h5"] = ("valid " if ok else "stale ") + ", ".join(f"{g}: {n} steps" for g, n in prog.items())
    if os.path.exists(os.path.join(outdir, "dynesty.save")):
        out["dynesty.save"] = "present"
    return out
