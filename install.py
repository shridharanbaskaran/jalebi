#!/usr/bin/env python3
"""
JALEBI interactive installer — makes a fresh jalebi in seven steps.

    python install.py              # interactive: asks where and what to install
    python install.py --yes        # accept every default (no questions)
    python install.py --check      # only report what is missing; change nothing
    python install.py --help       # all options

It uses only the Python standard library, so it runs before anything else is installed.  Every
command it runs is printed, and pip's full output goes to install_log.txt.

Steps
  1. Checking the kitchen      Python, pip, conda, disk space, internet
  2. Choosing the pan          a new conda env, a new virtual env, or this Python
  3. Picking the toppings      optional extras: web app, HITRAN downloads, notebooks, ...
  4. Counting the ingredients  which requirements are present, missing or too old
  5. Frying                    pip installs only what is missing
  6. Soaking in syrup          `jalebi doctor`: packages, line lists, example data, speed
  7. Taste test                an optional one-minute fit of a synthetic disk
"""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src", "jalebi")
LOG = os.path.join(HERE, "install_log.txt")
IS_WIN = os.name == "nt"


# ------------------------------------------------------------------------------------------------
# Shared pieces from the package (loaded by path: nothing is installed yet)
# ------------------------------------------------------------------------------------------------

def _load(name):
    path = os.path.join(SRC, name + ".py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("jalebi_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if sys.version_info < (3, 6):
    sys.exit("JALEBI needs Python 3.10 or newer (this is %s)." % platform.python_version())

REQ = _load("_requirements")
BAN = _load("_banner")
if REQ is None:
    sys.exit("install.py must be run from the JALEBI source folder (the one with pyproject.toml and src/jalebi/).")
PY_MIN = REQ.PYTHON_MIN


def _version():
    try:
        for line in open(os.path.join(SRC, "__init__.py")):
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip("\"'")
    except OSError:
        pass
    return "?"


VERSION = _version()


# ------------------------------------------------------------------------------------------------
# Terminal helpers
# ------------------------------------------------------------------------------------------------

class UI:
    colour = True
    assume_yes = False
    interactive = True

    @classmethod
    def c(cls, code, text, bold=False):
        if not cls.colour:
            return text
        return "\033[%s38;5;%dm%s\033[0m" % ("1;" if bold else "", code, text)


def amber(t, bold=False):
    return UI.c(214, t, bold)


def green(t):
    return UI.c(78, t)


def red(t):
    return UI.c(203, t)


def grey(t):
    return UI.c(245, t)


def cyan(t):
    return UI.c(81, t)


def yellow(t):
    return UI.c(221, t)


OKM, BADM, WARNM, DOTM = "✔", "✘", "!", "·"


def say(text=""):
    print(text, flush=True)


def step(n, total, title, blurb=""):
    bar = amber("━" * 4)
    say()
    say("%s %s %s" % (bar, amber("Step %d/%d  %s" % (n, total, title), bold=True), grey(blurb)))


def ok(text):
    say("  %s %s" % (green(OKM), text))


def bad(text):
    say("  %s %s" % (red(BADM), text))


def warn(text):
    say("  %s %s" % (yellow(WARNM), text))


def info(text):
    say("  %s %s" % (grey(DOTM), text))


def ask_yes(question, default=True):
    if UI.assume_yes or not UI.interactive:
        info("%s %s" % (question, grey("-> %s" % ("yes" if default else "no"))))
        return default
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            a = input("  %s %s %s " % (cyan("?"), question, grey(hint))).strip().lower()
        except EOFError:
            return default
        if not a:
            return default
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False


def ask_choice(question, options, default=0):
    """options: list of (label, description). Returns the index."""
    say("  %s %s" % (cyan("?"), question))
    for i, (label, desc) in enumerate(options):
        mark = amber("❯") if i == default else " "
        say("    %s %s %s  %s" % (mark, amber("%d)" % (i + 1)), label, grey(desc)))
    if UI.assume_yes or not UI.interactive:
        info("-> %s" % options[default][0])
        return default
    while True:
        try:
            a = input("    %s " % grey("choose 1-%d [%d]:" % (len(options), default + 1))).strip()
        except EOFError:
            return default
        if not a:
            return default
        if a.isdigit() and 1 <= int(a) <= len(options):
            return int(a) - 1


def ask_text(question, default):
    if UI.assume_yes or not UI.interactive:
        return default
    try:
        a = input("  %s %s %s " % (cyan("?"), question, grey("[%s]" % default))).strip()
    except EOFError:
        return default
    return a or default


class Spinner:
    """A spiralling spinner that shows a fun fact every few seconds while a command runs."""
    FRAMES = ["◜", "◠", "◝", "◞", "◡", "◟"]

    def __init__(self, label):
        self.label = label
        self.stop_ev = threading.Event()
        self.t0 = time.time()
        facts = list(getattr(BAN, "FUN_FACTS", [])) or ["Frying..."]
        import random
        random.shuffle(facts)
        self.facts = facts
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_ev.set()
        self.thread.join()
        if sys.stdout.isatty():
            sys.stdout.write("\r" + " " * (shutil.get_terminal_size((100, 20)).columns - 1) + "\r")
            sys.stdout.flush()

    def _run(self):
        tty = sys.stdout.isatty()
        i = 0
        last_dot = 0
        while not self.stop_ev.wait(0.12):
            el = time.time() - self.t0
            if tty:
                width = shutil.get_terminal_size((100, 20)).columns - 1
                fact = self.facts[int(el // 7) % len(self.facts)]
                head = "  %s %s %s  " % (amber(self.FRAMES[i % len(self.FRAMES)]), self.label, grey("%3.0fs" % el))
                plain_len = len(self.label) + 12
                room = max(width - plain_len, 0)
                tail = fact if len(fact) <= room else fact[:max(room - 1, 0)] + "…"
                sys.stdout.write("\r" + head + grey(tail) + " " * max(room - len(tail), 0))
                sys.stdout.flush()
            elif el - last_dot > 15:
                last_dot = el
                say("  ... %s (%.0f s)" % (self.label, el))
            i += 1


def run(cmd, label, log=True, cwd=None, env=None, show=True):
    """Run a command with a spinner; output goes to the log. Returns (returncode, seconds)."""
    if show:
        info(grey("$ " + " ".join(_q(c) for c in cmd)))
    if RUNTIME.dry_run:
        return 0, 0.0
    t0 = time.time()
    with open(LOG, "a") as fh:
        fh.write("\n\n$ %s\n" % " ".join(cmd))
        fh.flush()
        with Spinner(label):
            p = subprocess.Popen(cmd, stdout=fh if log else None, stderr=subprocess.STDOUT if log else None,
                                 cwd=cwd or HERE, env=env)
            rc = p.wait()
    return rc, time.time() - t0


def _q(s):
    return '"%s"' % s if (" " in s or "[" in s) else s


def tail_log(n=25):
    try:
        lines = open(LOG).read().splitlines()[-n:]
    except OSError:
        return
    say(grey("  ─── last lines of %s ───" % os.path.basename(LOG)))
    for ln in lines:
        say(grey("  │ ") + ln)


class RUNTIME:
    dry_run = False


# ------------------------------------------------------------------------------------------------
# Step 1: the kitchen
# ------------------------------------------------------------------------------------------------

def find_conda():
    for exe in (os.environ.get("CONDA_EXE"), shutil.which("mamba"), shutil.which("conda")):
        if exe and os.path.exists(exe):
            return exe
    home = os.path.expanduser("~")
    for base in ("miniforge3", "mambaforge", "miniconda3", "anaconda3", "micromamba"):
        for sub in (("bin", "conda"), ("condabin", "conda"), ("Scripts", "conda.exe")):
            p = os.path.join(home, base, *sub)
            if os.path.exists(p):
                return p
    return None


def in_venv():
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def in_conda():
    return bool(os.environ.get("CONDA_PREFIX")) and os.path.abspath(sys.prefix) == os.path.abspath(os.environ["CONDA_PREFIX"])


def externally_managed(python=None):
    """PEP 668: pip refuses to touch a system Python that carries an EXTERNALLY-MANAGED marker."""
    code = ("import sysconfig,os,sys;p=os.path.join(sysconfig.get_path('stdlib'),'EXTERNALLY-MANAGED');"
            "print(int(os.path.exists(p) and sys.prefix==getattr(sys,'base_prefix',sys.prefix)))")
    try:
        out = subprocess.run([python or sys.executable, "-c", code], capture_output=True, text=True, timeout=30).stdout
        return out.strip() == "1"
    except Exception:
        return False


def internet(host="pypi.org", port=443, timeout=3.0):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def mem_gb():
    try:
        if sys.platform.startswith("linux"):
            for ln in open("/proc/meminfo"):
                if ln.startswith("MemTotal"):
                    return int(ln.split()[1]) / 1024 ** 2
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
            return int(out) / 1024 ** 3
    except Exception:
        pass
    return None


def check_kitchen():
    facts = {}
    v = sys.version_info
    facts["python_ok"] = v[:2] >= PY_MIN
    (ok if facts["python_ok"] else warn)("Python %s at %s%s" % (platform.python_version(), sys.executable,
                                         "" if facts["python_ok"] else "  (JALEBI needs >= %d.%d; a conda env can bring its own)" % PY_MIN))
    ok("%s %s, %d CPU cores%s" % (platform.system(), platform.machine(), os.cpu_count() or 1,
                                  (", %.0f GB RAM" % mem_gb()) if mem_gb() else ""))
    try:
        import pip  # noqa: F401
        facts["pip"] = True
        ok("pip %s" % __import__("pip").__version__)
    except ImportError:
        facts["pip"] = False
        warn("pip is not available for this Python (a new environment will bring its own)")
    facts["conda"] = find_conda()
    (ok if facts["conda"] else info)("conda: %s" % (facts["conda"] or "not found (fine: a virtual environment works too)"))
    facts["venv"] = in_venv()
    facts["in_conda"] = in_conda()
    if facts["venv"] or facts["in_conda"]:
        ok("you are inside an environment: %s" % sys.prefix)
    facts["externally_managed"] = externally_managed()
    if facts["externally_managed"]:
        info("this Python is managed by the operating system (PEP 668): JALEBI goes into its own environment")
    free = shutil.disk_usage(HERE).free / 1024 ** 3
    facts["disk_gb"] = free
    (ok if free > 2 else warn)("%.1f GB free disk here (a full install needs about 1 GB)" % free)
    facts["internet"] = internet()
    (ok if facts["internet"] else warn)("internet: %s" % ("pypi.org reachable" if facts["internet"] else
                                                         "pypi.org NOT reachable (installing needs it, or a local wheel cache)"))
    facts["repo"] = os.path.exists(os.path.join(HERE, "pyproject.toml"))
    ok("JALEBI %s source at %s" % (VERSION, HERE))
    return facts


# ------------------------------------------------------------------------------------------------
# Step 2: the pan (environment)
# ------------------------------------------------------------------------------------------------

def conda_env_prefix(conda, name):
    try:
        out = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True, timeout=60).stdout
        for p in json.loads(out).get("envs", []):
            if os.path.basename(p.rstrip("/\\")) == name:
                return p
    except Exception:
        pass
    return None


def env_python(prefix):
    return os.path.join(prefix, "python.exe") if IS_WIN else os.path.join(prefix, "bin", "python")


def choose_env(args, facts):
    """Returns dict(kind, python, create_cmd, name/path, activate)."""
    opts, kinds = [], []
    if facts["conda"]:
        opts.append(("new conda environment", "Python 3.12 from conda-forge, isolated; recommended with conda"))
        kinds.append("conda")
    opts.append(("new virtual environment", "./.venv-jalebi using this Python %d.%d" % sys.version_info[:2]))
    kinds.append("venv")
    here_ok = facts["python_ok"] and not facts["externally_managed"]
    if here_ok:
        where = "the active environment" if (facts["venv"] or facts["in_conda"]) else "your user/system Python"
        opts.append(("this Python", "install into %s (%s)" % (where, sys.executable)))
        kinds.append("current")
    if args.env != "auto":
        if args.env not in kinds:
            bad("--env %s is not possible here (options: %s)" % (args.env, ", ".join(kinds)))
            sys.exit(2)
        kind = args.env
        info("environment: %s" % kind)
    else:
        default = 0
        if (facts["venv"] or facts["in_conda"]) and "current" in kinds:
            default = kinds.index("current")
        if not facts["python_ok"] and "conda" in kinds:
            default = kinds.index("conda")
        kind = kinds[ask_choice("Where should JALEBI live?", opts, default)]

    env = {"kind": kind}
    if kind == "conda":
        name = args.name or ask_text("name of the conda environment", "jalebi")
        prefix = conda_env_prefix(facts["conda"], name)
        env.update(name=name, activate="conda activate %s" % name)
        if prefix and os.path.exists(env_python(prefix)):
            ok("conda env '%s' exists (%s): it will be updated, not recreated" % (name, prefix))
            env.update(python=env_python(prefix), create=None)
        else:
            env.update(python=None, create=[facts["conda"], "create", "-y", "-n", name, "--override-channels", "-c", "conda-forge",
                                            "python=%s" % args.python, "pip"])
    elif kind == "venv":
        if not facts["python_ok"]:
            bad("a virtual environment copies this Python %s, which is too old; use conda or a newer Python" % platform.python_version())
            sys.exit(2)
        path = os.path.abspath(args.venv or ask_text("folder for the virtual environment", ".venv-jalebi"))
        py = os.path.join(path, "Scripts", "python.exe") if IS_WIN else os.path.join(path, "bin", "python")
        rel = os.path.relpath(path, os.getcwd())
        shown = path if rel.startswith("..") else rel
        act = (os.path.join(shown, "Scripts", "activate") if IS_WIN else "source %s" % os.path.join(shown, "bin", "activate"))
        env.update(path=path, activate=act, python=py if os.path.exists(py) else None,
                   create=None if os.path.exists(py) else [sys.executable, "-m", "venv", path])
        if os.path.exists(py):
            ok("virtual environment %s exists: it will be updated" % path)
    else:
        env.update(python=sys.executable, create=None, activate=None)
    return env


# ------------------------------------------------------------------------------------------------
# Step 3: toppings
# ------------------------------------------------------------------------------------------------

def choose_extras(args):
    if args.extras is not None:
        chosen = [e.strip() for e in args.extras.split(",") if e.strip()]
        unknown = [e for e in chosen if e not in REQ.OPTIONAL]
        if unknown:
            bad("unknown extras: %s (known: %s)" % (", ".join(unknown), ", ".join(REQ.OPTIONAL)))
            sys.exit(2)
        info("extras: %s" % (", ".join(chosen) or "none"))
        return chosen
    info("the core (model, continuum, fitting, MCMC, CLI, bundled line lists and data) is always installed")
    chosen = []
    for name, default, desc in REQ.EXTRAS_MENU:
        pk = ", ".join(REQ.OPTIONAL[name])
        if ask_yes("%s %s  %s" % (amber("[%s]" % name), desc, grey("(%s)" % pk)), default):
            chosen.append(name)
    return chosen


# ------------------------------------------------------------------------------------------------
# Step 4: ingredients
# ------------------------------------------------------------------------------------------------

PROBE = r"""
import json, sys
try:
    import importlib.metadata as md
except ImportError:
    import importlib_metadata as md
names = json.loads(sys.argv[1])
out = {}
for n in names:
    try:
        out[n] = md.version(n)
    except Exception:
        out[n] = None
print(json.dumps({"python": "%d.%d.%d" % sys.version_info[:3], "versions": out}))
"""


def probe(python, names):
    try:
        r = subprocess.run([python, "-c", PROBE, json.dumps(names)], capture_output=True, text=True, timeout=120)
        return json.loads(r.stdout)
    except Exception:
        return None


def count_ingredients(env, extras, optional_soft=False):
    """Status of the core + chosen extras in the target environment.  With optional_soft, missing
    optional packages are shown as '·' (not needed) instead of '✘'."""
    wanted = dict(REQ.CORE)
    soft = set()
    for e in extras:
        wanted.update(REQ.OPTIONAL[e])
        if optional_soft:
            soft |= set(REQ.OPTIONAL[e])
    names = list(wanted) + ["jalebi"]
    if env.get("python") is None:
        info("a fresh pan: all %d ingredients will be added" % len(wanted))
        return {"missing": list(wanted), "old": [], "ok": []}
    res = probe(env["python"], names)
    if res is None:
        warn("could not inspect %s; pip will sort it out" % env["python"])
        return {"missing": list(wanted), "old": [], "ok": []}
    vers = res["versions"]
    rows = {"missing": [], "old": [], "ok": []}
    for dist, (vmin, _mod, why) in wanted.items():
        v = vers.get(dist)
        if v is None and dist in soft:
            say("  %s %-14s %-10s %s" % (grey(DOTM), dist, grey("optional"), grey(why)))
        elif v is None:
            rows["missing"].append(dist)
            say("  %s %-14s %-10s %s" % (red(BADM), dist, grey("missing"), grey(why)))
        elif not REQ.version_ok(v, vmin):
            rows["old"].append(dist)
            say("  %s %-14s %-10s %s" % (yellow("↑"), dist, v, grey("needs >= %s" % vmin)))
        else:
            rows["ok"].append(dist)
            say("  %s %-14s %-10s %s" % (green(OKM), dist, v, grey(why)))
    if vers.get("jalebi"):
        info("jalebi %s is already installed here; it will be replaced by %s" % (vers["jalebi"], VERSION))
    say()
    (ok if not (rows["missing"] or rows["old"]) else info)(
        "%d ready, %d missing, %d too old  (Python %s)%s" % (len(rows["ok"]), len(rows["missing"]), len(rows["old"]), res["python"],
                                                           "; pip will add only these" if (rows["missing"] or rows["old"]) else ""))
    return rows


# ------------------------------------------------------------------------------------------------
# Steps 5-7
# ------------------------------------------------------------------------------------------------

def fry(env, extras, args):
    if env.get("create"):
        rc, dt = run(env["create"], "preparing the pan (%s environment)" % env["kind"])
        if rc != 0:
            bad("could not create the environment (see %s)" % LOG); tail_log(); sys.exit(1)
        ok("environment created in %.0f s" % dt)
        if env["kind"] == "conda" and not RUNTIME.dry_run:
            prefix = conda_env_prefix(env_conda(), env["name"])
            env["python"] = env_python(prefix) if prefix else None
            if not env["python"] or not os.path.exists(env["python"]):
                bad("created the conda env but cannot find its python"); sys.exit(1)
        elif env["kind"] == "venv":
            env["python"] = os.path.join(env["path"], "Scripts", "python.exe") if IS_WIN else os.path.join(env["path"], "bin", "python")
    py = env["python"] or sys.executable
    rc, _ = run([py, "-m", "pip", "install", "--upgrade", "pip"], "sharpening the knives (upgrading pip)")
    if rc != 0:
        warn("pip upgrade failed; continuing with the installed pip")
    target = HERE + ("[%s]" % ",".join(extras) if extras else "")
    cmd = [py, "-m", "pip", "install"]
    if args.user:
        cmd.append("--user")
    cmd += (["-e", target] if args.editable else [target])
    rc, dt = run(cmd, "frying the jalebi (pip install%s)" % (" -e" if args.editable else ""))
    if rc != 0:
        bad("pip install failed after %.0f s" % dt)
        tail_log()
        say()
        info("common fixes: check the internet connection / proxy; upgrade pip; for an 'externally-managed' error use a")
        info("virtual or conda environment (python install.py --env venv); full log: %s" % LOG)
        sys.exit(1)
    ok("installed in %.0f s%s" % (dt, " (editable: edits in src/jalebi take effect immediately)" if args.editable else ""))
    if "notebook" in extras:
        rc, _ = run([py, "-m", "ipykernel", "install", "--user", "--name", "jalebi", "--display-name", "Python (JALEBI)"],
                    "registering the Jupyter kernel")
        (ok if rc == 0 else warn)("Jupyter kernel 'Python (JALEBI)' %s" % ("registered" if rc == 0 else "could not be registered"))
    return py


def env_conda():
    return find_conda()


def soak(py):
    if RUNTIME.dry_run:
        info("(dry run: skipping the checks)"); return True
    with Spinner("letting it soak (jalebi doctor)"):
        r = subprocess.run([py, "-m", "jalebi.doctor", "--json"], capture_output=True, text=True, cwd=os.path.expanduser("~"))
    try:
        rep = json.loads(r.stdout)
    except ValueError:
        bad("jalebi doctor did not run:"); say(r.stdout[-2000:] + r.stderr[-2000:]); return False
    core = rep["packages"]["core"]
    n_ok = sum(1 for x in core if x["status"] == "ok")
    (ok if n_ok == len(core) else bad)("%d/%d required packages import cleanly" % (n_ok, len(core)))
    for x in rep.get("imports", []):
        if x["status"] != "ok":
            bad("import %s: %s" % (x["module"], x.get("error")))
    for g, rows in rep["packages"]["optional"].items():
        if all(x["status"] == "ok" for x in rows):
            ok("extra [%s]: %s" % (g, ", ".join("%s %s" % (x["name"], x["installed"]) for x in rows)))
    L = rep.get("linelists", {})
    if L.get("status") == "ok":
        ok("%d line lists (%.0f MB) ready: %s" % (L["n_lists"], L["MB"], ", ".join(sorted({s.split(":")[0] for s in L["lists"]}))))
        for t in L.get("tests", []):
            info("%s: %d lines, Z(500 K) = %s" % (t["list"], t["lines"], t["Z(500K)"]))
    elif L:
        bad("line lists: %s" % (L.get("error") or [t for t in L.get("tests", []) if t["status"] != "ok"]))
    E = rep.get("examples", {})
    if E:
        (ok if E.get("status") == "ok" else warn)("example data: FZ Tau (%s/12 MIRI x1d files), synthetic spectrum %s"
                                                  % (E.get("FZ_Tau_x1d_files"), "yes" if E.get("synthetic") else "no"))
    B = rep.get("benchmark", {})
    if "ms_per_model" in B:
        ok("speed: %.1f ms per model evaluation on %d cores" % (B["ms_per_model"], rep.get("parallel", {}).get("cpu_count", 1)))
    if not rep.get("ready"):
        bad("not ready: %s" % (rep.get("fix") or "see `jalebi doctor`"))
    return bool(rep.get("ready"))


def taste(py, args):
    if args.no_demo:
        info("skipped (--no-demo); later: jalebi demo")
        return
    if not ask_yes("Taste test: fit the bundled synthetic disk now (about a minute, no MCMC)?", default=not args.yes):
        info("later: jalebi demo")
        return
    if RUNTIME.dry_run:
        return
    out = os.path.join(os.getcwd(), "jalebi_demo")
    say(grey("  $ jalebi demo --no-mcmc --out %s" % out))
    say()
    env = dict(os.environ)
    env.setdefault("FORCE_COLOR", "1" if UI.colour else "0")
    rc = subprocess.call([py, "-m", "jalebi", "demo", "--no-mcmc", "--out", out], env=env)
    (ok if rc == 0 else bad)("taste test %s" % ("passed: figures in %s" % out if rc == 0 else "failed (rc=%d)" % rc))


def serve_card(env, extras, ready):
    say()
    line = amber("─" * 74)
    say(line)
    if RUNTIME.dry_run:
        say("  " + yellow("Dry run: nothing was installed. These are the next steps after a real install."))
    else:
        say("  " + amber("Your jalebi is ready!", bold=True) if ready else "  " + yellow("Installed, with warnings (see above)."))
    say(line)
    if env.get("activate"):
        say("  1. %-30s %s" % ("switch to its environment", cyan(env["activate"])))
    else:
        say("  1. %-30s %s" % ("it is installed in", cyan(sys.executable)))
    rows = [("check the installation", "jalebi doctor"),
            ("fit a synthetic disk (+ MCMC)", "jalebi demo"),
            ("copy the examples", "jalebi examples ./my_examples"),
            ("start from a config", "jalebi init my_disk.yaml --example fz_tau"),
            ("run a fit", "jalebi fit my_disk.yaml"),
            ("all commands", "jalebi --help")]
    if "app" in extras:
        rows.insert(2, ("open the web app", "jalebi serve --show"))
    for i, (what, cmd) in enumerate(rows, start=2):
        say("  %d. %-30s %s" % (i, what, cyan(cmd)))
    say()
    say(grey("  README.md explains the physics, the fitting and every option.  Install log: %s" % LOG))
    say()


# ------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="JALEBI interactive installer",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=textwrap.dedent("""
                                 examples:
                                   python install.py                         interactive
                                   python install.py --yes                   defaults, no questions
                                   python install.py --env conda --name jalebi --extras app,fetch,notebook
                                   python install.py --env venv --venv .venv
                                   python install.py --env current --no-editable
                                   python install.py --check                 report only
                                 """))
    ap.add_argument("--yes", "-y", action="store_true", help="accept all defaults (non-interactive)")
    ap.add_argument("--check", action="store_true", help="only check what is installed/missing; change nothing")
    ap.add_argument("--env", choices=["auto", "conda", "venv", "current"], default="auto", help="where to install")
    ap.add_argument("--name", default=None, help="conda environment name (default jalebi)")
    ap.add_argument("--python", default="3.12", help="Python version for a new conda env (default 3.12)")
    ap.add_argument("--venv", default=None, help="folder for a new virtual environment (default .venv-jalebi)")
    ap.add_argument("--extras", default=None, help="comma-separated extras: " + ",".join(REQ.OPTIONAL) + " (default: ask)")
    ap.add_argument("--editable", dest="editable", action="store_true", default=True,
                    help="editable install: your edits in src/jalebi take effect immediately (default)")
    ap.add_argument("--no-editable", dest="editable", action="store_false", help="regular (non-editable) install")
    ap.add_argument("--user", action="store_true", help="pip install --user (only with --env current)")
    ap.add_argument("--no-demo", action="store_true", help="skip the taste test")
    ap.add_argument("--no-color", action="store_true", help="plain output")
    ap.add_argument("--dry-run", action="store_true", help="show the commands without running them")
    args = ap.parse_args(argv)

    UI.colour = not args.no_color and BAN is not None and BAN.supports_colour()
    UI.assume_yes = args.yes
    UI.interactive = sys.stdin.isatty()
    RUNTIME.dry_run = args.dry_run
    if not UI.interactive and not args.yes and not args.check:
        UI.assume_yes = True
    open(LOG, "w").write("JALEBI %s installer log, %s\n" % (VERSION, time.strftime("%Y-%m-%d %H:%M:%S")))

    say()
    if BAN is not None:
        say(BAN.banner(VERSION, colour=UI.colour))
    else:
        say("JALEBI %s" % VERSION)
    say()
    say("  " + grey(getattr(BAN, "TAGLINE", "")))
    total = 4 if args.check else 7
    say("  " + ("Checking your setup in %d steps; nothing will be changed." % total if args.check else
                "Let's make a fresh jalebi. %d steps; every command is shown before it runs." % total))

    step(1, total, "Checking the kitchen", "(system)")
    facts = check_kitchen()

    step(2, total, "Choosing the pan", "(environment)")
    if args.check:
        env = {"kind": "current", "python": sys.executable, "create": None, "activate": None}
        if args.env == "conda" and facts["conda"]:
            p = conda_env_prefix(facts["conda"], args.name or "jalebi")
            env.update(kind="conda", python=env_python(p) if p else None, name=args.name or "jalebi")
        elif args.env == "venv":
            path = os.path.abspath(args.venv or ".venv-jalebi")
            py = os.path.join(path, "Scripts", "python.exe") if IS_WIN else os.path.join(path, "bin", "python")
            env.update(kind="venv", python=py if os.path.exists(py) else None)
        info("checking %s" % (env["python"] or "(environment does not exist yet)"))
    else:
        env = choose_env(args, facts)

    step(3, total, "Picking the toppings", "(optional extras)")
    if args.check:
        extras = list(REQ.OPTIONAL)
        info("checking the core and every optional extra (%s)" % ", ".join(extras))
    else:
        extras = choose_extras(args)

    step(4, total, "Counting the ingredients", "(requirements)")
    rows = count_ingredients(env, extras, optional_soft=args.check)
    if args.check:
        say()
        if env.get("python"):
            res = probe(env["python"], ["jalebi"])
            installed = res and res["versions"].get("jalebi")
            (ok if installed else info)("jalebi itself: %s" % (installed or "not installed"))
            if installed:
                info("full check of line lists, data and speed: %s -m jalebi.doctor" % env["python"])
        missing_core = [d for d in rows["missing"] + rows["old"] if d in REQ.CORE]
        if missing_core:
            info("to fix: python install.py   (or: pip install -e \".\")")
        return 0 if not missing_core else 1

    if not facts["internet"] and (rows["missing"] or rows["old"] or env.get("create")):
        warn("pypi.org is not reachable; the next step will probably fail unless pip has a cache or proxy")
    if not ask_yes("Ready to fry? (%s into %s)" % ("install" if env.get("create") is None else "create the environment and install",
                                                 env.get("name") or env.get("path") or env["python"]), True):
        info("nothing changed. Bye!"); return 0

    step(5, total, "Frying", "(installing; full output in install_log.txt)")
    py = fry(env, extras, args)

    step(6, total, "Soaking in syrup", "(jalebi doctor)")
    ready = soak(py)

    step(7, total, "Taste test", "(optional)")
    if ready:
        taste(py, args)
    else:
        info("skipped: fix the problems above first")
    serve_card(env, extras, ready)
    return 0 if ready else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\n  interrupted: nothing half-done is left behind except what the log shows (%s)" % LOG)
        sys.exit(130)
