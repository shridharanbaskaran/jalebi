"""What JALEBI is doing, printed in the terminal (the activity log of the web app).

`jalebi serve` switches this on, so the terminal that runs the server shows every user action, every callback of
the app, every file read, cache hits and misses, each step of a fit, map or rotation diagram with its timing, and
progress bars for the long jobs (reading cubes, grid, optimiser, MCMC, detection, batch, maps).  Library and
CLI use stay quiet: everything here goes through the standard `logging` logger "jalebi", which prints nothing
until `configure()` is called.

    jalebi serve                          # level debug (the default): everything below "trace"
    jalebi serve --log-level info         # user actions, main steps with timings, results, warnings
    jalebi serve --log-level trace        # + slider drags, typing, redraws, pan/zoom
    jalebi serve --log-file serve.log     # also write the log to a file (no colours)
    JALEBI_LOG_LEVEL=info python my.py    # the same switch for pn.serve(make_app) from Python or a notebook

A line reads   14:02:11.348 INF source  s1 │ opened V-HV-TAU-C (12 cubes, 318 MB) in 2.31 s
               time         level area   browser session (s1, s2 … one per browser tab; "bg" = worker thread)

In Python (notebooks, scripts):  `import jalebi.activity as act; act.configure("info")`.
"""
from __future__ import annotations

import functools
import html as _html_mod
import logging
import os
import re
import sys
import threading
import time
from contextlib import contextmanager

TRACE = 5
DEBUG, INFO, WARNING, ERROR = logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR
logging.addLevelName(TRACE, "TRACE")
LOG = logging.getLogger("jalebi")
LOG.addHandler(logging.NullHandler())

LEVELS = {"trace": TRACE, "debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
_TAGS = {TRACE: "TRC", logging.DEBUG: "DBG", logging.INFO: "INF", logging.WARNING: "WRN", logging.ERROR: "ERR", logging.CRITICAL: "CRT"}
_COLOURS = {TRACE: "\033[2m", logging.DEBUG: "\033[2;37m", logging.INFO: "\033[0m", logging.WARNING: "\033[33m",
            logging.ERROR: "\033[31;1m", logging.CRITICAL: "\033[31;1m"}
_AREA_COLOUR = "\033[36m"
_USER_COLOUR = "\033[1;38;5;215m"          # amber: what the user did
_RESET = "\033[0m"

_state = {"configured": False, "handler": None, "file_handler": None, "colour": False, "tty": False,
          "hooks": False, "bars": 0, "level": logging.WARNING}
_tl = threading.local()
_SESSIONS: dict[int, str] = {}
_SESS_COUNT = [0]
_SAVED: dict = {}                              # what configure() changed, for reset()
_SESS_LOCK = threading.Lock()
_write_lock = threading.RLock()


# ------------------------------------------------------------------------------------------------ basics
def get_logger(area: str) -> logging.Logger:
    """The logger of one part of JALEBI ("source", "lte", "cube", "rotdiag", "fit" ...)."""
    area = area.replace("jalebi.", "") if area.startswith("jalebi.") else area
    return logging.getLogger(f"jalebi.{area}")


def enabled(level: int = logging.DEBUG) -> bool:
    """True when the activity log is switched on at `level` (cheap; use it to skip building expensive messages)."""
    return _state["configured"] and LOG.isEnabledFor(level)


def fmt_time(dt: float) -> str:
    if dt < 1e-3:
        return f"{dt * 1e6:.0f} µs"
    if dt < 1:
        return f"{dt * 1e3:.0f} ms"
    if dt < 120:
        return f"{dt:.2f} s"
    m, s = divmod(dt, 60)
    return f"{int(m)} min {s:.0f} s" if m < 60 else f"{int(m // 60)} h {int(m % 60)} min"


def fmt_bytes(n: float) -> str:
    n = float(n)
    for u in ("B", "kB", "MB", "GB"):
        if abs(n) < 1000 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1000.0
    return f"{n:.1f} GB"


def file_size(path) -> str:
    try:
        return fmt_bytes(os.path.getsize(path))
    except OSError:
        return "?"


def short_path(path, n: int = 2) -> str:
    """The last `n` parts of a path (enough to recognise a file without the full machine path)."""
    try:
        parts = os.path.normpath(str(path)).split(os.sep)
    except Exception:
        return str(path)
    return os.sep.join(parts[-n:]) if len(parts) > n else str(path)


_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(s) -> str:
    """Visible text of an HTML snippet, on one line."""
    s = _TAG_RE.sub(" ", str(s or ""))
    s = _html_mod.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def describe(v, n: int = 90) -> str:
    """Short, readable form of a value for the log (no arrays, no file contents)."""
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return f"array{v.shape}"
    except Exception:
        pass
    if isinstance(v, (bytes, bytearray)):
        return fmt_bytes(len(v))
    if isinstance(v, float):
        s = f"{v:.6g}"
    elif isinstance(v, (str, int, bool)) or v is None:
        s = repr(v) if isinstance(v, str) else str(v)
    elif isinstance(v, (list, tuple, set)):
        s = "[" + ", ".join(describe(x, 30) for x in list(v)[:8]) + (", …" if len(v) > 8 else "") + "]"
    elif isinstance(v, dict):
        s = "{" + ", ".join(f"{k}: {describe(x, 25)}" for k, x in list(v.items())[:6]) + (", …" if len(v) > 6 else "") + "}"
    else:
        s = type(v).__name__
        try:
            if hasattr(v, "shape"):
                s += f"{tuple(v.shape)}"
        except Exception:
            pass
    return s if len(s) <= n else s[: n - 1] + "…"


# ------------------------------------------------------------------------------------------------ formatting
def _session_tag() -> str:
    """s1, s2 … for the browser session the current code runs for; 'bg' in worker threads; '—' otherwise."""
    pn_state = None
    if "panel" in sys.modules:
        try:
            from panel.io.state import state as pn_state
        except Exception:
            pn_state = None
    t = threading.current_thread()
    if t is not threading.main_thread():
        # worker thread: the session that started it (bind_thread / thread_target); never Bokeh's curdoc, which is
        # shared by all threads and names whichever session the server thread happens to be serving
        tag = getattr(_tl, "session", None)
        return f"{tag}·bg" if tag else "bg"
    doc = getattr(pn_state, "curdoc", None) if pn_state is not None else None
    if doc is not None:
        with _SESS_LOCK:
            tag = _SESSIONS.get(id(doc))
            if tag is None:
                _SESS_COUNT[0] += 1
                tag = _SESSIONS[id(doc)] = f"s{_SESS_COUNT[0]}"
        return tag
    return "—"


class _Formatter(logging.Formatter):
    def __init__(self, colour: bool):
        super().__init__()
        self.colour = colour

    def format(self, record: logging.LogRecord) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}"
        name = record.name
        area = getattr(record, "area", None) or (name.split(".", 1)[1] if name.startswith("jalebi.") else
                                                   ("jalebi" if name == "jalebi" else name.split(".")[0]))
        area = area[:8]
        sess = getattr(record, "sess", "")
        depth = getattr(record, "depth", 0) if name.startswith("jalebi") else 0
        msg = record.getMessage()
        if record.exc_info:
            msg = msg + "\n" + self.formatException(record.exc_info)
        indent = "  " * min(depth, 12)
        lvl = _TAGS.get(record.levelno, record.levelname[:3])
        user = getattr(record, "user", False)
        if self.colour:
            c = _COLOURS.get(record.levelno, "")
            body = (_USER_COLOUR + msg + _RESET) if user and record.levelno < logging.WARNING else (c + msg + _RESET)
            return f"\033[2m{t}{_RESET} {c}{lvl}{_RESET} {_AREA_COLOUR}{area:<8}{_RESET} \033[2m{sess:<5}│{_RESET} {indent}{body}"
        return f"{t} {lvl} {area:<8} {sess:<5}│ {indent}{msg}"


class _Context(logging.Filter):
    """Adds the session tag and the call depth to every record (also those of bokeh/panel/tornado)."""

    def filter(self, record):
        if not hasattr(record, "sess"):
            try:
                record.sess = _session_tag()
            except Exception:
                record.sess = "?"
        if not hasattr(record, "depth"):
            record.depth = getattr(_tl, "depth", 0)
        _tl.count = getattr(_tl, "count", 0) + 1
        return True


class _Handler(logging.StreamHandler):
    """Writes above any progress bars on screen (tqdm.write) so bars and log lines do not tear each other."""

    def emit(self, record):
        try:
            msg = self.format(record)
            with _write_lock:
                if _state["bars"] > 0 and _state["tty"]:
                    try:
                        from tqdm import tqdm
                        tqdm.write(msg, file=self.stream)
                        return
                    except Exception:
                        pass
                self.stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


def configure(level: str | int = "debug", log_file: str | None = None, colour: bool | None = None, stream=None,
              quiet_libs: bool = True) -> logging.Logger:
    """Print JALEBI's activity in the terminal.  level: trace | debug | info | warning | error.
    Calling it again changes the level (and adds/replaces the log file)."""
    lv = LEVELS.get(str(level).lower(), level) if not isinstance(level, int) else level
    if not isinstance(lv, int):
        raise ValueError(f"unknown log level {level!r}; use one of {', '.join(LEVELS)}")
    stream = stream or sys.stderr
    tty = bool(getattr(stream, "isatty", lambda: False)())
    if colour is None:
        colour = tty and os.environ.get("NO_COLOR") is None and os.environ.get("TERM") != "dumb"
    ctx = _Context()
    if _state["handler"] is None:
        h = _Handler(stream)
        h.addFilter(ctx)
        LOG.addHandler(h)
        _state["handler"] = h
    h = _state["handler"]
    h.setStream(stream) if hasattr(h, "setStream") else None
    h.setFormatter(_Formatter(colour))
    h.setLevel(lv)
    _state.update(configured=True, colour=colour, tty=tty, level=lv)
    LOG.propagate = False
    if log_file:
        if _state["file_handler"] is not None:
            LOG.removeHandler(_state["file_handler"])
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh._jalebi_file = True
        fh.addFilter(_Context())
        fh.setFormatter(_Formatter(False))
        fh.setLevel(lv)
        LOG.addHandler(fh)
        _state["file_handler"] = fh
    LOG.setLevel(lv)
    # other libraries: their warnings and errors go to the same terminal in the same format
    fh = _state["file_handler"]
    for name, libl in (("bokeh", logging.INFO if lv <= logging.INFO else logging.WARNING),
                       ("panel", logging.WARNING), ("tornado", logging.WARNING), ("tornado.access", logging.ERROR),
                       ("py.warnings", logging.WARNING), ("asyncio", logging.WARNING)):
        lg = logging.getLogger(name)
        _SAVED.setdefault(name, (lg.level, lg.propagate))
        if quiet_libs:
            lg.setLevel(libl)
        mine = [x for x in lg.handlers if getattr(x, "_jalebi", False)]
        if not mine:
            lh = _Handler(stream)
            lh._jalebi = True
            lh.addFilter(_Context())
            lg.addHandler(lh)
            mine = [lh]
        for x in mine:
            x.setFormatter(_Formatter(colour))
            if hasattr(x, "setStream") and not isinstance(x, logging.FileHandler):
                x.setStream(stream)
        for x in [x for x in lg.handlers if getattr(x, "_jalebi_file", False) and x is not fh]:
            lg.removeHandler(x)
        if fh is not None and fh not in lg.handlers:
            lg.addHandler(fh)
        lg.propagate = False
    logging.captureWarnings(True)
    _install_thread_excepthook()
    return LOG


def reset():
    """Switch the activity log off again (removes JALEBI's handlers; tests use it)."""
    for h in (_state["handler"], _state["file_handler"]):
        if h is not None:
            LOG.removeHandler(h)
            try:
                h.close() if h is _state["file_handler"] else None
            except Exception:
                pass
    for name in ("bokeh", "panel", "tornado", "tornado.access", "py.warnings", "asyncio"):
        lg = logging.getLogger(name)
        for x in list(lg.handlers):
            if getattr(x, "_jalebi", False) or getattr(x, "_jalebi_file", False):
                lg.removeHandler(x)
        level, prop = _SAVED.pop(name, (logging.NOTSET, True))
        lg.setLevel(level)
        lg.propagate = prop
    logging.captureWarnings(False)
    if getattr(threading, "_jalebi_hook", False) and "excepthook" in _SAVED:
        threading.excepthook = _SAVED.pop("excepthook")
        threading._jalebi_hook = False
    LOG.setLevel(logging.NOTSET)
    LOG.propagate = True
    _state.update(configured=False, handler=None, file_handler=None, bars=0, level=logging.WARNING)


def configure_from_env() -> bool:
    """`JALEBI_LOG_LEVEL` (and `JALEBI_LOG_FILE`) switch the log on without the CLI: the app calls this when it is
    built, so `pn.serve(make_app)` from Python or a notebook prints the same activity log."""
    lv = os.environ.get("JALEBI_LOG_LEVEL")
    if lv and not _state["configured"]:
        configure(lv, log_file=os.environ.get("JALEBI_LOG_FILE") or None)
        return True
    return False


def _install_thread_excepthook():
    if getattr(threading, "_jalebi_hook", False):
        return
    prev = threading.excepthook
    _SAVED["excepthook"] = prev

    def hook(args):
        if args.exc_type is not SystemExit:
            get_logger("thread").error("uncaught error in thread %s: %s: %s", getattr(args.thread, "name", "?"),
                                       args.exc_type.__name__, args.exc_value,
                                       exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        else:
            prev(args)
    threading.excepthook = hook
    threading._jalebi_hook = True


def bind_thread(session: str | None = None):
    """Call at the start of a worker thread so its lines carry the session that started it (`s2·bg`)."""
    _tl.session = session


def current_session() -> str:
    return _session_tag()


def thread_target(fn):
    """Wrap a worker-thread target so its log lines carry the browser session that started it."""
    sess = _session_tag() if _state["configured"] else None

    @functools.wraps(fn)
    def run(*a, **k):
        bind_thread(sess if sess not in ("—", "bg") else None)
        return fn(*a, **k)
    return run


# ------------------------------------------------------------------------------------------------ steps
class _Step:
    def __init__(self, logger, msg, level):
        self.logger, self.msg, self.level = logger, msg, level
        self.notes: list[str] = []
        self.t0 = time.perf_counter()

    def note(self, text):
        """Add a result to the closing line ("… in 1.2 s · 4 lines found")."""
        if text:
            self.notes.append(str(text))

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


@contextmanager
def step(msg: str, area: str = "jalebi", level: int = logging.INFO, start_level: int = logging.DEBUG):
    """Log one operation: '▶ msg' when it starts (debug), 'msg — 1.23 s · notes' when it ends (info), or
    '✗ msg failed after … : error' (error, with the traceback at debug).

        with act.step("reading x1d", "source") as s:
            ...; s.note("12 sub-bands")
    """
    lg = get_logger(area)
    st = _Step(lg, msg, level)
    if not enabled(min(level, start_level)):
        yield st
        return
    lg.log(start_level, "▶ %s …", msg)
    _tl.depth = getattr(_tl, "depth", 0) + 1
    try:
        yield st
    except BaseException as ex:
        _tl.depth = max(getattr(_tl, "depth", 1) - 1, 0)
        if not getattr(ex, "_jalebi_logged", False) and not isinstance(ex, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            lg.error("✗ %s failed after %s: %s: %s", msg, fmt_time(st.elapsed), type(ex).__name__, ex,
                     exc_info=LOG.isEnabledFor(logging.DEBUG))
            try:
                ex._jalebi_logged = True
            except Exception:
                pass
        raise
    _tl.depth = max(getattr(_tl, "depth", 1) - 1, 0)
    lg.log(level, "✓ %s — %s%s", msg, fmt_time(st.elapsed), "".join(f" · {n}" for n in st.notes))


def log(area: str, msg: str, *args, level: int = logging.INFO, user: bool = False, **kw):
    """One line in the activity log (no-op unless configured)."""
    if enabled(level):
        get_logger(area).log(level, msg, *args, extra={"user": user} if user else None, **kw)


def user(area: str, msg: str, *args, level: int = logging.INFO):
    """A line describing something the user did (drawn in amber)."""
    log(area, "👤 " + msg, *args, level=level, user=True)


def debug(area, msg, *args):
    log(area, msg, *args, level=logging.DEBUG)


def info(area, msg, *args):
    log(area, msg, *args, level=logging.INFO)


def warning(area, msg, *args):
    log(area, msg, *args, level=logging.WARNING)


def trace(area, msg, *args):
    log(area, msg, *args, level=TRACE)


# ------------------------------------------------------------------------------------------------ progress
class Progress:
    """A progress bar in the terminal for one long job.

    On a terminal it is a live tqdm bar (log lines are printed above it); when the output is redirected
    (log file, `nohup`) it becomes a line every 10 % (at most every 2 s).  It does nothing unless the
    activity log is on at level info or lower.

        bar = act.Progress("reading cubes", total=12, unit="cube", area="source")
        bar.update(n=3, extra="ch2-medium")       # or bar.update(frac=0.25)
        bar.close()                               # logs "reading cubes: 12/12 cube in 3.1 s"
    """

    def __init__(self, label: str, total: float | None = None, unit: str = "", area: str = "jalebi", level: int = logging.INFO):
        self.label, self.total, self.unit, self.area, self.level = label, total, unit, area, level
        self.on = enabled(level)
        self.t0 = time.perf_counter()
        self.n = 0.0
        self.frac = 0.0
        self.extra = ""
        self._bar = None
        self._last_line = (-1.0, 0.0)          # (fraction, time) of the last text line
        self.closed = False
        if not self.on:
            return
        get_logger(area).log(logging.DEBUG, "▶ %s%s", label, f" ({total:g} {unit})" if total else "")
        if _state["tty"]:
            try:
                from tqdm import tqdm
                with _write_lock:
                    sess = _session_tag()
                    desc = f"{sess} {label}" if sess not in ("—",) else label
                    self._bar = tqdm(total=_num(total) if total else 100, desc=desc[:48], unit=unit or "%",
                                     leave=False, dynamic_ncols=True, file=_state["handler"].stream,
                                     bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}{postfix}]",
                                     mininterval=0.2)
                    _state["bars"] += 1
            except Exception:
                self._bar = None

    def update(self, n: float | None = None, frac: float | None = None, extra: str = "", total: float | None = None):
        if not self.on or self.closed:
            return
        if total is not None and total != self.total:
            self.total = total
            if self._bar is not None:
                self._bar.total = _num(total); self._bar.refresh()
        if n is not None:
            self.n = float(n)
            self.frac = self.n / self.total if self.total else 0.0
        elif frac is not None:
            self.frac = max(0.0, min(1.0, float(frac)))
            self.n = self.frac * (self.total or 100.0)
        self.extra = extra or self.extra
        if self._bar is not None:
            with _write_lock:
                self._bar.n = _num(self.n) if self.total else int(round(100.0 * self.frac))
                if extra:
                    self._bar.set_postfix_str(str(extra)[:40], refresh=False)
                self._bar.refresh()
            return
        now = time.perf_counter()
        lf, lt = self._last_line
        if self.frac - lf >= 0.1 and now - lt >= 2.0 or self.frac >= 1.0 and lf < 1.0:
            self._last_line = (self.frac, now)
            el = now - self.t0
            eta = el * (1 - self.frac) / self.frac if self.frac > 0.01 else float("nan")
            nb = int(round(20 * self.frac))
            cnt = f" {self.n:g}/{self.total:g} {self.unit}" if self.total else ""
            get_logger(self.area).log(self.level, "%s ▕%s▏ %3.0f%%%s · %s%s%s", self.label, "█" * nb + "░" * (20 - nb),
                                      100 * self.frac, cnt, fmt_time(el), f" · ETA {fmt_time(eta)}" if eta == eta else "",
                                      f" · {self.extra}" if self.extra else "")

    def close(self, msg: str | None = None, ok: bool = True):
        if not self.on or self.closed:
            return
        self.closed = True
        if self._bar is not None:
            with _write_lock:
                try:
                    self._bar.close()
                finally:
                    _state["bars"] = max(_state["bars"] - 1, 0)
        el = time.perf_counter() - self.t0
        what = msg or (f"{self.n:g}/{self.total:g} {self.unit}".strip() if self.total else f"{100 * self.frac:.0f} %")
        get_logger(self.area).log(self.level if ok else logging.WARNING, "%s %s: %s in %s", "✓" if ok else "■",
                                  self.label, what, fmt_time(el))

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        self.close(ok=et is None)
        return False


def _num(x):
    """An int when the number is whole (tqdm then prints 3/12, not 3.0/12.0)."""
    try:
        return int(x) if float(x).is_integer() else float(x)
    except (TypeError, ValueError):
        return x


class Stages:
    """Progress bars for a job with named stages (a fit: grid H2O_hot → grid CO → optimise → mcmc): a new
    bar starts whenever the stage label changes and the previous one is closed."""

    def __init__(self, title: str, area: str = "jalebi"):
        self.title, self.area = title, area
        self.bar: Progress | None = None
        self.label = None
        self._lock = threading.Lock()

    def update(self, label: str, frac: float | None = None, n: float | None = None, total: float | None = None,
               unit: str = "", extra: str = ""):
        if not enabled(logging.INFO):
            return
        with self._lock:
            if label != self.label:
                if self.bar is not None:          # a new stage started: the previous one is complete
                    self.bar.update(frac=1.0)
                    self.bar.close()
                self.label = label
                self.bar = Progress(f"{self.title} · {label}" if self.title else label, total=total, unit=unit, area=self.area)
            self.bar.update(n=n, frac=frac, extra=extra, total=total)

    def close(self, ok: bool = True):
        with self._lock:
            if self.bar is not None:
                self.bar.close(ok=ok)
            self.bar, self.label = None, None


class Job:
    """A background job of a web-app module (maps, PV cut, line measurement, rotation-diagram fit …):
    '▶ label …' when it starts, a progress bar (when the job reports its fraction) or a 'still running' line every
    10 s, and '✓ label — 3.2 s' / '✗ label failed after … ' with the traceback at the end."""

    def __init__(self, label: str, area: str, bar: bool = False, heartbeat_s: float = 10.0):
        self.label, self.area, self.heartbeat_s = label, area, heartbeat_s
        self.t0 = self.last = time.perf_counter()
        self.lg = get_logger(area)
        self.bar = Progress(label, area=area) if bar and enabled(logging.INFO) else None
        if enabled(logging.INFO):
            self.lg.info("▶ %s …", label)

    def tick(self, frac: float | None = None, extra: str = ""):
        if not enabled(logging.INFO):
            return
        if self.bar is not None and frac is not None:
            self.bar.update(frac=frac, extra=extra)
            return
        now = time.perf_counter()
        if now - self.last >= self.heartbeat_s:
            self.last = now
            self.lg.info("… %s still running (%s)%s", self.label, fmt_time(now - self.t0), f" · {extra}" if extra else "")

    def end_bar(self, ok: bool = True):
        """Close the progress bar (call it from the worker when the work ends, so a closed browser tab, whose poll
        never runs again, does not leave a bar on the terminal)."""
        if self.bar is not None:
            self.bar.close(ok=ok)

    def done(self, note: str = ""):
        if self.bar is not None:
            self.bar.close(ok=True)
        if enabled(logging.INFO):
            self.lg.info("✓ %s — %s%s", self.label, fmt_time(time.perf_counter() - self.t0), f" · {note}" if note else "")

    def failed(self, error_text: str):
        if self.bar is not None:
            self.bar.close("failed", ok=False)
        if enabled(logging.ERROR):
            self.lg.error("✗ %s failed after %s:\n%s", self.label, fmt_time(time.perf_counter() - self.t0), str(error_text).rstrip())


# ------------------------------------------------------------------------------------------------ callbacks
def _argstr(a, k) -> str:
    parts = []
    for x in a[:3]:
        if isinstance(x, (str, int, float, bool)) or x is None:
            parts.append(describe(x, 40))
        elif type(x).__name__ == "Event":                # param event of a widget watcher
            parts.append(f"{getattr(x, 'name', '')}={describe(getattr(x, 'new', None), 40)}")
        else:
            parts.append(type(x).__name__)
    for kk, x in list(k.items())[:3]:
        parts.append(f"{kk}={describe(x, 30)}")
    return "(" + ", ".join(parts) + ")"


def _wrap_method(fn, qual: str, area: str, level: int):
    lg = get_logger(area)

    @functools.wraps(fn)
    def w(*a, **k):
        if not _state["configured"] or not lg.isEnabledFor(level):
            return fn(*a, **k)
        lg.log(level, "→ %s%s", qual, _argstr(a[1:], k))
        n0 = getattr(_tl, "count", 0)
        _tl.depth = getattr(_tl, "depth", 0) + 1
        t0 = time.perf_counter()
        try:
            r = fn(*a, **k)
        except BaseException as ex:
            _tl.depth = max(getattr(_tl, "depth", 1) - 1, 0)
            if not getattr(ex, "_jalebi_logged", False) and isinstance(ex, Exception):
                lg.error("✗ %s raised %s: %s", qual, type(ex).__name__, ex, exc_info=LOG.isEnabledFor(logging.DEBUG))
                try:
                    ex._jalebi_logged = True
                except Exception:
                    pass
            raise
        _tl.depth = max(getattr(_tl, "depth", 1) - 1, 0)
        dt = time.perf_counter() - t0
        if dt >= 0.05 or getattr(_tl, "count", 0) > n0:
            lg.log(level, "← %s %s", qual, fmt_time(dt))
        return r
    w._jalebi_traced = True
    return w


def traced(area: str, skip: tuple = (), hot: tuple = (), level: int = logging.DEBUG):
    """Class decorator: log every call of the class's methods (debug: '→ Class.method(args)' and, when it took
    ≥ 50 ms or logged something itself, '← Class.method 0.31 s').  `skip`: never logged; `hot`: called on every
    slider tick / poll, logged at trace level only.  Properties and dunder methods (except __init__) are left alone."""
    def deco(cls):
        for name, fn in list(vars(cls).items()):
            if name in skip or (name.startswith("__") and name != "__init__"):
                continue
            if isinstance(fn, (staticmethod, classmethod, property)) or not callable(fn) or isinstance(fn, type):
                continue
            if getattr(fn, "_jalebi_traced", False):
                continue
            setattr(cls, name, _wrap_method(fn, f"{cls.__name__}.{name}", area, TRACE if name in hot else level))
        return cls
    return deco


def watch_text(pane, area: str, label: str = "status", level: int = logging.INFO):
    """Log the visible text of an HTML/Markdown pane whenever the app changes it (status lines, notes).
    Consecutive '⏳ …' progress texts of one pane are logged at most every 2 s."""
    if pane is None or getattr(pane, "_jalebi_watched", False):
        return pane
    last = {"text": None, "t": 0.0}
    lg = get_logger(area)

    def cb(e):
        if not _state["configured"]:
            return
        txt = strip_html(e.new)
        if not txt or txt == last["text"] or txt in ("—", "-"):
            return
        now = time.perf_counter()
        busy = txt.startswith("⏳") or txt.startswith("…")
        if busy and (last["text"] or "").startswith("⏳") and now - last["t"] < 2.0:
            return
        last.update(text=txt, t=now)
        lv = logging.WARNING if txt.startswith("⚠") or "failed" in txt.lower()[:40] else level
        if len(txt) > 400:
            txt = txt[:399] + "…"
        lg.log(lv if not busy else logging.DEBUG, "%s: %s", label, txt)
    try:
        pane.param.watch(cb, "object")
        pane._jalebi_watched = True
    except Exception:
        pass
    return pane


def tag(obj, area: str):
    """Mark every widget inside a Panel layout as belonging to `area` (for the 'user …' lines)."""
    try:
        import panel as pn
        ws = list(obj.select(pn.widgets.Widget)) if hasattr(obj, "select") else []
        if isinstance(obj, pn.widgets.Widget):
            ws.append(obj)
        for w in ws:
            if getattr(w, "_jalebi_area", None) is None:
                w._jalebi_area = area
            # the parts of a composite widget (editable slider = slider + number box) carry its label
            comp = getattr(w, "_composite", None)
            if comp is not None and hasattr(comp, "select"):
                for inner in comp.select(pn.widgets.Widget):
                    if inner is not w:
                        inner._jalebi_area = getattr(inner, "_jalebi_area", None) or area
                        if not getattr(inner, "name", "") and getattr(w, "name", ""):
                            inner._jalebi_label = getattr(w, "_jalebi_label", None) or w.name
        if isinstance(obj, pn.widgets.Widget) and getattr(obj, "_jalebi_area", None) is None:
            obj._jalebi_area = area
        for t in (obj.select(pn.Tabs) if hasattr(obj, "select") else []):
            if getattr(t, "_jalebi_area", None) is None:
                t._jalebi_area = area
    except Exception:
        pass
    return obj


# ------------------------------------------------------------------------------------------------ web hooks
_NOISE_PROPS = {"width", "height", "min_width", "min_height", "max_width", "max_height", "sizing_mode", "scroll_position",
                "scroll_button_threshold", "visible", "css_classes", "loading", "margin", "styles", "stylesheets",
                "_inner_width", "_inner_height", "description", "dark", "theme", "children", "inner_width", "inner_height",
                "outer_width", "outer_height", "computed_width", "computed_height"}
_CHATTY_PROPS = {"value_input", "value_throttled_input"}


def _widget_label(w) -> str:
    name = getattr(w, "_jalebi_label", None) or getattr(w, "name", "") or ""
    if not name or re.fullmatch(r"[A-Z][A-Za-z]+\d{3,}", name):
        name = getattr(w, "label", "") or getattr(w, "placeholder", "") or type(w).__name__
    name = strip_html(name)[:60]
    prefix = getattr(w, "_jalebi_prefix", None)
    if prefix is not None:
        try:
            p = prefix() if callable(prefix) else prefix
            if p:
                name = f"{p} · {name}"
        except Exception:
            pass
    return name


def label_widgets(obj, labels: dict | None = None, prefix=None):
    """Give the widgets held as attributes of `obj` readable names for the log: unnamed ones get their attribute
    name (or `labels[attr]`); `prefix` (a string or a function) is put in front, e.g. the component name of a card."""
    try:
        import panel as pn
    except Exception:
        return
    labels = labels or {}
    for attr, w in list(vars(obj).items()):
        if not isinstance(w, pn.widgets.Widget):
            continue
        if attr in labels:
            w._jalebi_label = labels[attr]
        elif not getattr(w, "name", ""):
            w._jalebi_label = attr.strip("_").replace("_", " ")
        if prefix is not None:
            w._jalebi_prefix = prefix
        comp = getattr(w, "_composite", None)
        if comp is not None and hasattr(comp, "select"):
            for inner in comp.select(pn.widgets.Widget):
                if inner is not w:
                    inner._jalebi_label = getattr(w, "_jalebi_label", None) or w.name
                    if prefix is not None:
                        inner._jalebi_prefix = prefix


def _describe_widget_events(obj, events: dict):
    """(level, text) lines for one batch of browser → server property changes of a Panel component."""
    import panel as pn
    out = []
    cls = type(obj).__name__
    if isinstance(obj, pn.widgets.Widget):
        label = _widget_label(obj)
        has_throttle = "value_throttled" in obj.param and "Slider" in type(obj).__name__
        for k, v in events.items():
            if k in _NOISE_PROPS:
                continue
            if isinstance(obj, pn.widgets.PasswordInput) and k in ("value", "value_input"):
                if k == "value":
                    out.append((logging.INFO, f"typed into '{label}' (hidden)"))
                continue
            if k in _CHATTY_PROPS:
                out.append((TRACE, f"typing in '{label}': {describe(v, 60)}")); continue
            if k == "value" and has_throttle:
                out.append((TRACE, f"dragging '{label}' → {describe(v)}")); continue
            if k == "value_throttled":
                out.append((logging.INFO, f"set '{label}' = {describe(v)}")); continue
            if isinstance(obj, pn.widgets.FileInput):
                if k == "filename":
                    out.append((logging.INFO, f"uploaded file {describe(v, 120)} ({label})"))
                continue
            if k == "active" and isinstance(getattr(obj, "options", None), (list, dict)) and not isinstance(v, bool):
                labs = list(obj.options)
                try:
                    shown = [labs[i] for i in v] if isinstance(v, (list, tuple)) else (labs[v] if v is not None else None)
                except Exception:
                    shown = v
                out.append((logging.INFO, f"chose '{label}' = {describe(shown)}")); continue
            if k == "active" and isinstance(obj, (pn.widgets.Toggle, pn.widgets.Checkbox)):
                out.append((logging.INFO, f"{'ticked' if v else 'unticked'} '{label}'")); continue
            if k in ("value", "active", "options"):
                verb = "ticked" if v is True else "unticked" if v is False else "set"
                out.append((logging.INFO, f"{verb} '{label}'" + ("" if isinstance(v, bool) else f" = {describe(v)}"))); continue
            if k in ("indices", "selection"):
                out.append((logging.INFO, f"selected rows {describe(v)} in '{label}'")); continue
            if k == "data":
                out.append((logging.INFO, f"edited the table '{label}'")); continue
            out.append((logging.DEBUG, f"'{label}' ({cls}).{k} = {describe(v)}"))
    elif isinstance(obj, pn.Tabs):
        for k, v in events.items():
            if k == "active":
                try:
                    names = obj._names if hasattr(obj, "_names") else [getattr(o, "name", "") for o in obj.objects]
                    name = names[v]
                except Exception:
                    name = v
                if getattr(obj, "_jalebi_nolog", False):
                    out.append((logging.DEBUG, f"tab → {name}"))
                else:
                    out.append((logging.INFO, f"opened the '{name}' tab"))
    elif isinstance(obj, (pn.Card,)):
        if "collapsed" in events:
            out.append((logging.DEBUG, f"{'collapsed' if events['collapsed'] else 'expanded'} '{strip_html(obj.title or '')}'"))
    else:
        keys = [k for k in events if k not in _NOISE_PROPS]
        if keys:
            out.append((TRACE, f"{cls}: {', '.join(f'{k}={describe(events[k], 40)}' for k in keys)}"))
    return out


def _bokeh_model_label(m) -> str:
    try:
        t = getattr(m, "title", None)
        t = getattr(t, "text", t)
        if t:
            return f"'{strip_html(t)[:50]}'"
    except Exception:
        pass
    return type(m).__name__ if m is not None else "plot"


_QUIET_BOKEH_EVENTS = {"mousemove", "mouseenter", "mouseleave", "wheel", "pan", "panstart", "panend", "pinch",
                       "pinchstart", "pinchend", "rotate", "rotatestart", "rotateend", "lodstart", "lodend",
                       "rangesupdate", "press_up", "presspress", "connection_lost_ignored"}


def _log_bokeh_event(event):
    name = getattr(event, "event_name", type(event).__name__)
    if name in ("button_click", "menu_item_click"):
        return                                               # the Panel button hook logs these
    lg = get_logger("ui")
    model = getattr(event, "model", None)
    if name in _QUIET_BOKEH_EVENTS:
        if name in ("rangesupdate",) and lg.isEnabledFor(TRACE):
            lg.log(TRACE, "pan/zoom on %s: x %s–%s", _bokeh_model_label(model), describe(getattr(event, "x0", None)),
                   describe(getattr(event, "x1", None)))
        return
    if name in ("tap", "doubletap", "press"):
        x, y = getattr(event, "x", None), getattr(event, "y", None)
        lg.info("👤 %s on %s at x=%s, y=%s", {"tap": "clicked", "doubletap": "double-clicked", "press": "long-pressed"}[name],
                _bokeh_model_label(model), describe(x), describe(y), extra={"user": True})
    elif name == "reset":
        lg.info("👤 reset the view of %s", _bokeh_model_label(model), extra={"user": True})
    elif name == "selectiongeometry":
        g = getattr(event, "geometry", {}) or {}
        lg.debug("👤 drew a %s selection on %s", g.get("type", "?"), _bokeh_model_label(model), extra={"user": True})
    elif name == "document_ready":
        lg.info("browser page ready")
    elif name == "connection_lost":
        lg.info("browser connection closed (tab closed, reloaded or network lost)")
    elif name == "clear_input":
        pass
    else:
        lg.debug("browser event %s on %s", name, _bokeh_model_label(model))


def _log_bokeh_change(event):
    """Browser-originated changes of Bokeh models that are not Panel widgets: plot data edited with the draw
    tools, point selections, pan/zoom (trace)."""
    setter = getattr(event, "setter", None)
    if setter is None or type(setter).__name__ not in ("ServerSession", "ServerConnection"):
        return
    m = getattr(event, "model", None)
    attr = getattr(event, "attr", None)
    tn = type(m).__name__
    lg = get_logger("ui")
    if tn == "ColumnDataSource" and attr in ("data", None):
        try:
            n = max((len(v) for v in m.data.values()), default=0)
        except Exception:
            n = "?"
        lg.debug("👤 edited plot data (draw/edit tool): %s now has %s rows", m.name or "source", n, extra={"user": True})
    elif tn == "Selection" and attr == "indices":
        lg.debug("👤 selected %d point(s)", len(getattr(event, "new", []) or []), extra={"user": True})
    elif tn in ("Range1d", "DataRange1d") and lg.isEnabledFor(TRACE):
        lg.log(TRACE, "pan/zoom: %s.%s = %s", tn, attr, describe(getattr(event, "new", None)))


def install_web_hooks():
    """Patch Panel/Bokeh once per process so browser actions, notifications, callback errors and browser
    sessions appear in the activity log.  Safe to call many times; does nothing until `configure()`."""
    if _state["hooks"]:
        return
    _state["hooks"] = True
    try:
        import panel as pn
        from panel.reactive import Syncable
        orig_events = Syncable._process_events

        def _process_events(self, events):
            if _state["configured"]:
                try:
                    area = getattr(self, "_jalebi_area", None) or "ui"
                    lg = get_logger(area)
                    for lv, txt in _describe_widget_events(self, dict(events)):
                        if lg.isEnabledFor(lv):
                            lg.log(lv, "👤 " + txt, extra={"user": True})
                except Exception:
                    pass
            return orig_events(self, events)
        Syncable._process_events = _process_events

        # tables (Tabulator, DataFrame): their data edits and row selections are taken out of the events before
        # Syncable sees them
        from panel.reactive import ReactiveData
        orig_data_events = ReactiveData._process_events

        def _process_data_events(self, events):
            if _state["configured"] and isinstance(events, dict) and ("data" in events or "indices" in events):
                try:
                    area = getattr(self, "_jalebi_area", None) or "ui"
                    sub = {k: events[k] for k in ("data", "indices") if k in events}
                    if "data" in sub:
                        sub["data"] = "…"
                    for lv, txt in _describe_widget_events(self, sub):
                        get_logger(area).log(lv, "👤 " + txt, extra={"user": True})
                except Exception:
                    pass
            return orig_data_events(self, events)
        ReactiveData._process_events = _process_data_events

        from panel.widgets.button import Button, MenuButton

        def _click_hook(cls):
            orig = cls._process_event

            def _process_event(self, event):
                if _state["configured"] and LOG.isEnabledFor(logging.INFO):
                    try:
                        item = getattr(event, "item", None)
                        area = getattr(self, "_jalebi_area", None) or "ui"
                        get_logger(area).info("👤 clicked '%s'%s", _widget_label(self), f" → {item}" if item else "",
                                              extra={"user": True})
                    except Exception:
                        pass
                return orig(self, event)
            cls._process_event = _process_event
        _click_hook(Button)
        _click_hook(MenuButton)

        # notifications (the toasts in the browser) also go to the terminal
        from panel.io.notifications import NotificationAreaBase
        orig_send = NotificationAreaBase.send

        def send(self, message, duration=3000, type=None, background=None, icon=None):
            if _state["configured"]:
                lv = {"error": logging.ERROR, "warning": logging.WARNING}.get(type, logging.INFO)
                get_logger("notice").log(lv, "🔔 %s%s", f"[{type}] " if type else "", strip_html(message))
            return orig_send(self, message, duration=duration, type=type, background=background, icon=icon)
        NotificationAreaBase.send = send
        for sub in NotificationAreaBase.__subclasses__():
            if "send" in vars(sub):
                o = sub.send

                def s2(self, message, duration=3000, type=None, background=None, icon=None, _o=o):
                    if _state["configured"]:
                        lv = {"error": logging.ERROR, "warning": logging.WARNING}.get(type, logging.INFO)
                        get_logger("notice").log(lv, "🔔 %s%s", f"[{type}] " if type else "", strip_html(message))
                    return _o(self, message, duration=duration, type=type, background=background, icon=icon)
                sub.send = s2

        # errors raised in callbacks: one readable line + traceback
        def on_exception(ex):
            if not _state["configured"]:
                raise ex                         # log off: Panel's own behaviour (the error propagates)
            if getattr(ex, "_jalebi_logged", False):
                return
            get_logger("ui").error("callback failed: %s: %s", type(ex).__name__, ex, exc_info=(type(ex), ex, ex.__traceback__))
        if pn.config.exception_handler is None:
            pn.config.exception_handler = on_exception

        def created(ctx):
            if _state["configured"]:
                try:
                    req = getattr(ctx, "request", None)
                    who = getattr(req, "remote_ip", None) or ""
                except Exception:
                    who = ""
                get_logger("server").info("browser tab connected%s", f" from {who}" if who else "")

        def destroyed(ctx):
            doc = getattr(ctx, "_document", None)
            with _SESS_LOCK:
                tag = _SESSIONS.pop(id(doc), None) if doc is not None else None
            if _state["configured"]:
                get_logger("server").info("browser session %s ended: its widgets and plots are freed (open sources stay cached)",
                                          tag or str(getattr(ctx, "id", "?"))[:8], extra={"sess": tag or "—"})
        if pn.state.curdoc is None:              # cannot be registered from inside a session (`panel serve app.py`)
            try:
                pn.state.on_session_created(created)
                pn.state.on_session_destroyed(destroyed)
            except Exception:
                pass
    except Exception as ex:                                       # never break the app for the log
        get_logger("ui").debug("could not hook Panel: %s", ex)
    try:
        from bokeh.document.callbacks import DocumentCallbackManager
        orig_trigger = DocumentCallbackManager.trigger_event

        def trigger_event(self, event):
            if _state["configured"]:
                try:
                    _log_bokeh_event(event)
                except Exception:
                    pass
            return orig_trigger(self, event)
        DocumentCallbackManager.trigger_event = trigger_event
        orig_change = DocumentCallbackManager.trigger_on_change

        def trigger_on_change(self, event):
            if _state["configured"] and LOG.isEnabledFor(logging.DEBUG):
                try:
                    _log_bokeh_change(event)
                except Exception:
                    pass
            return orig_change(self, event)
        DocumentCallbackManager.trigger_on_change = trigger_on_change
    except Exception as ex:
        get_logger("ui").debug("could not hook Bokeh: %s", ex)


def banner_lines(**facts) -> None:
    """Log the server's start-up facts (version, level, folders, caches)."""
    lg = get_logger("server")
    for k, v in facts.items():
        if v is not None and v != "":
            lg.info("%-14s %s", k, v)
