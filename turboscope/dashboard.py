"""
turboscope/dashboard.py

Live terminal dashboard for TurboScopeIndex observability.

Entry points
------------
Programmatic use — keep a dashboard running while your index is live::

    from turboscope import TurboScopeIndex
    from turboscope.dashboard import Dashboard

    index = TurboScopeIndex(dim=1536, bit_width=4)
    # ... populate the index ...

    dash = Dashboard(index)
    dash.run()                      # blocks; Ctrl-C to exit
    # or: dash.run(duration_s=60)   # run for 60 seconds then return

JSON export (for Prometheus / Grafana piping)::

    dash.run(json_mode=True)        # emits one JSON line per refresh to stdout

One-shot print (for scripts, CI)::

    from turboscope.dashboard import print_snapshot
    print_snapshot(index)

CLI (from __main__ when this module is run directly)::

    python -m turboscope.dashboard --index my_index.tvim [--json] [--interval 2.0]

Layout
------

  ┌──────────────────────────────────────────────────────────┐
  │  turboscope · dim=1536 · 4-bit · 10,234 vectors  14:23  │
  ├─────────────────────────────┬────────────────────────────┤
  │  RECALL                     │  DRIFT                     │
  │  @1   ████████████░░  0.912 │  severity  ██░░░░░░  0.065 │
  │  @5   █████████████░  0.961 │  ks stat              0.039│
  │  @10  ██████████████  0.981 │  var shift            0.055│
  │       247 shadow q    ±0.008│  1,000 baseline vectors    │
  ├─────────────────────────────┼────────────────────────────┤
  │  LATENCY         THROUGHPUT │  MEMORY                    │
  │  p50    2.31 ms  qps  1,234 │  compressed       7.9 MB  │
  │  p95    4.12 ms  c/s    123 │  float32 equiv   62.9 MB  │
  │  p99   11.43 ms  tot  1.2 M │  ratio               8×   │
  │  ▂▁▁█▁▁▁  ▁▁▂▃▁  skip   64%│  1536-dim  4-bit           │
  └─────────────────────────────┴────────────────────────────┘

Colors
------
- Recall bars: green ≥ 0.9, yellow ≥ 0.7, red < 0.7
- Drift bar:   green < 0.15, yellow < 0.35, red ≥ 0.35
- Latency p99: green < 5 ms, yellow < 20 ms, red ≥ 20 ms
- Header: yellow ⚠ badge when recalibrate_suggested
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from turboscope.index import TurboScopeIndex
    from turboscope.stats import Snapshot


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"
_REFRESH_DEFAULT = 1.0   # seconds between renders
_SPARK_HISTORY   = 40    # number of p99 samples kept for the sparkline


# ---------------------------------------------------------------------------
# Primitive renderers
# ---------------------------------------------------------------------------

def _gauge(
    value: float,
    width: int = 18,
    color_fn=None,
) -> Text:
    """
    Render a horizontal block-character gauge bar.

    Parameters
    ----------
    value : float
        Value in [0, 1].  Negative values render as 'n/a'.
    width : int
        Number of block characters in the full bar.
    color_fn : callable(float) -> str or None
        Maps value to a rich color string.  If None, uses white.

    Returns
    -------
    rich.text.Text
        Single-line Text with the bar and numeric value appended.
    """
    if value < 0:
        return Text("n/a", style="dim")
    filled = max(0, min(width, round(value * width)))
    bar = "█" * filled + "░" * (width - filled)
    style = color_fn(value) if color_fn else "white"
    t = Text(no_wrap=True)
    t.append(bar, style=style)
    t.append(f" {value:.3f}", style="bold")
    return t


def _recall_color(v: float) -> str:
    return "green" if v >= 0.9 else ("yellow" if v >= 0.7 else "red")


def _drift_color(v: float) -> str:
    return "green" if v < 0.15 else ("yellow" if v < 0.35 else "red bold")


def _p99_color(v_ms: float) -> str:
    return "green" if v_ms < 5 else ("yellow" if v_ms < 20 else "red")


def _sparkline(values: list[float], width: int = 20) -> str:
    """
    Render a Unicode block sparkline from a list of floats.

    Uses the last ``width`` values.  Returns a dash-line if there are
    fewer than 2 values.
    """
    if len(values) < 2:
        return "─" * min(width, max(1, len(values)))
    arr = np.array(values[-width:], dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return "▄" * len(arr)
    norm = (arr - lo) / (hi - lo)
    return "".join(_SPARK_CHARS[round(v * 8)] for v in norm)


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} MB"
    if n >= 1_000:
        return f"{n / 1e3:.1f} KB"
    return f"{n} B"


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} M"
    if n >= 1_000:
        return f"{n / 1e3:.1f} K"
    return str(n)


# ---------------------------------------------------------------------------
# Panel builders — each returns a rich.panel.Panel
# ---------------------------------------------------------------------------

def _recall_panel(snap: "Snapshot") -> Panel:
    t = Table.grid(padding=(0, 1))
    t.add_column(width=4,  style="bold",  no_wrap=True)
    t.add_column(width=24, no_wrap=True)
    t.add_column(width=7,  style="dim",   no_wrap=True)

    def row(label, val, ci):
        ci_str = f"±{ci:.3f}" if val >= 0 and ci >= 0 else ""
        t.add_row(label, _gauge(val, width=18, color_fn=_recall_color), ci_str)

    row("@1",  snap.recall_at_1,  snap.recall_ci.get(1,  -1.0))
    row("@5",  snap.recall_at_5,  snap.recall_ci.get(5,  -1.0))
    row("@10", snap.recall_at_10, snap.recall_ci.get(10, -1.0))

    # Status line: shadow sample count or "waiting"
    if snap.recall_at_10 < 0:
        status = Text("waiting for shadow samples…", style="dim")
    else:
        status = Text(
            f"{snap.window_calls} calls sampled",
            style="dim",
        )
    t.add_row("", status, "")

    border = "blue" if snap.recall_at_10 < 0 else (
        "green" if snap.recall_at_10 >= 0.9 else "yellow"
    )
    return Panel(t, title="[bold]Recall[/bold]", border_style=border)


def _drift_panel(snap: "Snapshot") -> Panel:
    t = Table.grid(padding=(0, 1))
    t.add_column(width=10, style="dim",  no_wrap=True)
    t.add_column(width=24, no_wrap=True)

    if snap.drift_severity < 0:
        t.add_row("", Text("waiting for baseline…", style="dim"))
        # snap.window_calls counts search() calls, not vectors — don't show
        # it here as "vectors needed". A generic hint is accurate enough.
        t.add_row("", Text("add ≥1 000 vectors to establish baseline", style="dim"))
    else:
        t.add_row(
            "severity",
            _gauge(snap.drift_severity, width=18, color_fn=_drift_color),
        )

        def stat_row(label, val):
            if val < 0:
                t.add_row(label, Text("n/a", style="dim"))
            else:
                style = _drift_color(val) if label == "ks stat" else "white"
                t.add_row(label, Text(f"{val:.3f}", style=style))

        stat_row("ks stat",   snap.drift_ks_stat)
        stat_row("var shift", snap.drift_rel_var_shift)

        if snap.recalibrate_suggested:
            t.add_row("", Text("⚠  recalibrate suggested", style="yellow bold"))
        else:
            t.add_row("", Text("distribution stable", style="green dim"))

    border = (
        "dim" if snap.drift_severity < 0 else
        "red" if snap.recalibrate_suggested else
        "yellow" if snap.drift_severity >= 0.15 else
        "green"
    )
    return Panel(t, title="[bold]Drift[/bold]", border_style=border)


def _latency_throughput_panel(
    snap: "Snapshot",
    p99_history: list[float],
) -> Panel:
    """
    Combined latency + throughput + filtering panel.

    Puts latency metrics on the left, throughput/filtering on the right,
    with a sparkline of recent p99 values spanning the full width.
    """
    inner = Table.grid(padding=(0, 1))
    inner.add_column(width=6,  style="dim",  no_wrap=True)  # label
    inner.add_column(width=10, style="bold", no_wrap=True)  # value
    inner.add_column(width=8,  style="dim",  no_wrap=True)  # label2
    inner.add_column(width=8,  style="bold", no_wrap=True)  # value2

    def ms(v):
        return "n/a" if v < 0 else f"{v:.2f} ms"

    p99_style = _p99_color(snap.p99_ms) if snap.p99_ms >= 0 else "dim"

    inner.add_row(
        "p50",  ms(snap.p50_ms),
        "qps",  _fmt_count(int(snap.qps)) if snap.qps > 0 else "n/a",
    )
    inner.add_row(
        "p95",  ms(snap.p95_ms),
        "c/s",  f"{snap.calls_per_s:.1f}" if snap.calls_per_s > 0 else "n/a",
    )
    inner.add_row(
        "p99",  Text(ms(snap.p99_ms), style=p99_style),
        "tot Q", _fmt_count(snap.total_queries),
    )
    inner.add_row(
        "mean", ms(snap.mean_ms),
        "skip/q",
        f"{snap.blocks_skipped_per_query:.0f}"
        if snap.blocks_skipped_per_query > 0 else "—",
    )

    # Sparkline row — full width, styled cyan
    spark = _sparkline(p99_history, width=_SPARK_HISTORY)
    spark_label = "p99 history"
    inner.add_row(
        Text(spark_label, style="dim"),
        Text(spark, style="cyan"),
        "",
        "",
    )

    return Panel(
        inner,
        title="[bold]Latency · Throughput[/bold]",
        border_style="cyan",
    )


def _memory_panel(
    snap: "Snapshot",
    n_vectors: int,
    dim: int | None,
    bit_width: int,
) -> Panel:
    t = Table.grid(padding=(0, 1))
    t.add_column(width=11, style="dim",  no_wrap=True)
    t.add_column(width=10, style="bold", no_wrap=True)

    if dim is not None and n_vectors > 0:
        compressed = n_vectors * dim * bit_width // 8
        fp32_bytes = n_vectors * dim * 4
        ratio = fp32_bytes / compressed if compressed > 0 else 0
        t.add_row("compressed", _fmt_bytes(compressed))
        t.add_row("float32",    _fmt_bytes(fp32_bytes))
        t.add_row("ratio",      Text(f"{ratio:.0f}×", style="green bold"))
        t.add_row("shape",      f"{_fmt_count(n_vectors)} × {dim}")
        t.add_row("bit width",  f"{bit_width}-bit")
    else:
        t.add_row("", Text("waiting for first add…", style="dim"))
        if dim is not None:
            t.add_row("dim", str(dim))
        t.add_row("bit width", f"{bit_width}-bit")

    return Panel(t, title="[bold]Memory[/bold]", border_style="magenta")


def _header_text(
    snap: "Snapshot",
    n_vectors: int,
    dim: int | None,
    bit_width: int,
    reservoir_stored: int,
    reservoir_cap: int,
) -> Text:
    now = datetime.now().strftime("%H:%M:%S")
    dim_str = str(dim) if dim is not None else "?"
    warn = "  [yellow bold]⚠ recalibrate[/yellow bold]" if snap.recalibrate_suggested else ""
    t = Text(justify="center")
    t.append("turboscope", style="bold cyan")
    t.append(
        f"  ·  dim={dim_str}  ·  {bit_width}-bit"
        f"  ·  {_fmt_count(n_vectors)} vectors"
        f"  ·  reservoir {reservoir_stored}/{reservoir_cap}"
        f"  ·  {now}"
    )
    if snap.recalibrate_suggested:
        t.append("  ⚠ recalibrate", style="yellow bold")
    return t


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class Dashboard:
    """
    Live terminal dashboard for a :class:`~turboscope.index.TurboScopeIndex`.

    Parameters
    ----------
    index : TurboScopeIndex
        The index to observe.
    refresh_interval : float
        Seconds between display refreshes. Defaults to 1.0.
    """

    def __init__(
        self,
        index: "TurboScopeIndex",
        refresh_interval: float = _REFRESH_DEFAULT,
    ) -> None:
        self._index = index
        self._interval = refresh_interval
        self._p99_history: deque[float] = deque(maxlen=_SPARK_HISTORY)
        self._console = Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        duration_s: float | None = None,
        json_mode: bool = False,
    ) -> None:
        """
        Start the dashboard. Blocks until Ctrl-C or ``duration_s`` elapses.

        Parameters
        ----------
        duration_s : float or None
            If given, stop after this many seconds. None means run until
            Ctrl-C.
        json_mode : bool
            If True, emit one JSON line per refresh to stdout instead of
            rendering the terminal UI. Useful for piping into Prometheus,
            Grafana, or log aggregators.
        """
        deadline = time.monotonic() + duration_s if duration_s else None

        if json_mode:
            self._run_json(deadline)
        else:
            self._run_live(deadline)

    def render(self, snap: "Snapshot") -> Layout:
        """
        Build and return the rich Layout for ``snap`` without displaying it.

        Useful for testing the rendering logic without a live terminal, or
        for embedding the layout inside a larger rich application.

        Parameters
        ----------
        snap : Snapshot
            The snapshot to render.

        Returns
        -------
        rich.layout.Layout
        """
        if snap.p99_ms >= 0:
            self._p99_history.append(snap.p99_ms)
        return self._build_layout(snap)

    def print_once(self) -> None:
        """
        Take one snapshot and print the dashboard to the terminal. Does not
        block or refresh. Useful in notebooks and one-shot scripts.
        """
        snap = self._get_snapshot()
        if snap.p99_ms >= 0:
            self._p99_history.append(snap.p99_ms)
        layout = self._build_layout(snap)
        self._console.print(layout)

    # ------------------------------------------------------------------
    # Internal — live rendering loop
    # ------------------------------------------------------------------

    def _run_live(self, deadline: float | None) -> None:
        snap = self._get_snapshot()
        layout = self.render(snap)

        with Live(
            layout,
            console=self._console,
            screen=True,
            refresh_per_second=1.0 / self._interval,
        ) as live:
            try:
                while True:
                    if deadline and time.monotonic() >= deadline:
                        break
                    time.sleep(self._interval)
                    snap = self._get_snapshot()
                    layout = self.render(snap)
                    live.update(layout)
            except KeyboardInterrupt:
                pass

    def _run_json(self, deadline: float | None) -> None:
        try:
            while True:
                if deadline and time.monotonic() >= deadline:
                    break
                snap = self._get_snapshot()
                d = snap.to_dict()
                d["timestamp"] = datetime.utcnow().isoformat() + "Z"
                print(json.dumps(d), flush=True)
                time.sleep(self._interval)
        except KeyboardInterrupt:
            pass

    # ------------------------------------------------------------------
    # Internal — snapshot and layout building
    # ------------------------------------------------------------------

    def _get_snapshot(self) -> "Snapshot":
        """Pull a combined snapshot from the index's stats, sampler, drift."""
        return self._index.stats.snapshot(
            sampler=self._index._get_sampler(),
            drift=self._index._get_drift(),
        )

    def _build_layout(self, snap: "Snapshot") -> Layout:
        idx = self._index
        n_vectors = len(idx)
        dim = idx.dim
        bw = idx.bit_width
        res = idx.reservoir
        res_stored = res.n_stored
        res_cap = res._capacity

        header_text = _header_text(snap, n_vectors, dim, bw, res_stored, res_cap)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="panels"),
        )
        layout["panels"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        layout["left"].split_column(
            Layout(name="recall", ratio=1),
            Layout(name="latency", ratio=1),
        )
        layout["right"].split_column(
            Layout(name="drift",  ratio=1),
            Layout(name="memory", ratio=1),
        )

        layout["header"].update(
            Panel(Align(header_text, align="center"), border_style="dim")
        )
        layout["recall"].update(_recall_panel(snap))
        layout["drift"].update(_drift_panel(snap))
        layout["latency"].update(
            _latency_throughput_panel(snap, list(self._p99_history))
        )
        layout["memory"].update(
            _memory_panel(snap, n_vectors, dim, bw)
        )

        return layout


# ---------------------------------------------------------------------------
# One-shot convenience
# ---------------------------------------------------------------------------

def print_snapshot(index: "TurboScopeIndex") -> None:
    """
    Take one snapshot and print the dashboard to the terminal.

    Equivalent to ``Dashboard(index).print_once()``.

    Example
    -------
    ::

        from turboscope.dashboard import print_snapshot
        print_snapshot(index)
    """
    Dashboard(index).print_once()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    """
    python -m turboscope.dashboard --index my_index.tvim [--json] [--interval 2.0]

    Loads a .tvim file, wraps it in TurboScopeIndex, and runs the dashboard.
    Searches are not run here — the dashboard shows the static state of the
    loaded index (memory, dim, bit_width) plus any signals that accumulate
    as you continue to search from another process.

    For live signals, run your application with TurboScopeIndex and call
    dashboard.run() in a background thread, or point a scraper at the JSON
    output from --json mode.
    """
    import argparse
    from turboscope.index import TurboScopeIndex
    from turboscope.stats import attach_snapshot_method
    attach_snapshot_method(TurboScopeIndex)

    parser = argparse.ArgumentParser(
        prog="python -m turboscope.dashboard",
        description="turboscope terminal dashboard",
    )
    parser.add_argument("--index",    required=True, help=".tvim index file to load")
    parser.add_argument("--json",     action="store_true", help="emit JSON lines instead of UI")
    parser.add_argument("--interval", type=float, default=1.0, help="refresh interval in seconds")
    parser.add_argument("--duration", type=float, default=None, help="exit after N seconds")
    args = parser.parse_args()

    index = TurboScopeIndex.load(args.index)
    dash  = Dashboard(index, refresh_interval=args.interval)
    dash.run(duration_s=args.duration, json_mode=args.json)


if __name__ == "__main__":
    _cli()