"""FT shaper as published against the AIS, both designed from the same online estimate.

    python3 -m ais_sim.ft_vs_ais_paper [--workers N]
    python3 -m ais_sim.ft_vs_ais_paper --replot SWEEP_CSV --history HISTORY_CSV

FT (Stein and Aruco, 2024) switches at T_FT and 2 T_FT with T_FT = pi/wd. A
switch whose time has passed when the estimate arrives at tau fires at tau
(``ft_impulses`` variant 'a'). The AIS switches at tau + T and tau + 2T.
wn = 3 pi, zeta = 0, K = 0.20.

  (a) shaped reference x*_TDF and response of both methods at tau = 1.5 T_FT,
      time in s, estimation interval shaded, first switches dotted: FT's
      design time T_FT (orange) and the AIS's tau + T (blue)
  (b) peak residual A_res against tau / T_FT in 0.2 .. 4, with tau = T_FT marked

A full run writes ais_sim/out/ft_vs_ais_paper_<stamp>.csv (the sweep),
_history.csv (the time histories of panel a) and .pdf/.png. ``--replot`` draws
the figure from a saved sweep CSV, and ``--history`` adds saved time histories,
so nothing is simulated.
"""

from __future__ import annotations

import os

# One BLAS thread per worker process. This must be set before numpy is imported,
# or each worker starts its own thread pool and the cores are oversubscribed.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

K = 0.20
OMEGA_N = 3.0 * math.pi
ZETA = 0.0
TAU_RATIO_HISTORY = 1.5
RATIOS = np.round(np.arange(0.2, 4.0 + 1e-9, 0.1), 3)
OUT = Path(__file__).resolve().parent / 'out'
COLORS = {'FT': '#eb6834', 'AIS': '#2a78d6'}


def designs(tau):
    """Online estimate on [0, tau], then the FT and AIS impulse trains from it."""
    from ais_sim.sim import Plant, ais_from_estimate, disabled_impulses, ft_impulses, identify
    plant = Plant(OMEGA_N, ZETA)
    est = identify(plant, K, tau)
    ais, _, mode = ais_from_estimate(est, K, tau)
    if est is not None and est.valid and mode == 'oscillatory':
        ft = ft_impulses(K, est.omega_n, est.zeta, tau, 'a')
    else:
        ft = disabled_impulses(K, tau)
    return plant, est, {'FT': ft, 'AIS': ais}


def run_case(tau_ratio):
    from ais_sim.sim import residual_metrics
    T_ft = math.pi / OMEGA_N
    tau = tau_ratio * T_ft
    plant, est, impulses = designs(tau)
    row = dict(tau_ratio=tau_ratio, tau=tau, T_FT=T_ft,
               est_mode=est.mode if est is not None else 'no-estimate',
               est_omega_n=est.omega_n if est is not None else math.nan)
    for name, train in impulses.items():
        metrics = residual_metrics(plant, train)
        row[f'{name}_A_res'] = metrics['A_res']
        row[f'{name}_t_maneuver'] = metrics['t_last']
    return row


def history_traces():
    """Reference and response at tau = TAU_RATIO_HISTORY * T_FT (noise-free, deterministic)."""
    from ais_sim.sim import response
    T_ft = math.pi / OMEGA_N
    tau = TAU_RATIO_HISTORY * T_ft
    plant, _, impulses = designs(tau)
    t_end = max(max(t for t, _ in train) for train in impulses.values()) + 4.0 * T_ft
    t = np.linspace(0.0, t_end, 4000)
    curves = {}
    for name, train in impulses.items():
        r = sum(a * (t >= ti) for ti, a in train)
        curves[name] = (r, response(plant, train, t)[0])
    return t, tau, curves, impulses


def plot(rows, history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.145, top=0.77, wspace=0.27)

    t, tau, curves, impulses = history
    T_ft = math.pi / OMEGA_N
    shade = '#e6e5e1'
    ax_t.axvspan(0.0, tau, color=shade, lw=0, zorder=0)
    handles = {}
    for name in ('FT', 'AIS'):
        r, x = curves[name]
        handles[name, 'r'], = ax_t.step(t, r, where='post', color=COLORS[name], lw=1.0, ls='--',
                                        label=rf'{name} $x^*_{{\mathrm{{TDF}}}}$')
        handles[name, 'x'], = ax_t.plot(t, x, color=COLORS[name], lw=1.6, label=rf'{name} $\bar x$')
    # First switches: FT's design time T_FT (passed by tau, so it fired at tau) and AIS's tau + T.
    ft_switch = ax_t.axvline(T_ft, color=COLORS['FT'], lw=0.8, ls=':')
    ais_switch = ax_t.axvline(impulses['AIS'][1][0], color=COLORS['AIS'], lw=0.8, ls=':')
    ax_t.set_xlim(t[0], t[-1])
    ax_t.set_xlabel(r'$t$ [s]')
    ax_t.set_ylabel(r'$x$')
    ax_t.set_title('(a)', loc='left')
    # Three columns of three, filled column by column: shading | FT | AIS.
    blank = Line2D([], [], color='none')
    fig.legend([Patch(color=shade), blank, blank,
                 handles['FT', 'r'], handles['FT', 'x'], ft_switch,
                 handles['AIS', 'r'], handles['AIS', 'x'], ais_switch],
                ['estimation interval', '', '',
                 r'FT $x^*_{\mathrm{TDF}}$', r'FT $\bar{x}$', r'FT first switch $T_{\mathrm{FT}}$',
                 r'AIS $x^*_{\mathrm{TDF}}$', r'AIS $\bar{x}$', r'AIS first switch $\tau+T$'],
                ncol=3, frameon=False, fontsize=7, loc='upper left',
                bbox_to_anchor=(ax_t.get_position().x0 - 0.01, 1.0), borderaxespad=0.3,
                columnspacing=1.2, handlelength=1.8)
    ax_t.grid(True, color='#e6e5e1', lw=0.6)

    x_ratio = np.array([r['tau_ratio'] for r in rows])
    for name in ('FT', 'AIS'):
        res = np.array([r[f'{name}_A_res'] for r in rows])
        ax_r.plot(x_ratio, res, color=COLORS[name], lw=1.8, label=name)
    ax_r.axvline(1.0, color='#0b0b0b', lw=0.9, ls='--')
    ax_r.text(0.96, 0.6, r'$\tau=T_{\mathrm{FT}}$', fontsize=8, va='center', ha='right')
    ax_r.set_ylim(-0.03, 1.03)  # AIS's zero line clears the axis
    ax_r.set_xlim(0.2, 4.0)
    ax_r.set_xlabel(r'$\tau/T_{\mathrm{FT}}$')
    ax_r.set_ylabel(r'$A_{\mathrm{res}}$')
    ax_r.set_title('(b)', loc='left')
    ax_r.grid(True, color='#e6e5e1', lw=0.6)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def read_history(path):
    """Read a saved _history.csv. The impulse trains are the jumps of its reference staircases."""
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    t = data[:, 0]
    curves = {'FT': (data[:, 1], data[:, 2]), 'AIS': (data[:, 3], data[:, 4])}
    impulses = {}
    for name, (r, _) in curves.items():
        jumps = np.flatnonzero(np.abs(np.diff(r)) > 1e-12) + 1
        impulses[name] = [(0.0, float(r[0]))] + [(float(t[i]), float(r[i] - r[i - 1])) for i in jumps]
    return t, TAU_RATIO_HISTORY * math.pi / OMEGA_N, curves, impulses


def write_history(history, path):
    t, _, curves, _ = history
    with path.with_name(path.name + '_history.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['t', 'FT_r', 'FT_x', 'AIS_r', 'AIS_x'])
        writer.writerows(zip(t, *curves['FT'], *curves['AIS']))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--replot', metavar='CSV', help='draw the figure from a saved sweep CSV')
    parser.add_argument('--history', metavar='CSV',
                        help='saved time histories (_history.csv) for panel (a)')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT / f"ft_vs_ais_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    T_ft = math.pi / OMEGA_N
    history = read_history(args.history) if args.history else history_traces()
    impulses = history[3]
    if args.replot:
        with Path(args.replot).open() as handle:
            rows = [{k: (v if k == 'est_mode' else float(v)) for k, v in r.items()}
                    for r in csv.DictReader(handle)]
        plot(rows, history, path)
        write_history(history, path)
        print(f'Redrew {args.replot} as {path}.pdf/.png (+ _history.csv)')
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run_case, [float(r) for r in RATIOS]))
        with path.with_suffix('.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot(rows, history, path)
        write_history(history, path)

    print(f'tau = {TAU_RATIO_HISTORY} T_FT: ' + ', '.join(
        f"{name} impulses {[(round(ti / T_ft, 3), round(a, 4)) for ti, a in train]}"
        for name, train in impulses.items()))
    late = [r for r in rows if r['tau_ratio'] > 1.0 + 1e-9]
    ok = [r for r in rows if r['AIS_A_res'] < 1e-6]
    print(f"AIS cancels (A_res < 1e-6) from tau/T_FT = {min(r['tau_ratio'] for r in ok):g}; "
          f"max AIS A_res there {max(r['AIS_A_res'] for r in ok):.1e}")
    print(f"FT for tau > T_FT: A_res {min(r['FT_A_res'] for r in late):.3f}"
          f"-{max(r['FT_A_res'] for r in late):.3f}")
    if not args.replot:
        print(f'Wrote {path}.csv/.pdf/.png (+ _history.csv)')


if __name__ == '__main__':
    main()
