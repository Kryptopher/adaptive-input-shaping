"""One-parameter sweeps of the simulated AIS.

    python3 -m ais_sim.sweeps [--workers N] [--only zeta|omega_n|tau]

Base point wn = 3 pi, zeta = 0.707, tau = 2 s, K = 0.20. One parameter varies at
a time. The tau sweep uses zeta = 0.05, because at 0.707 the oscillation after
the K step has decayed before tau and the schedule would not depend on tau.

  zeta  0 .. 1.0, 101 points
  wn    pi .. 3000 pi, 61 log-spaced points
  tau   0.5 .. 20 s, 79 points (the maneuver end time is compared with the
        hardware pulse length t_f = 6.667 s)

Identification uses the light online setup (coarse grids in the window, then
the full estimator once on the selected window). Per run: zeta_hat, wn error,
mode, peak residual A_res, overshoot and the maneuver end time (tau + 2T, or
tau when the shaper is disabled).

Writes ais_sim/out/sweeps_<stamp>.csv and sweep_{zeta,omega_n,tau}_<stamp>.pdf/.png
(only the selected sweep with --only).
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
BASE = dict(omega_n=3.0 * math.pi, zeta=0.707, tau=2.0)
SWEEP_BASE = dict(zeta=BASE, omega_n=BASE, tau=dict(BASE, zeta=0.05))
T_F = 1000.0 / 150.0  # hardware pulse length: 1000 mm at 150 mm/s
OUT = Path(__file__).resolve().parent / 'out'
SWEEPS = {
    'zeta': np.round(np.linspace(0.0, 1.0, 101), 4),
    'omega_n': np.geomspace(math.pi, 3000.0 * math.pi, 61),
    'tau': np.round(np.linspace(0.5, 20.0, 79), 4),
}
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#52514e'


def run_point(args):
    name, value = args
    from ais_sim.sim import Plant, ais_from_estimate, identify, residual_metrics
    params = dict(SWEEP_BASE[name], **{name: float(value)})
    plant = Plant(params['omega_n'], params['zeta'])
    tau = params['tau']
    est = identify(plant, K, tau, light=True)
    impulses, solution, mode = ais_from_estimate(est, K, tau)
    metrics = residual_metrics(plant, impulses)
    return dict(sweep=name, value=float(value), **params,
                zeta_hat=est.zeta if est is not None else math.nan,
                omega_hat=est.omega_n if est is not None else math.nan,
                omega_err_pct=(100.0 * (est.omega_n / params['omega_n'] - 1.0)
                               if est is not None else math.nan),
                mode=mode, A_res=metrics['A_res'], overshoot=metrics['overshoot'],
                e_bar=metrics['e_bar'], t_end=metrics['t_last'],
                T=solution.T if solution is not None and solution.feasible else math.nan)


def _residual_panel(ax, x, rows, disabled):
    """Peak residual, overshoot and the unshaped step's residual, log y."""
    from ais_sim.sim import Plant, residual_metrics, unshaped_impulses
    # A_res is not drawn where the shaper is disabled (it equals 1 - K there).
    a_res = np.where(disabled, np.nan, np.maximum([r['A_res'] for r in rows], 1e-16))
    ax.semilogy(x, a_res, color=BLUE, lw=1.6, label=r'peak residual $A_{res}$')
    ax.semilogy(x, np.maximum([r['overshoot'] for r in rows], 1e-16), color=ORANGE, lw=1.2,
                label='overshoot')
    # Unshaped reference: the overshoot of the plain unit step. Its peak |x - 1|
    # from t = 0 is always 1.
    unshaped = [residual_metrics(Plant(r['omega_n'], r['zeta']), unshaped_impulses())['overshoot']
                for r in rows]
    ax.semilogy(x, np.maximum(unshaped, 1e-16), color='#b5b4ae', lw=1.0, label='unshaped')
    ax.axhline(0.01, color=GRAY, lw=0.8, ls=':')
    ax.set_ylabel('fraction of step')
    ax.set_ylim(1e-16, 2.0)
    ax.legend(frameon=False, fontsize=7, loc='center right')


def _end_time_panel(ax, x, rows, name):
    """Maneuver end time. The tau sweep adds the line t = tau and the hardware t_f."""
    ax.plot(x, [r['t_end'] for r in rows], color=BLUE, lw=1.6, label='maneuver end')
    if name == 'tau':
        ax.plot(x, x, color=GRAY, lw=0.8, ls='--', label=r'$\tau$')
        ax.axhline(T_F, color=ORANGE, lw=0.9, ls=':', label=rf'$t_f$ = {T_F:.3f} s (hardware)')
        ax.legend(frameon=False, fontsize=7)
    ax.set_ylabel('maneuver end time (s)')


def plot_sweep(rows, name, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    rows = sorted(rows, key=lambda r: r['value'])
    x = np.array([r['value'] for r in rows])
    if name == 'omega_n':
        x = x / math.pi
    xlabel = {'zeta': r'true damping ratio $\zeta$', 'omega_n': r'$\omega_n$ / $\pi$',
              'tau': r'estimation window $\tau$ (s)'}[name]
    disabled = np.array([r['mode'] != 'oscillatory' for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True, sharex=True)
    ax_z, ax_w, ax_r, ax_t = axes.ravel()
    zeta_hat = np.array([r['zeta_hat'] for r in rows])
    ax_z.plot(x, zeta_hat, color=BLUE, lw=1.6, label=r'$\hat\zeta$')
    if name == 'zeta':
        ax_z.plot(x, x, color=GRAY, lw=0.8, ls='--', label=r'true $\zeta$')
        ax_z.legend(frameon=False, fontsize=8, loc='lower right')
    else:
        base_zeta = SWEEP_BASE[name]['zeta']
        ax_z.axhline(base_zeta, color=GRAY, lw=0.8, ls='--')
        ax_z.set_ylim(max(0.0, base_zeta - 0.05), base_zeta + 0.05)
    ax_z.set_ylabel(r'$\hat\zeta$')

    err = np.array([r['omega_err_pct'] for r in rows], dtype=float)
    ax_w.plot(x, err, color=BLUE, lw=1.6)
    ax_w.axhline(0.0, color=GRAY, lw=0.8, ls='--')
    span = max(0.1, 1.1 * float(np.nanmax(np.abs(err))))
    ax_w.set_ylim(-span, span)
    ax_w.set_ylabel(r'$\hat\omega_n$ error (%)')
    for ax in (ax_z, ax_w):
        ax.ticklabel_format(axis='y', useOffset=False, style='plain')

    _residual_panel(ax_r, x, rows, disabled)
    _end_time_panel(ax_t, x, rows, name)

    for ax in axes.ravel():
        if name == 'omega_n':
            ax.set_xscale('log')
        if disabled.any():
            lo = x[disabled].min()
            ax.axvspan(lo, x.max(), color='#efeee9', lw=0, zorder=0)
        ax.grid(True, color='#e6e5e1', lw=0.6)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--only', choices=list(SWEEPS), help='run a single sweep')
    args = parser.parse_args()
    names = [args.only] if args.only else list(SWEEPS)
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cases = [(name, v) for name in names for v in SWEEPS[name]]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run_point, cases), 1):
            rows.append(row)
            if i % 40 == 0 or i == len(cases):
                print(f'[{i}/{len(cases)}]', flush=True)
    path = OUT / f'sweeps_{stamp}'
    with path.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in names:
        plot_sweep([r for r in rows if r['sweep'] == name], name,
                   OUT / f'sweep_{name}_{stamp}')
    for name in names:
        sub = [r for r in rows if r['sweep'] == name]
        active = [r for r in sub if r['mode'] == 'oscillatory']
        print(f"{name}: {len(active)}/{len(sub)} shaped; max |wn err| "
              f"{max(abs(r['omega_err_pct']) for r in active):.4f}%; max |zeta err| "
              f"{max(abs(r['zeta_hat'] - r['zeta']) for r in active):.4f}; max A_res "
              f"{max(r['A_res'] for r in active):.1e}; max overshoot (all) "
              f"{max(r['overshoot'] for r in sub):.1e}")
    if 'tau' in names:
        tau_rows = sorted((r for r in rows if r['sweep'] == 'tau'), key=lambda r: r['value'])
        crossing = next((r['value'] for r in tau_rows if r['t_end'] >= T_F), math.nan)
        print(f'tau sweep: maneuver end first reaches t_f = {T_F:.3f} s at tau = {crossing:g} s')
    print(f'\nWrote {path}.csv and sweep_*_{stamp}.pdf/.png')


if __name__ == '__main__':
    main()
