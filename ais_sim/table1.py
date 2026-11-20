"""Representative cases at K = 0.20: the table and the 3x3 figure of time histories.

    python3 -m ais_sim.table1 [--workers N]

Table rows (tau = 2 s unless varied):
  zeta in {0, 0.5, 0.707, 1.0}            at wn = pi, tau = 2.5 s (phi = 2.5 pi)
  wn   in {3, 30, 300, 3000} * pi          at zeta = 0.707
  tau  in {0.3, 2, tau_f, 15} s           at wn = 3 pi, zeta = 0.05

At zeta = 0.05 the oscillation after the K step is still present at tau, so the
schedule depends on tau. tau_f solves tau + 2 T(tau) = t_f = 6.667 s (the
hardware pulse length), with the shaper designed at the true parameters and
the root found by a scan and brentq.

Each row runs the online estimator on the K step response over [0, tau] (latest
valid estimate before the deadline, at most the first 20,000 samples of the
window), designs the shaper (numerical solver started from the undamped closed
form, disabled when non-oscillatory), and simulates the shaped and unshaped
references on the true plant.

Figure panels: row 1 zeta in {0, 0.5, 1.0} at wn = pi, tau = 2.5 s. Row 2
wn in {3, 30, 300} pi at zeta = 0.707, tau = 2 s. Row 3 tau in {0.3, 2, tau_f} s
at wn = 3 pi, zeta = 0.05, with t_f marked on the last panel. If tau = 0.3 s
does not identify the plant, 0.35 and then 0.40 s are tried and the choice is
printed.

Writes ais_sim/out/table1_<stamp>.csv (the table), table1_fig2_<stamp>.pdf/.png
(the figure) and table1_fig2_<stamp>_cases.csv (its nine panels).
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
TAU = 2.0
# At wn = pi, tau = 2.5 s gives phi = 2.5 pi, away from the half period branch phi = m pi.
TAU_ROW1 = 2.5
T_F = 1000.0 / 150.0  # hardware pulse length: 1000 mm at 150 mm/s
TAU_SHORT_CANDIDATES = (0.3, 0.35, 0.40)
ZETA_TAU_ROW = 0.05
UNSHAPED = dict(color='#b5b4ae', lw=0.8, alpha=0.5)
PI = math.pi
TAU_LONG = 15.0


def table_rows(tau_short, tau_f):
    """Table cases. The tau rows use the figure's windows, so every panel is a table row."""
    return ([(PI, z, TAU_ROW1) for z in (0.0, 0.5, 0.707, 1.0)]
            + [(f * PI, 0.707, TAU) for f in (3, 30, 300, 3000)]
            + [(3 * PI, ZETA_TAU_ROW, tau) for tau in (tau_short, TAU, tau_f, TAU_LONG)])
OUT = Path(__file__).resolve().parent / 'out'


def run_row(spec):
    omega_n, zeta, tau = spec
    from ais_sim.sim import (Plant, ais_from_estimate, identify, residual_metrics,
                             unshaped_impulses)
    plant = Plant(omega_n, zeta)
    est = identify(plant, K, tau)
    impulses, solution, mode = ais_from_estimate(est, K, tau)
    shaped = residual_metrics(plant, impulses)
    unshaped = residual_metrics(plant, unshaped_impulses())
    feasible = solution is not None and solution.feasible
    return dict(
        omega_n=omega_n, omega_n_over_pi=omega_n / PI, zeta=zeta, tau=tau,
        zeta_hat=est.zeta if est is not None else math.nan,
        omega_hat=est.omega_n if est is not None else math.nan,
        omega_err_pct=100.0 * (est.omega_n / omega_n - 1.0) if est is not None else math.nan,
        zeta_err=(est.zeta - zeta) if est is not None else math.nan,
        mode=mode,
        solver=('disabled' if solution is None else solution.method),
        A=solution.A if feasible else math.nan,
        B=solution.B if feasible else math.nan,
        T=solution.T if feasible else math.nan,
        t_end=max(t for t, _ in impulses),
        A_res=shaped['A_res'], overshoot=shaped['overshoot'], e_bar=shaped['e_bar'],
        A_res_unshaped=unshaped['A_res'], e_bar_unshaped=unshaped['e_bar'],
        nfev=solution.nfev if feasible else 0,
        solve_time_ms=1e3 * solution.wall_time_s if solution is not None else math.nan,
        estimate_time_s=est.wall_time_s if est is not None else math.nan,
        impulses=';'.join(f'{t:.9g}:{a:.9g}' for t, a in impulses),
    )


def solve_tau_f(omega_n=3 * PI, zeta=ZETA_TAU_ROW, t_f=T_F, step=0.005):
    """Largest tau with tau + 2 T(tau) = t_f (scan for the sign change, then brentq)."""
    from scipy.optimize import brentq
    from ais_sim.shaper import design_shaper

    def gap(tau):
        return tau + 2.0 * design_shaper(K, tau, omega_n, zeta).T - t_f
    taus = np.arange(0.3, t_f, step)
    values = np.array([gap(tau) for tau in taus])
    changes = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
    if changes.size == 0:
        raise RuntimeError('tau + 2T never reaches t_f')
    i = changes[-1]
    return brentq(gap, taus[i], taus[i + 1], xtol=1e-12), changes.size


def fig2_panels(tau_short, tau_f):
    """(spec, x range in s, inset, title) for each panel."""
    return [
        ((PI, 0.0, TAU_ROW1), 6.0, False, r'$\zeta$ = 0.0'),
        ((PI, 0.5, TAU_ROW1), 6.0, False, r'$\zeta$ = 0.5'),
        ((PI, 1.0, TAU_ROW1), 6.0, False, r'$\zeta$ = 1.0'),
        ((3 * PI, 0.707, TAU), 3.5, False, r'$\omega_n$ = 3$\pi$'),
        ((30 * PI, 0.707, TAU), 3.5, True, r'$\omega_n$ = 30$\pi$'),
        ((300 * PI, 0.707, TAU), 3.5, True, r'$\omega_n$ = 300$\pi$'),
        ((3 * PI, ZETA_TAU_ROW, tau_short), 2.0, False, rf'$\tau$ = {tau_short:g} s'),
        ((3 * PI, ZETA_TAU_ROW, TAU), 3.5, False, r'$\tau$ = 2 s'),
        ((3 * PI, ZETA_TAU_ROW, tau_f), 8.0, False, rf'$\tau$ = $\tau_f$ = {tau_f:.3f} s'),
    ]


def identified(row):
    """The online estimate exists, is oscillatory and has the frequency right."""
    return row['mode'] == 'oscillatory' and abs(row['omega_err_pct']) < 0.1


def plot(path, panels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from ais_sim.sim import Plant, response, unshaped_impulses
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 7.2))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.11, wspace=0.30, hspace=0.62)
    for i, (ax, (row, xmax, has_inset, title)) in enumerate(zip(axes.ravel(), panels)):
        plant = Plant(row['omega_n'], row['zeta'])
        impulses = [tuple(float(v) for v in item.split(':'))
                    for item in row['impulses'].split(';')]
        t = np.linspace(0.0, xmax, 20000)
        x_unshaped, _ = response(plant, unshaped_impulses(), t)
        x_shaped, _ = response(plant, impulses, t)
        reference = np.zeros_like(t)
        for t0, a in impulses:
            reference += a * (t >= t0)

        def draw(target):
            target.axvspan(0.0, row['tau'], color='lightblue', alpha=0.35, lw=0)
            target.plot(t, x_unshaped, label='unshaped', **UNSHAPED)
            target.plot(t, reference, color='red', lw=1.0, label=r'$x^*_{\mathrm{TDF}}$')
            target.plot(t, x_shaped, color='blue', lw=1.0, label=r'$x$')

        draw(ax)
        ax.set_xlim(0.0, xmax)
        # Room for the gray unshaped trace where it overshoots strongly.
        tall = i in (0, 6, 7, 8)
        ymax = 2.05 if tall else 1.25
        ax.set_ylim(0.0, ymax)
        ax.set_yticks(np.arange(0.0, ymax, 0.4 if tall else 0.2))
        ax.grid(True, color='#b0b0b0', lw=0.6)
        ax.set_xlabel(r'$t$ [s]', labelpad=1)
        ax.set_ylabel(r'$x$ [m]', labelpad=1)
        ax.set_title(title, fontsize=8)
        ax.text(0.5, -0.34, f'({chr(ord("a") + i)})', transform=ax.transAxes, ha='center',
                va='top', fontsize=10, family='serif')
        if row['tau'] < T_F < xmax:
            ax.axvline(T_F, color='black', lw=0.9, ls=':')
            ax.text(T_F, 0.93 * ymax, r' $t_f$', fontsize=8, va='center')
        if has_inset:
            period = 2 * PI / row['omega_n']
            t1 = row['tau'] + row['T']
            # Left-centre, spanning y in [0.35, 0.80] of the 0-1.25 y-range: clear of
            # the 1.0 line and of the gray unshaped rise at the left edge.
            y0, y1 = 0.35 / 1.25, 0.80 / 1.25
            inset = ax.inset_axes([0.12, y0, 0.36, y1 - y0])
            draw(inset)
            inset.set_xlim(t1 - 0.6 * period, row['t_end'] + 0.9 * period)
            inset.set_ylim(0.86, 1.02)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.grid(True, color='#b0b0b0', lw=0.4)
            mark_inset(ax, inset, loc1=1, loc2=4, fc='none', ec='black', lw=0.7)

    # Legend swatches thicker than the plotted lines.
    handles = [Line2D([], [], color='red', lw=2.0), Line2D([], [], color='blue', lw=2.0),
               Line2D([], [], color=UNSHAPED['color'], lw=2.0, alpha=UNSHAPED['alpha']),
               Patch(color='lightblue', alpha=0.35)]
    labels = [r'$x^*_{\mathrm{TDF}}$', r'$x$', 'unshaped', 'Estimation Phase']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.985),
               ncol=len(labels), frameon=True, fontsize=8)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Panel (g) and the first tau row: the first short window that identifies.
    tau_short, attempts = None, []
    for candidate in TAU_SHORT_CANDIDATES:
        trial = run_row((3 * PI, ZETA_TAU_ROW, candidate))
        attempts.append(trial)
        if identified(trial):
            tau_short = candidate
            break
    if tau_short is None:
        raise SystemExit('no short window in TAU_SHORT_CANDIDATES identified the plant')
    tau_f, n_crossings = solve_tau_f()
    known = {(r['omega_n'], r['zeta'], r['tau']): r for r in attempts}
    table_specs = table_rows(tau_short, tau_f)
    specs = table_specs + [spec for spec, *_ in fig2_panels(tau_short, tau_f)]
    missing = list(dict.fromkeys(spec for spec in specs if spec not in known))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for spec, row in zip(missing, pool.map(run_row, missing)):
            known[spec] = row
    rows = [known[spec] for spec in table_specs]
    path = OUT / f'table1_{stamp}'
    with path.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    panels = [(known[spec], xmax, inset, title)
              for spec, xmax, inset, title in fig2_panels(tau_short, tau_f)]
    with (OUT / f'table1_fig2_{stamp}_cases.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['panel'] + list(rows[0]))
        writer.writeheader()
        for i, (row, *_) in enumerate(panels):
            writer.writerow(dict(panel=chr(ord('a') + i), **row))
    plot(OUT / f'table1_fig2_{stamp}', panels)
    print('Panel (g) short window:')
    for trial in attempts:
        print(f"  tau = {trial['tau']:g} s: mode {trial['mode']}, wn^ = {trial['omega_hat']:.6f} "
              f"(err {trial['omega_err_pct']:+.4f} %), zeta^ = {trial['zeta_hat']:.5f}, "
              f"A_res {trial['A_res']:.2e} -> {'used' if identified(trial) else 'rejected'}")
    print(f'tau_f = {tau_f:.6f} s (tau + 2T = t_f = {T_F:.4f} s; {n_crossings} crossing(s) in '
          f'[0.3, t_f])')
    print('Figure panels:')
    for i, (row, *_) in enumerate(panels):
        print(f"  ({chr(ord('a') + i)}) wn/pi={row['omega_n_over_pi']:g} zeta={row['zeta']:g} "
              f"tau={row['tau']:g}: mode {row['mode']}, A={row['A']:.4f}, T={row['T']:.4f}, "
              f"t_end={row['t_end']:.4f}, A_res={row['A_res']:.2e}")
    print('  zeta  zeta^     wn/pi  tau   wn^           err%      mode             solver       '
          'A       T       A_res    overshoot  e_bar    nfev  solve_ms')
    for r in rows:
        print(f"  {r['zeta']:<5g} {r['zeta_hat']:<8.5f}  {r['omega_n_over_pi']:<5g} "
              f"{r['tau']:<5g} {r['omega_hat']:<12.6f}  {r['omega_err_pct']:+.4f}  "
              f"{r['mode']:15s}  {r['solver']:11s}  {r['A']:.4f}  {r['T']:.4f}  "
              f"{r['A_res']:.2e} {r['overshoot']:.2e}   {r['e_bar']:.2e}  {r['nfev']:4d}  "
              f"{r['solve_time_ms']:.1f}")
    print(f'\nWrote {path}.csv and table1_fig2_{stamp}.pdf/.png/_cases.csv')


if __name__ == '__main__':
    main()
