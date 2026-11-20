"""Shaper solver checks: closed form against brentq, and damped solver statistics.

    python3 -m ais_sim.solver_report

Writes to ais_sim/out/:
  solver_closed_vs_brentq_<stamp>.csv   one row per (K, phi)
  solver_closed_vs_brentq_<stamp>.png   feasibility and three real root map
  solver_damped_<stamp>.csv             one row per damped design
  solver_damped_summary_<stamp>.csv     the damped solver table
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from ais_sim.shaper import (
    cubic_discriminant_sign,
    solve_damped,
    solve_undamped,
    solve_undamped_brentq,
    solver_statistics,
)

OUT = Path(__file__).resolve().parent / 'out'

# Damped grid at K = 0.20: the windows of ais_sim.table1. 5.9996 s is tau_f, where
# tau + 2T = t_f at wn = 3 pi, zeta = 0.05.
DAMPED_ZETAS = (0.05, 0.3, 0.5, 0.707, 0.8, 0.9)
DAMPED_OMEGAS = (math.pi, 3 * math.pi, 30 * math.pi, 300 * math.pi, 3000 * math.pi)
DAMPED_TAUS = (0.3, 2.0, 2.5, 5.9996, 15.0)


def closed_vs_brentq(stamp):
    ks = np.linspace(0.02, 0.98, 49)
    phis = np.linspace(0.0, 8.0 * math.pi, 401)
    rows = []
    for K in ks:
        for phi in phis:
            closed = solve_undamped(K, phi, 1.0)
            bracketed = solve_undamped_brentq(K, phi, 1.0)
            delta = (abs(closed.T - bracketed.T)
                     if closed.feasible and bracketed.feasible else math.nan)
            rows.append(dict(
                K=K, phi=phi, three_real_roots=int(cubic_discriminant_sign(K, phi) < 0.0),
                closed_feasible=int(closed.feasible), brentq_feasible=int(bracketed.feasible),
                closed_method=closed.method, T_closed=closed.T, T_brentq=bracketed.T,
                abs_dT=delta, A=closed.A, residual=closed.residual))
    path = OUT / f'solver_closed_vs_brentq_{stamp}.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grid = np.array([[r['closed_feasible'] + 2 * r['three_real_roots'] for r in rows]],
                    dtype=float).reshape(ks.size, phis.size)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    image = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis',
                      extent=(phis[0] / math.pi, phis[-1] / math.pi, ks[0], ks[-1]))
    ax.set_xlabel(r'$\phi = \omega_n\tau$ / $\pi$')
    ax.set_ylabel('K')
    ax.set_title('0: infeasible, 1: feasible, 2/3: three real cubic roots (+feasible)')
    fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3])
    fig.savefig(path.with_suffix('.png'), dpi=150)
    plt.close(fig)

    deltas = np.array([r['abs_dT'] for r in rows], dtype=float)
    mismatched = sum(r['closed_feasible'] != r['brentq_feasible'] for r in rows)
    return dict(
        points=len(rows),
        three_real_fraction=float(np.mean([r['three_real_roots'] for r in rows])),
        feasible_fraction=float(np.mean([r['closed_feasible'] for r in rows])),
        feasibility_mismatches=int(mismatched),
        max_abs_dT=float(np.nanmax(deltas)),
        csv=path)


def damped_stats(stamp):
    solutions, rows = [], []
    for zeta in DAMPED_ZETAS:
        for omega_n in DAMPED_OMEGAS:
            for tau in DAMPED_TAUS:
                solution = solve_damped(0.2, tau, omega_n, zeta)
                solutions.append(solution)
                rows.append(dict(
                    zeta=zeta, omega_n=omega_n, tau=tau, feasible=int(solution.feasible),
                    A=solution.A, T=solution.T, residual=solution.residual,
                    nfev=solution.nfev, nfev_total=solution.nfev_total,
                    n_distinct=solution.n_distinct, wall_time_s=solution.wall_time_s))
    path = OUT / f'solver_damped_{stamp}.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stats = solver_statistics(solutions)
    summary = OUT / f'solver_damped_summary_{stamp}.csv'
    with summary.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['quantity', 'value'])
        writer.writerow(['success rate', f"{stats['success_rate']:.4f}"])
        writer.writerow(['median function evaluations (accepted run)',
                         f"{stats['median_nfev']:.0f}"])
        writer.writerow(['median wall time per design (all 8-9 starts) [ms]',
                         f"{1e3 * stats['median_time_s']:.2f}"])
        writer.writerow(['max normalized residual', f"{stats['max_residual']:.2e}"])
    stats['csv'] = path
    stats['summary_csv'] = summary
    return stats


def main():
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    closed = closed_vs_brentq(stamp)
    damped = damped_stats(stamp)
    print('Closed form vs brentq (omega_n = 1, T = theta):')
    print(f"  {closed['points']} (K, phi) points, "
          f"three-real-root fraction {closed['three_real_fraction']:.3f}, "
          f"feasible fraction {closed['feasible_fraction']:.3f}")
    print(f"  feasibility mismatches {closed['feasibility_mismatches']}, "
          f"max |T_closed - T_brentq| = {closed['max_abs_dT']:.2e}")
    print(f"  {closed['csv']}")
    print(f"Damped solver over {damped['runs']} designs (K = 0.20):")
    print(f"  success {damped['success_rate']:.1%}, median nfev {damped['median_nfev']:.0f}, "
          f"median time {1e3 * damped['median_time_s']:.2f} ms, "
          f"max residual {damped['max_residual']:.1e}, "
          f"max distinct solutions {damped['max_distinct']}")
    print(f"  {damped['summary_csv']}")


if __name__ == '__main__':
    main()
