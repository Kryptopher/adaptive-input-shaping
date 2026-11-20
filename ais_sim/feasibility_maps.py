"""Feasibility of the AIS design equations.

    python3 -m ais_sim.feasibility_maps [--workers N]

(K, phi) map at zeta = 0, phi = wn*tau (the figure): feasible or infeasible
from the closed form (feasible for every phi when K < 1/2, for none when
K >= 1/2), feasible cells with three real cubic roots in a darker blue, and the
exact phi = m*pi points (the sin(phi) guard branch) as markers, since a regular
grid only hits them by accident. The dotted line marks the lower limit of the
three real root region, its cusp at K = 1/4, phi = 2 pi j (a triple root). The
lowest grid K with three real roots is 0.265, printed in the summary.

(zeta, phi) scan at K = 0.20 (CSV and printed summary). After scaling by
exp(-sigma*tau) the damped equations depend on wn and tau only through phi and
zeta, so one scan covers every plant. Classes:
  interior         feasible, B = 1-K-A >= B_DEGENERATE * (1-K)
  near-degenerate  feasible but B < B_DEGENERATE * (1-K)  (B -> 0 as zeta -> 1)
  none             no feasible (A, T) with 0 < A < 1-K, T > 0

Writes ais_sim/out/feasibility_k_phi_<stamp>.csv/.pdf/.png, _exact_m_pi.csv and
feasibility_zeta_phi_<stamp>.csv.
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

K_PAPER = 0.20
K_THREE_ROOT_CUSP = 0.25
B_DEGENERATE = 1.0e-3
PHI_MAX = 8.0 * math.pi
OUT = Path(__file__).resolve().parent / 'out'
FEASIBLE_COLORS = ['#d9d8d3', '#2a78d6', '#164f8c']  # infeasible, feasible one / three real roots


def _zeta_phi_row(zeta):
    from ais_sim.shaper import solve_damped, solve_undamped
    phis = np.linspace(0.0, PHI_MAX, 321)[1:]
    out = []
    for phi in phis:
        if zeta == 0.0:
            s = solve_undamped(K_PAPER, phi, 1.0)
        else:
            s = solve_damped(K_PAPER, phi, 1.0, zeta)
        if not s.feasible:
            cls = 0
        elif s.B < B_DEGENERATE * (1.0 - K_PAPER):
            cls = 1
        else:
            cls = 2
        out.append(dict(zeta=zeta, phi=phi, cls=cls, A=s.A, B=s.B, T=s.T))
    return out


def _k_phi_row(K):
    from ais_sim.shaper import cubic_discriminant_sign, solve_undamped
    phis = np.linspace(0.0, PHI_MAX, 801)
    phis = 0.5 * (phis[1:] + phis[:-1])  # cell centres: never exactly m*pi
    out = []
    for phi in phis:
        s = solve_undamped(K, phi, 1.0)
        out.append(dict(K=K, phi=phi, feasible=int(s.feasible),
                        three_real=int(cubic_discriminant_sign(K, phi) < 0.0), A=s.A, T=s.T))
    exact = []
    for m in range(1, int(round(PHI_MAX / math.pi)) + 1):
        s = solve_undamped(K, m * math.pi, 1.0)
        exact.append(dict(K=K, m=m, feasible=int(s.feasible), method=s.method))
    return out, exact


def _write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _style(plt):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False})


def scan_zeta_phi(pool, stamp):
    zetas = np.round(np.arange(0.0, 0.991, 0.01), 2)
    rows = [r for chunk in pool.map(_zeta_phi_row, zetas) for r in chunk]
    path = OUT / f'feasibility_zeta_phi_{stamp}'
    _write_csv(path.with_suffix('.csv'), rows)

    by_zeta = {}
    for r in rows:
        by_zeta.setdefault(r['zeta'], []).append(r['cls'])
    first_degenerate = next((z for z in zetas if 1 in by_zeta[z]), math.nan)
    all_interior_max = max((z for z in zetas if all(c == 2 for c in by_zeta[z])),
                           default=math.nan)
    return dict(path=path, first_degenerate_zeta=first_degenerate,
                all_interior_up_to_zeta=all_interior_max,
                interior_fraction_by_zeta={z: float(np.mean(np.array(by_zeta[z]) == 2))
                                           for z in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99)})


def map_k_phi(pool, stamp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ks = np.round(np.arange(0.01, 0.99 + 1e-9, 0.005), 4)
    results = list(pool.map(_k_phi_row, ks))
    rows = [r for chunk, _ in results for r in chunk]
    exact = [r for _, chunk in results for r in chunk]
    path = OUT / f'feasibility_k_phi_{stamp}'
    _write_csv(path.with_suffix('.csv'), rows)
    _write_csv(path.with_name(path.name + '_exact_m_pi.csv'), exact)

    phis = np.array(sorted({r['phi'] for r in rows}))
    feasible = np.array([r['feasible'] for r in rows], dtype=float).reshape(ks.size, phis.size)
    three = np.array([r['three_real'] for r in rows], dtype=float).reshape(ks.size, phis.size)
    k_all = max((k for k, row in zip(ks, feasible) if row.all()), default=math.nan)
    k_any = max((k for k, row in zip(ks, feasible) if row.any()), default=math.nan)

    _style(plt)
    fig, ax = plt.subplots(figsize=(7.6, 3.6), constrained_layout=True)
    region = feasible * (1.0 + three)  # 0 infeasible, 1 one real root, 2 three real roots
    ax.pcolormesh(phis / math.pi, ks, region, cmap=ListedColormap(FEASIBLE_COLORS),
                  vmin=-0.5, vmax=2.5, shading='nearest', rasterized=True)
    k_three = min(k for k, row in zip(ks, three) if row.any())
    ax.axhline(0.5, color='#0b0b0b', ls='-', lw=1.2)
    ax.axhline(K_THREE_ROOT_CUSP, color='#0b0b0b', ls=':', lw=0.8)
    ax.text(PHI_MAX / math.pi, 0.5, ' K = 1/2\n feasibility\n boundary', va='center',
            ha='left', fontsize=7.5, linespacing=1.1)
    ax.text(PHI_MAX / math.pi, K_THREE_ROOT_CUSP, ' K = 1/4\n three-real-root\n boundary',
            va='center', ha='left', fontsize=7.5, linespacing=1.1)
    sparse = [r for r in exact if abs(round(r['K'] / 0.05) * 0.05 - r['K']) < 1e-9
              and r['K'] <= 0.95]
    for feas, marker_face in ((1, '#0b0b0b'), (0, 'white')):
        pts = [(r['m'], r['K']) for r in sparse if r['feasible'] == feas]
        if pts:
            m, k = zip(*pts)
            ax.scatter(m, k, s=10, marker='o', facecolors=marker_face, edgecolors='#0b0b0b',
                       linewidths=0.5, zorder=3, clip_on=False)
    ax.set_xlabel(r'$\phi = \omega_n \tau$ / $\pi$')
    ax.set_ylabel('identification fraction K')
    ax.legend(handles=[Patch(color=FEASIBLE_COLORS[1], label='feasible, one real root'),
                       Patch(color=FEASIBLE_COLORS[2], label='feasible, three real roots'),
                       Patch(color=FEASIBLE_COLORS[0], label='infeasible'),
                       Line2D([], [], marker='o', ls='', markerfacecolor='#0b0b0b',
                              markeredgecolor='#0b0b0b', markersize=3,
                              label=r'exact $\phi = m\pi$, feasible (half-period branch)'),
                       Line2D([], [], marker='o', ls='', markerfacecolor='white',
                              markeredgecolor='#0b0b0b', markersize=3, label=r'exact $\phi = m\pi$, infeasible')],
              loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False,
              fontsize=8)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)

    exact_feasible_k = {m: max((r['K'] for r in exact if r['m'] == m and r['feasible']),
                               default=math.nan)
                        for m in sorted({r['m'] for r in exact})}
    return dict(path=path, k_all_phi_feasible=k_all, k_any_phi_feasible=k_any,
                k_three_real_min=k_three,
                exact_m_pi_max_feasible_k=exact_feasible_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        b = map_k_phi(pool, stamp)
        print(f"K-phi, zeta = 0: every phi feasible up to K = {b['k_all_phi_feasible']:.3f}; "
              f"no phi feasible above K = {b['k_any_phi_feasible']:.3f}; three real roots "
              f"from K = {b['k_three_real_min']:.3f}")
        print(f"    exact phi = m pi, largest feasible K per m: {b['exact_m_pi_max_feasible_k']}")
        print(f"    {b['path']}.png")
        a = scan_zeta_phi(pool, stamp)
        print(f"zeta-phi, K = {K_PAPER}: interior everywhere up to zeta = "
              f"{a['all_interior_up_to_zeta']}; first near-degenerate at zeta = "
              f"{a['first_degenerate_zeta']}")
        print(f"    interior fraction by zeta: {a['interior_fraction_by_zeta']}")
        print(f"    {a['path']}.csv")


if __name__ == '__main__':
    main()
