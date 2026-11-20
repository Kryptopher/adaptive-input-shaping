"""Sensitivity of ZV, ZVD and the AIS as percentage residual vibration.

    python3 -m ais_sim.sensitivity

Percentage residual vibration of a shaper designed for (wn_hat, zeta_hat) on a
plant with the true wn:

    V(wn) = 100 * e^{-sigma t_N} |sum_i A_i e^{sigma t_i} e^{j wd t_i}|,
    sigma = zeta wn, wd = wn sqrt(1 - zeta^2), sum_i A_i = 1,

against wn / wn_hat in [0.5, 1.5] (V = 100 % unshaped). For an undamped plant V
equals the peak residual A_res of the simulation. The shaded band is the 95th
percentile frequency estimation error at 10 dB (ais_sim.noise_study).

  ZV, ZVD   designed for (wn_hat, zeta_hat)
  AIS       K = 0.20 at 0, A at tau+T, B at tau+2T, tau in {1, 2, 4} s

wn_hat = 3.245 rad/s (the hardware pendulum at L = 0.90 m), zeta = zeta_hat in
{0, 0.1}.

Writes ais_sim/out/sensitivity_<stamp>.csv/.pdf/.png and
sensitivity_<stamp>_band.csv (V at the errors in BAND_ERRORS, also for the
hardware window tau = 3.7 s).
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

K = 0.20
OMEGA_HAT = 3.245
TAUS = (1.0, 2.0, 4.0)
HARDWARE_TAU = 3.7
BAND = (0.99, 1.01)
BAND_ERRORS = (-0.01, -0.003, 0.003, 0.01)
ZETAS = (0.0, 0.1)
RATIOS = np.linspace(0.5, 1.5, 1001)
LEVELS = (5.0, 10.0)
OUT = Path(__file__).resolve().parent / 'out'
COLORS = {'ZV': '#52514e', 'ZVD': '#0b0b0b', 'AIS tau=1': '#9cc3f0',
          'AIS tau=2': '#2a78d6', 'AIS tau=4': '#173f73'}
STYLES = {'ZV': '--', 'ZVD': ':'}
BAND_SHADE = dict(color='#d9d8d2', alpha=0.6, lw=0)
BAND_EDGE = dict(color='#52514e', lw=0.7)


def zv(omega_n, zeta):
    omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)
    k = math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta * zeta))
    half = math.pi / omega_d
    return [(0.0, 1.0 / (1.0 + k)), (half, k / (1.0 + k))]


def zvd(omega_n, zeta):
    omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)
    k = math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta * zeta))
    half = math.pi / omega_d
    norm = (1.0 + k) ** 2
    return [(0.0, 1.0 / norm), (half, 2.0 * k / norm), (2.0 * half, k * k / norm)]


def ais(omega_n, zeta, tau):
    from ais_sim.shaper import design_shaper
    from ais_sim.sim import ais_impulses
    solution = design_shaper(K, tau, omega_n, zeta)
    if not solution.feasible:
        raise RuntimeError(f'AIS infeasible at tau={tau}, zeta={zeta}')
    return ais_impulses(solution)


def vibration(impulses, omega_n, zeta):
    """Percentage residual vibration of ``impulses`` on plant(s) ``omega_n``."""
    omega_n = np.asarray(omega_n, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    sigma = zeta * omega_n
    omega_d = omega_n * np.sqrt(1.0 - zeta * zeta)
    t_last = max(t for t, _ in impulses)
    total = sum(a * np.exp(sigma * (t - t_last)) * np.exp(1j * omega_d * t)
                for t, a in impulses)
    return 100.0 * np.abs(total)


def width(ratios, values, level):
    """Width of the contiguous band around ratio 1 with V <= level."""
    centre = int(np.argmin(np.abs(ratios - 1.0)))
    if values[centre] > level:
        return 0.0
    lo = centre
    while lo > 0 and values[lo - 1] <= level:
        lo -= 1
    hi = centre
    while hi < len(ratios) - 1 and values[hi + 1] <= level:
        hi += 1
    return float(ratios[hi] - ratios[lo])


def shapers(zeta):
    out = {'ZV': zv(OMEGA_HAT, zeta), 'ZVD': zvd(OMEGA_HAT, zeta)}
    for tau in TAUS:
        out[f'AIS tau={tau:g}'] = ais(OMEGA_HAT, zeta, tau)
    return out


def plot_curves(curves, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, len(ZETAS), figsize=(7.2, 3.0), sharey=True,
                             constrained_layout=True)
    for ax, zeta in zip(axes, ZETAS):
        ax.axvspan(*BAND, **BAND_SHADE)
        for edge in BAND:
            ax.axvline(edge, **BAND_EDGE)
        for name, values in curves[zeta].items():
            ax.plot(RATIOS, values, color=COLORS[name], ls=STYLES.get(name, '-'), lw=1.5,
                    label=name.replace('tau', r'$\tau$') + (' s' if 'AIS' in name else ''))
        ax.axhline(5.0, color='#888888', lw=0.6, ls=':')
        ax.annotate('AIS estimation error\n(95th pct., 10 dB)', xy=(BAND[1], 88.0),
                    xytext=(1.05, 88.0), fontsize=7, color='#52514e', va='center',
                    arrowprops=dict(arrowstyle='-', color='#52514e', lw=0.6))
        ax.set_title(rf'$\zeta = \hat\zeta = {zeta:g}$', loc='left')
        ax.set_xlabel(r'$\omega_n / \hat\omega_n$')
        ax.set_xlim(0.5, 1.5)
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, color='#e6e5e1', lw=0.6)
    axes[0].set_ylabel('residual vibration (%)')
    axes[1].legend(frameon=False, fontsize=7, loc='upper left')
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    curves, rows, summary = {}, [], []
    for zeta in ZETAS:
        curves[zeta] = {}
        for name, impulses in shapers(zeta).items():
            values = vibration(impulses, RATIOS * OMEGA_HAT, zeta)
            curves[zeta][name] = values
            for r, v in zip(RATIOS, values):
                rows.append(dict(zeta=zeta, shaper=name, ratio=float(r), V_pct=float(v)))
            t_end = max(t for t, _ in impulses)
            summary.append(dict(zeta=zeta, shaper=name, t_end=t_end,
                                duration_after_estimate=t_end - (
                                    float(name.split('=')[1]) if 'AIS' in name else 0.0),
                                V_at_nominal=float(vibration(impulses, OMEGA_HAT, zeta)),
                                **{f'width_{lv:g}pct': width(RATIOS, values, lv) for lv in LEVELS},
                                V_at_0p9=float(vibration(impulses, 0.9 * OMEGA_HAT, zeta)),
                                V_at_1p1=float(vibration(impulses, 1.1 * OMEGA_HAT, zeta))))

    path = OUT / f'sensitivity_{stamp}'
    with path.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_curves(curves, path)

    band_rows = []
    for zeta in ZETAS:
        band_shapers = shapers(zeta)
        band_shapers[f'AIS tau={HARDWARE_TAU:g}'] = ais(OMEGA_HAT, zeta, HARDWARE_TAU)
        for name, impulses in band_shapers.items():
            band_rows.append(dict(zeta=zeta, shaper=name, **{
                f'V_at_{100.0 * e:+g}pct': float(vibration(impulses, (1.0 + e) * OMEGA_HAT, zeta))
                for e in BAND_ERRORS}))
    with path.with_name(f'{path.name}_band.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(band_rows[0]))
        writer.writeheader()
        writer.writerows(band_rows)

    print(f'wn_hat = {OMEGA_HAT} rad/s, K = {K}')
    print('  zeta  shaper      span after estimate  V(nominal)  V(0.9)  V(1.1)  '
          'width V<=5%   width V<=10%')
    for s in summary:
        print(f"  {s['zeta']:<4g}  {s['shaper']:<10s}  {s['duration_after_estimate']:6.3f} s"
              f"            {s['V_at_nominal']:.1e}     {s['V_at_0p9']:5.1f}   {s['V_at_1p1']:5.1f}"
              f"   {s['width_5pct']:.3f}         {s['width_10pct']:.3f}")
    print('\n  zeta  shaper      ' + '  '.join(f'V({100.0 * e:+g} %)' for e in BAND_ERRORS))
    for row in band_rows:
        print(f"  {row['zeta']:<4g}  {row['shaper']:<10s}  " + '  '.join(
            f"{row[f'V_at_{100.0 * e:+g}pct']:8.2f}" for e in BAND_ERRORS))
    print(f'\nWrote {path}.csv/.pdf/.png and {path.name}_band.csv')


if __name__ == '__main__':
    main()
