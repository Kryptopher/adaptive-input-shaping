"""Step-wise reference tracking from one identification at t = 0.

    python3 -m ais_sim.step_wise

Reference x* = 1, 2, 0, -2, -1 from t = 0, 4, 6, 8, 10 s, zeta = 0.707,
tau = 2 s, K = 0.20, one estimation on [0, tau], at wn = 3 pi and 30 pi.

The first step uses the AIS shaper: K at 0, then A and B at tau + T and
tau + 2T. Every later change of size d uses the same estimate, with no new
identification, as two impulses at T and 2T after the change with amplitudes
A d / (1 - K) and B d / (1 - K). These sum to d and keep the ratio B / A of
the identified shaper.

``gap_to_next`` is the time from a schedule's last impulse to the next change.
``dwell_residual`` is the peak |x - x*| over that gap, as a fraction of |d|.

Writes ais_sim/out/step_wise_<stamp>.csv/.pdf/.png.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from ais_sim.sim import Plant, ais_from_estimate, identify, response

K = 0.20
ZETA = 0.707
TAU = 2.0
OMEGAS = (3.0 * math.pi, 30.0 * math.pi)
SEQUENCE = ((0.0, 1.0), (4.0, 2.0), (6.0, 0.0), (8.0, -2.0), (10.0, -1.0))
T_END = 12.0
OUT = Path(__file__).resolve().parent / 'out'


def run_sequence(omega_n):
    """Returns the commanded impulses and one row per reference change."""
    plant = Plant(omega_n, ZETA)
    est = identify(plant, K, TAU)
    first, _, mode = ais_from_estimate(est, K, TAU)
    if mode != 'oscillatory':
        raise RuntimeError(f'shaper not active at wn = {omega_n}: {mode}')
    A, B, T = first[1][1], first[2][1], first[2][0] - first[1][0]
    later = [(T, A / (1.0 - K)), (2.0 * T, B / (1.0 - K))]
    schedules, previous = [], 0.0
    for i, (change, value) in enumerate(SEQUENCE):
        delta = value - previous
        schedules.append((change, delta, value, [(change + ti, a * delta)
                                                 for ti, a in (first if i == 0 else later)]))
        previous = value
    impulses = [pulse for _, _, _, train in schedules for pulse in train]
    rows = []
    for i, (change, delta, value, train) in enumerate(schedules):
        t_last = train[-1][0]
        t_next = SEQUENCE[i + 1][0] if i + 1 < len(SEQUENCE) else T_END
        dwell = np.linspace(t_last, t_next, 4001)
        rows.append(dict(
            omega_n_over_pi=round(omega_n / math.pi, 6), t_change=change, delta=delta,
            shaper='AIS' if i == 0 else 'same estimate', omega_hat=est.omega_n,
            omega_hat_ratio=est.omega_n / omega_n, zeta_hat=est.zeta, A=A, B=B, T=T,
            t_last=t_last, gap_to_next=t_next - t_last,
            dwell_residual=float(np.max(np.abs(response(plant, impulses, dwell)[0] - value))
                                 / abs(delta))))
    return plant, impulses, rows


def final_residual(plant, impulses):
    """Peak |x - x*| after the last impulse of the sequence, as a fraction of |x*|."""
    t_last = max(t for t, _ in impulses)
    t = t_last + np.linspace(0.0, 10.0 * 2.0 * math.pi / plant.omega_n, 4001)
    final = SEQUENCE[-1][1]
    return float(np.max(np.abs(response(plant, impulses, t)[0] - final)) / abs(final))


def plot(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.76, wspace=0.08)
    t = np.linspace(0.0, T_END, 20001)
    target = np.zeros_like(t)
    for change, value in SEQUENCE:
        target[t >= change] = value
    for ax, (omega_n, plant, impulses), letter in zip(axes, results, 'ab'):
        shaped = sum(a * (t >= ti) for ti, a in impulses)
        ax.axvspan(0.0, TAU, color='lightblue', alpha=0.3, lw=0)
        ax.axvline(TAU, color='lightblue', lw=1.0)
        ax.plot(t, target, color='m', ls='-.', lw=1.2)
        ax.plot(t, shaped, color='r', ls='--', lw=1.2)
        ax.plot(t, response(plant, impulses, t)[0], color='b', lw=1.4)
        ax.set_xlim(0.0, T_END)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks(np.arange(0.0, T_END + 1.0, 2.0))
        ax.set_xlabel(r'$t$ [s]')
        ax.set_title(rf'({letter}) $\omega_n = {round(omega_n / math.pi)}\pi$', loc='left', fontsize=9)
        ax.grid(True, color='#b0b0b0', lw=0.6)
    axes[0].set_ylabel(r'$x$')
    handles = [Line2D([], [], color='m', ls='-.', lw=1.6), Line2D([], [], color='r', ls='--', lw=1.6),
               Line2D([], [], color='b', lw=1.6), Patch(color='lightblue', alpha=0.3)]
    labels = [r'$x^*$', r'$x^*_{\mathrm{TDF}}$', r'$\bar{x}$', 'estimation interval']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=9)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def main():
    rows, results = [], []
    for omega_n in OMEGAS:
        plant, impulses, case = run_sequence(omega_n)
        residual = final_residual(plant, impulses)
        for row in case:
            row['final_A_res'] = residual
        rows += case
        results.append((omega_n, plant, impulses))
        print(f'wn = {omega_n / math.pi:.0f} pi: final residual {residual:.2e} of |x*|')
        print(f"  estimate at t = 0: wn_hat {case[0]['omega_hat']:.6f} rad/s "
              f"(ratio {case[0]['omega_hat_ratio']:.6f}), zeta_hat {case[0]['zeta_hat']:.5f}, "
              f"A {case[0]['A']:.4f}, B {case[0]['B']:.4f}, T {case[0]['T']:.4f}")
        for row in case:
            print(f"  t = {row['t_change']:4.1f}, d = {row['delta']:+.0f} ({row['shaper']}): last impulse "
                  f"{row['t_last']:.3f} s, gap to next {row['gap_to_next']:.3f} s, "
                  f"dwell residual {row['dwell_residual']:.1e}")

    OUT.mkdir(exist_ok=True)
    path = OUT / f"step_wise_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with path.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(results, path)
    print(f'Wrote {path}.csv/.pdf/.png')


if __name__ == '__main__':
    main()
