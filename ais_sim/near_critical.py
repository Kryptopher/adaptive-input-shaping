"""Near-critical damping table.

    python3 -m ais_sim.near_critical

True zeta in {0.80, 0.85, 0.90, 0.95, 0.99, 1.00} at wn = pi, tau = 2 s,
K = 0.20, identified with the light online setup of ais_sim.sweeps. Per case:
zeta_hat, mode, the overshoot with the shaper the rule chose (none when
non-oscillatory), and the overshoot with the shaper disabled (K at 0, the
remaining 1-K at tau).

Writes ais_sim/out/near_critical_<stamp>.csv.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

K = 0.20
OMEGA_N = math.pi
TAU = 2.0
ZETAS = (0.80, 0.85, 0.90, 0.95, 0.99, 1.00)
OUT = Path(__file__).resolve().parent / 'out'


def main():
    from ais_sim.sim import (Plant, ais_from_estimate, disabled_impulses, identify,
                             residual_metrics)
    rows = []
    for zeta in ZETAS:
        plant = Plant(OMEGA_N, zeta)
        est = identify(plant, K, TAU, light=True)
        impulses, _, mode = ais_from_estimate(est, K, TAU)
        shaped = residual_metrics(plant, impulses)
        disabled = residual_metrics(plant, disabled_impulses(K, TAU))
        rows.append(dict(zeta=zeta, zeta_hat=est.zeta, mode=mode,
                         shaped_overshoot_pct=(100.0 * shaped['overshoot']
                                               if mode == 'oscillatory' else math.nan),
                         disabled_overshoot_pct=100.0 * disabled['overshoot']))
    OUT.mkdir(exist_ok=True)
    path = OUT / f"near_critical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('  zeta  zeta_hat  mode             shaped overshoot (%)  disabled overshoot (%)')
    for r in rows:
        print(f"  {r['zeta']:.2f}  {r['zeta_hat']:.3f}     {r['mode']:15s}  "
              f"{r['shaped_overshoot_pct']:<20.2g}  {r['disabled_overshoot_pct']:.3g}")
    print(f'\nWrote {path}')


if __name__ == '__main__':
    main()
