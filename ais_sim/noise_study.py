"""Estimator and shaper performance under measurement noise.

    python3 -m ais_sim.noise_study [--trials 200] [--workers N]
    python3 -m ais_sim.noise_study --replot TRIALS_CSV

Plant wn = 3 pi, zeta in {0, 0.3}, tau = 2.5 s, K = 0.20. Zero-mean Gaussian
noise is added to the K step response over the identification window at
SNR = 40, 20, 10 dB, where

    SNR = mean[(x(t) - K)^2] / sigma_noise^2    over [0, tau], noise-free x.

tau = 2.5 s gives phi = 7.5 pi at zeta = 0, away from the half period branch
phi = m pi.

Each trial runs the full online estimator (latest valid estimate before the
deadline) twice on the same noise realization: without a pre-filter, and with
a zero-phase 2nd-order Butterworth low-pass at 10 wn applied to each update's
window.

Per trial: wn and zeta estimates, mode, and the peak residual A_res on the true
plant for the AIS designed from the estimate. A trial is gated out when no
valid oscillatory estimate exists by the deadline (shaper disabled). The trial
also records A_res for FT-b (``sim.ft_impulses`` variant 'b') and AIS-min
(``two_impulse``) designed from the same estimate.

A full run writes ais_sim/out/noise_study_<stamp>.csv (one row per trial and
filter), _summary.csv and .pdf/.png. ``--replot`` writes the summary and the
figure from a saved trials CSV.
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
TAU = 2.5
ZETAS = (0.0, 0.3)
SNRS_DB = (40.0, 20.0, 10.0)
OUT = Path(__file__).resolve().parent / 'out'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def signal_power(zeta):
    from ais_sim.sim import Plant, sample_interval
    plant = Plant(OMEGA_N, zeta)
    dt = sample_interval(plant, TAU)
    t = np.arange(0.0, TAU + 0.5 * dt, dt)
    return float(np.mean((K * plant.step(t) - K) ** 2))


def butterworth_prefilter(tw, yw):
    from scipy.signal import butter, sosfiltfilt
    fs = 1.0 / float(np.median(np.diff(tw)))
    sos = butter(2, 10.0 * OMEGA_N / (2.0 * math.pi), fs=fs, output='sos')
    return sosfiltfilt(sos, yw)


def run_trial(args):
    zeta, snr_db, trial, seed = args
    from ais_sim.sim import (Plant, ais_from_estimate, disabled_impulses, ft_impulses,
                             identify, residual_metrics)
    from ais_sim.two_impulse import min_end_two_impulse
    plant = Plant(OMEGA_N, zeta)
    noise_std = math.sqrt(signal_power(zeta) / 10.0 ** (snr_db / 10.0))
    rows = []
    for filtered in (False, True):
        rng = np.random.default_rng(seed)  # same noise with and without the pre-filter
        est = identify(plant, K, TAU, noise_std=noise_std, rng=rng,
                       prefilter=butterworth_prefilter if filtered else None)
        impulses, solution, mode = ais_from_estimate(est, K, TAU)
        usable = mode == 'oscillatory'
        row = dict(zeta=zeta, snr_db=snr_db, trial=trial, prefilter=int(filtered),
                   noise_std=noise_std, mode=mode, gated_out=int(not usable),
                   omega_hat=est.omega_n if est is not None else math.nan,
                   zeta_hat=est.zeta if est is not None else math.nan,
                   omega_err_pct=(100.0 * (est.omega_n / OMEGA_N - 1.0)
                                  if est is not None else math.nan),
                   AIS_A_res=residual_metrics(plant, impulses)['A_res'],
                   AIS_t_end=max(t for t, _ in impulses))
        if usable:
            z_hat = est.zeta
            ft_b = ft_impulses(K, est.omega_n, z_hat, TAU, 'b')
            family = min_end_two_impulse(K, TAU, est.omega_n, z_hat)
            family_imp = family.impulses() if family.feasible else disabled_impulses(K, TAU)
        else:
            ft_b = family_imp = disabled_impulses(K, TAU)
        row['FT-b_A_res'] = residual_metrics(plant, ft_b)['A_res']
        row['AIS-min_A_res'] = residual_metrics(plant, family_imp)['A_res']
        rows.append(row)
    return rows


def summarize(rows):
    out = []
    for zeta in ZETAS:
        for snr in SNRS_DB:
            for filtered in (0, 1):
                sub = [r for r in rows if r['zeta'] == zeta and r['snr_db'] == snr
                       and r['prefilter'] == filtered]
                ok = [r for r in sub if not r['gated_out']]

                def q(key, rows_=ok):
                    values = np.array([r[key] for r in rows_], dtype=float)
                    if values.size == 0:
                        return (math.nan,) * 3
                    return tuple(float(v) for v in np.percentile(values, [25, 50, 75]))

                summary = dict(zeta=zeta, snr_db=snr, prefilter=filtered, trials=len(sub),
                               gated_out_fraction=1.0 - len(ok) / len(sub) if sub else math.nan)
                for key in ('omega_err_pct', 'zeta_hat', 'AIS_A_res', 'FT-b_A_res',
                            'AIS-min_A_res'):
                    q25, q50, q75 = q(key)
                    summary[f'{key}_q25'], summary[f'{key}_median'], summary[f'{key}_q75'] = \
                        q25, q50, q75
                abs_err = np.abs([r['omega_err_pct'] for r in ok]) if ok else np.array([])
                summary['omega_abs_err_pct_p95'] = (float(np.percentile(abs_err, 95))
                                                    if abs_err.size else math.nan)
                out.append(summary)
    return out


A_RES_YLIM = (1e-7, 1e-1)  # both A_res panels, covers every whisker
ZETA0_HAT_TOP = 0.0025     # panel (c): headroom above the 10 dB whiskers (0.0022)


def _boxes(ax, groups, positions, colors):
    parts = ax.boxplot(groups, positions=positions, widths=0.32, patch_artist=True,
                       showfliers=False,
                       medianprops=dict(color='#0b0b0b', lw=1.0))
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(color)


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import LogLocator
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

    metrics = (('omega_err_pct', r'$\hat\omega_n$ error (%)', False),
               ('zeta_hat', r'$\hat\zeta$', False),
               ('AIS_A_res', r'$A_{res}$ (fraction of step)', True))
    fig, axes = plt.subplots(len(metrics), len(ZETAS), figsize=(7.2, 7.0),
                             constrained_layout=True, sharex=True)
    for col, zeta in enumerate(ZETAS):
        for row_i, (key, label, log) in enumerate(metrics):
            ax = axes[row_i, col]
            ax.text(0.02, 0.97, f'({"abcdef"[row_i * len(ZETAS) + col]})', transform=ax.transAxes,
                    ha='left', va='top', fontsize=9)
            groups, positions, colors = [], [], []
            for i, snr in enumerate(SNRS_DB):
                for filtered, offset, color in ((0, -0.18, BLUE), (1, 0.18, ORANGE)):
                    values = [r[key] for r in rows if r['zeta'] == zeta and r['snr_db'] == snr
                              and r['prefilter'] == filtered and not r['gated_out']]
                    groups.append(values if values else [math.nan])
                    positions.append(i + offset)
                    colors.append(color)
            _boxes(ax, groups, positions, colors)
            if log:
                ax.set_yscale('log')
                ax.set_ylim(*A_RES_YLIM)
                ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            if key == 'zeta_hat':
                ax.axhline(zeta, color='#888888', lw=0.8, ls='--')
                if zeta == 0.0:
                    ax.set_ylim(top=ZETA0_HAT_TOP)
            if key == 'omega_err_pct':
                ax.axhline(0.0, color='#888888', lw=0.8, ls='--')
            ax.grid(True, axis='y', color='#e6e5e1', lw=0.6)
            if col == 0:
                ax.set_ylabel(label)
        axes[0, col].set_title(rf'$\zeta = {zeta:g}$', loc='center', fontsize=9)
        axes[-1, col].set_xticks(range(len(SNRS_DB)))
        axes[-1, col].set_xticklabels([f'{snr:.0f} dB' for snr in SNRS_DB])
        axes[-1, col].set_xlabel('SNR')
    axes[0, 0].legend(handles=[Patch(color=BLUE, alpha=0.75, label='no pre-filter'),
                               Patch(color=ORANGE, alpha=0.75, label='with pre-filter')],
                      frameon=False, fontsize=8, loc='lower left')
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--replot', metavar='CSV', help='summary and figure from a saved trials CSV')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = OUT / f'noise_study_{stamp}'
    if args.replot:
        with Path(args.replot).open() as handle:
            rows = [{k: (float(v) if k not in ('mode',) else v) for k, v in r.items()}
                    for r in csv.DictReader(handle)]
        for r in rows:
            r['prefilter'] = int(r['prefilter'])
            r['gated_out'] = int(r['gated_out'])
    else:
        cases = [(z, snr, trial, 1_000_000 * zi + 10_000 * si + trial)
                 for zi, z in enumerate(ZETAS) for si, snr in enumerate(SNRS_DB)
                 for trial in range(args.trials)]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, trial_rows in enumerate(pool.map(run_trial, cases, chunksize=4), 1):
                rows.extend(trial_rows)
                if i % 50 == 0 or i == len(cases):
                    print(f'[{i}/{len(cases)}] trials done', flush=True)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with path.with_suffix('.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows)
    with path.with_name(path.name + '_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot(rows, path)

    print('\nzeta  SNR  filt  gated  |err wn| med/p95 (%)  zeta_hat med [IQR]       '
          'A_res med: AIS / AIS-min / FT-b')
    for s in summary:
        print(f"{s['zeta']:.1f}  {s['snr_db']:4.0f}  {s['prefilter']}    "
              f"{100 * s['gated_out_fraction']:5.1f}%  "
              f"{abs(s['omega_err_pct_median']):.3f} / {s['omega_abs_err_pct_p95']:.3f}     "
              f"{s['zeta_hat_median']:.4f} [{s['zeta_hat_q25']:.4f}, {s['zeta_hat_q75']:.4f}]  "
              f"{s['AIS_A_res_median']:.2e} / {s['AIS-min_A_res_median']:.2e} / "
              f"{s['FT-b_A_res_median']:.2e}")
    print(f'\nWrote {path}_summary.csv and .pdf/.png' + ('' if args.replot else ' and .csv'))


if __name__ == '__main__':
    main()
