"""Offline analysis of the 25 hardware AIS runs and the 25 pulse runs.

    python3 -m ais_sim.hardware_reanalysis [--data DIR] [--workers N]

The data are in data/hardware/L050 .. L130, one folder per run with raw.csv
(move_time_sec, payload_sample_time_s, swing_mm) and summary.txt (the values
the crane logged). For every AIS run:

1. Rebuild the identification window the crane used. The crane stored
   (payload_sample_time - t0, swing_mm) per payload sample and locked the fit
   made at ``chosen_id_time_s``. t0 is found by trying the rows next to that
   time and keeping the one whose crane grid fit reproduces the logged score
   (``id_cond_b``). Both the unique payload samples and the raw CSV rows are
   tried. raw.csv has one row per control tick, and in some runs a tick repeats
   one payload sample and misses the next, so those windows cannot be rebuilt
   exactly. The number of skipped samples and the score error are reported.
2. Crane grid (:func:`ais_sim.estimator.crane_zero_zeta_grid`): the locked
   omega must be reproduced.
3. RMSE and NRMSE argmin on the crane's 240-point grid, on the locked window
   and on every 0.1 s window from 0.8 s to the lock.
4. The estimator of ais_sim.estimator (2-D grid and refinement) on the locked
   window: refined wn, zeta_hat, score, cond, mode, and the difference from the
   locked omega in percent and in crane grid steps. The same window is also
   run with ``objective='nrmse'``.
5. Free decay of the pulse runs: the estimator on the first 10 s after the
   command stop, after a 0.2 s guard.

Writes ais_sim/out/hardware_reanalysis_<stamp>.csv (one row per AIS run),
hardware_reanalysis_summary_<stamp>.csv (one row per rope length: the table),
hardware_pulse_decay_<stamp>.csv and hardware_argmin_windows_<stamp>.csv.
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
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / 'data' / 'hardware'
LENGTHS = ('L050', 'L070', 'L090', 'L110', 'L130')
OUT = Path(__file__).resolve().parent / 'out'
# Crane identifier settings used in every AIS run.
GRID = dict(t_min=0.45, t_max=1.45, grid_count=240, min_amp=1.0, max_score=1.5)
AFTER_S, PERIOD_S = 0.8, 0.1
GUARD_S, DECAY_S = 0.2, 10.0


def read_summary(run_dir):
    values = {}
    for line in (run_dir / 'summary.txt').read_text().splitlines():
        key, sep, value = line.partition(':')
        if sep:
            try:
                values[key.strip()] = float(value)
            except ValueError:
                values[key.strip()] = value.strip()
    return values


def read_raw(run_dir):
    with (run_dir / 'raw.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    def column(name):
        return np.array([float(r[name]) if r[name] not in ('', 'nan') else math.nan
                         for r in rows])
    return column('move_time_sec'), column('payload_sample_time_s'), column('swing_mm')


def crane_grid_arrays(times, swing, t_min, t_max, grid_count, min_amp, max_score):
    """The computation of ``crane_zero_zeta_grid``, returning every grid point."""
    tc = times - float(times[0])
    periods = np.linspace(t_min, t_max, max(5, grid_count))
    omegas = math.pi / periods
    phase = omegas[:, None] * tc[None, :]
    design = np.empty((len(periods), len(tc), 4), dtype=float)
    design[:, :, 0] = 1.0
    design[:, :, 1] = tc
    design[:, :, 2] = np.cos(phase)
    design[:, :, 3] = np.sin(phase)
    xtx = np.einsum('gni,gnj->gij', design, design, optimize=True)
    xty = np.einsum('gni,n->gi', design, swing, optimize=True)
    coefs = np.linalg.solve(xtx, xty)
    pred = np.einsum('gni,gi->gn', design, coefs, optimize=True)
    rmse = np.sqrt(np.mean((swing[None, :] - pred) ** 2, axis=1))
    amp = np.hypot(coefs[:, 2], coefs[:, 3])
    score = rmse / np.maximum(amp, 1.0e-9)
    cond = np.sqrt(np.linalg.cond(xtx))
    valid = (np.isfinite(score) & np.isfinite(rmse) & np.isfinite(amp)
             & np.isfinite(cond) & (amp >= min_amp) & (score <= max_score))
    return periods, omegas, rmse, score, valid


def argmins(times, swing):
    """(NRMSE argmin, RMSE argmin) grid indices over the valid points."""
    periods, omegas, rmse, score, valid = crane_grid_arrays(times, swing, **GRID)
    if not valid.any():
        return None
    idx = np.flatnonzero(valid)
    return int(idx[np.argmin(score[idx])]), int(idx[np.argmin(rmse[idx])]), omegas


def reconstruct_window(pt, swing, t_fit, logged_score):
    """Recover t0 (and the sample list) so the crane-grid fit at t_fit
    reproduces the logged score.  Returns (score error, t0, fit, variant, pt, swing)."""
    from ais_sim.estimator import crane_zero_zeta_grid
    ok = np.isfinite(pt) & np.isfinite(swing)
    _, first = np.unique(pt[ok], return_index=True)
    variants = {'unique samples': (pt[ok][first], swing[ok][first]),
                'csv rows': (pt[ok], swing[ok])}
    best = None
    for variant, (pv, sv) in variants.items():
        for k in np.flatnonzero(np.abs((pv - pv[0]) - t_fit) < 0.15):
            t = pv - (pv[k] - t_fit)
            mask = (t > 0.0) & (t <= t_fit + 1e-9)
            fit = crane_zero_zeta_grid(t[mask], sv[mask], **GRID)
            if fit is None:
                continue
            err = abs(fit['score'] - logged_score)
            if best is None or err < best[0]:
                best = (err, pv[k] - t_fit, fit, variant, pv, sv)
    return best


def analyse_ais_run(run_dir):
    from ais_sim.estimator import CRANE_CONFIG, estimate
    summary = read_summary(run_dir)
    _, pt, swing = read_raw(run_dir)
    t_fit = summary['chosen_id_time_s']
    locked_omega = summary['id_omega_n_rad_s']
    in_window = (pt - pt[0] <= t_fit + 0.1)
    skipped = int(np.sum(np.diff(pt[in_window]) == 0.0))
    score_err, t0, crane, variant, pt, swing = reconstruct_window(
        pt, swing, t_fit, summary['id_cond_b'])
    t = pt - t0

    def window(end):
        mask = (t > 0.0) & (t <= end + 1e-9)
        return t[mask], swing[mask]

    tw, yw = window(t_fit)
    nrmse_i, rmse_i, omegas = argmins(tw, yw)
    step = abs(omegas[nrmse_i] - omegas[min(nrmse_i + 1, len(omegas) - 1)])
    crane_steps = (crane['omega'] - locked_omega) / step

    windows = []
    ends = np.arange(AFTER_S, t_fit + 1e-9, PERIOD_S)
    for end in list(ends) + [t_fit]:
        we, ye = window(end)
        if len(we) < 80 or np.ptp(ye) < 4.0:
            continue
        picks = argmins(we, ye)
        if picks is None:
            continue
        windows.append(dict(window_end=float(end), nrmse_index=picks[0], rmse_index=picks[1],
                            same=int(picks[0] == picks[1]),
                            nrmse_omega=float(picks[2][picks[0]]),
                            rmse_omega=float(picks[2][picks[1]])))

    tz = tw - tw[0]
    unified = estimate(tz, yw, CRANE_CONFIG)
    unified_nrmse = estimate(tz, yw, replace(CRANE_CONFIG, objective='nrmse'))
    name = run_dir.name
    row = dict(
        length=run_dir.parent.name, rope_m=float(summary['rope_length_m']),
        rep=name[name.rfind('rep'):], run=name,
        window_s=float(t_fit), n_samples=len(tw),
        locked_omega=locked_omega, locked_T_half=summary['chosen_id_T_s'],
        schedule_T=summary['schedule_T_s'],
        reconstruct_score_err=score_err, reconstruct_variant=variant,
        csv_skipped_samples=skipped,
        crane_omega=crane['omega'], crane_reproduces=int(abs(crane['omega'] - locked_omega) < 1e-12),
        crane_minus_locked_steps=crane_steps,
        nrmse_argmin=nrmse_i, rmse_argmin=rmse_i, argmin_same=int(nrmse_i == rmse_i),
        windows_checked=len(windows), windows_differ=sum(1 - w['same'] for w in windows),
        refined_omega=unified.omega_n, zeta_hat=unified.zeta, mode=unified.mode,
        score=unified.score, rmse=unified.rmse, cond=unified.cond, amp_mm=unified.amp,
        refined_minus_locked_pct=100.0 * (unified.omega_n / locked_omega - 1.0),
        refined_minus_locked_steps=(unified.omega_n - locked_omega) / step,
        nrmse_obj_omega=unified_nrmse.omega_n, nrmse_obj_zeta=unified_nrmse.zeta,
        refined_period_s=2 * math.pi / unified.omega_n,
        grid_step_omega=step,
    )
    for w in windows:
        w['run'] = name
        w['length'] = row['length']
    return row, windows


def analyse_pulse_run(run_dir):
    from ais_sim.estimator import CRANE_CONFIG, estimate
    summary = read_summary(run_dir)
    mt, pt, swing = read_raw(run_dir)
    stop_pt = float(np.interp(summary['command_stop_time_s'], mt, pt))
    t = pt - (stop_pt + GUARD_S)
    mask = (t >= 0.0) & (t <= DECAY_S) & np.isfinite(swing)
    tw, yw = t[mask], swing[mask]
    _, k = np.unique(tw, return_index=True)
    tw, yw = tw[k], yw[k]
    result = estimate(tw - tw[0], yw, CRANE_CONFIG)
    name = run_dir.name
    return dict(length=run_dir.parent.name, rope_m=float(summary['rope_length_m']),
                rep=name[name.rfind('rep'):], run=name, window_s=float(tw[-1] - tw[0]),
                n_samples=len(tw), omega=result.omega_n, zeta_hat=result.zeta,
                mode=result.mode, score=result.score, amp_mm=result.amp, valid=int(result.valid))


def discover(data, kind):
    pattern = {'ais': '*_is2_*isa_zerozeta*', 'pulse': '*_pulse_*'}[kind]
    return sorted(p for L in LENGTHS for p in (data / L).glob(pattern) if p.is_dir())


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else math.nan


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    ais_runs, pulse_runs = discover(args.data, 'ais'), discover(args.data, 'pulse')
    print(f'{len(ais_runs)} AIS runs, {len(pulse_runs)} pulse runs under {args.data}')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        ais = list(pool.map(analyse_ais_run, ais_runs))
        pulses = list(pool.map(analyse_pulse_run, pulse_runs))
    rows = [a[0] for a in ais]
    windows = [w for a in ais for w in a[1]]
    final = {r['run']: r['refined_omega'] for r in rows}
    for w in windows:
        w['nrmse_err_pct'] = 100.0 * abs(w['nrmse_omega'] / final[w['run']] - 1.0)
        w['rmse_err_pct'] = 100.0 * abs(w['rmse_omega'] / final[w['run']] - 1.0)

    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    write_csv(OUT / f'hardware_reanalysis_{stamp}.csv', rows)
    write_csv(OUT / f'hardware_pulse_decay_{stamp}.csv', pulses)
    write_csv(OUT / f'hardware_argmin_windows_{stamp}.csv', windows)

    print('\nPer run: locked = what ran, refined = unified estimator on the same window')
    print('  run          locked   skipped  rebuilt (score err)  argmin same  refined   '
          'zeta_hat  d%       d steps  mode')
    for r in rows:
        rebuilt = ('exact' if r['crane_reproduces'] else
                   f"{r['crane_minus_locked_steps']:+.0f} step")
        print(f"  {r['length']} {r['rep']}   {r['locked_omega']:.4f}   {r['csv_skipped_samples']:3d}"
              f"      {rebuilt:9s} ({r['reconstruct_score_err']:.0e})  "
              f"{'yes' if r['argmin_same'] else 'NO '}          {r['refined_omega']:.4f}    "
              f"{r['zeta_hat']:.4f}   {r['refined_minus_locked_pct']:+.3f}  "
              f"{r['refined_minus_locked_steps']:+.2f}    {r['mode']}")
    summary = []
    for length in LENGTHS:
        sub = [r for r in rows if r['length'] == length]
        ps = [p for p in pulses if p['length'] == length and p['valid']]
        lo, ls = mean_std([r['locked_omega'] for r in sub])
        ro, rs = mean_std([r['refined_omega'] for r in sub])
        po, _ = mean_std([p['omega'] for p in ps])
        pz, pzs = mean_std([p['zeta_hat'] for p in ps])
        summary.append(dict(length=length, rope_m=sub[0]['rope_m'], n_ais=len(sub),
                            locked_omega_mean=lo, locked_omega_sd=ls,
                            refined_omega_mean=ro, refined_omega_sd=rs,
                            zeta_hat_mean=float(np.mean([r['zeta_hat'] for r in sub])),
                            zeta_hat_max=max(r['zeta_hat'] for r in sub),
                            n_pulse=len(ps), pulse_omega_mean=po,
                            pulse_zeta_mean=pz, pulse_zeta_sd=pzs))
    write_csv(OUT / f'hardware_reanalysis_summary_{stamp}.csv', summary)

    print('\nPer length (mean ± std over reps):')
    print('  L      locked wn          refined wn          zeta_hat (AIS window)   '
          'pulse wn            pulse zeta_hat')
    for length in LENGTHS:
        sub = [r for r in rows if r['length'] == length]
        ps = [p for p in pulses if p['length'] == length and p['valid']]
        lo, ls = mean_std([r['locked_omega'] for r in sub])
        ro, rs = mean_std([r['refined_omega'] for r in sub])
        zo, zs = mean_std([r['zeta_hat'] for r in sub])
        po, pss = mean_std([p['omega'] for p in ps])
        pz, pzs = mean_std([p['zeta_hat'] for p in ps])
        print(f'  {length}   {lo:.4f} ± {ls:.4f}   {ro:.4f} ± {rs:.4f}   '
              f'{zo:.4f} ± {zs:.4f}         {po:.4f} ± {pss:.4f}   {pz:.5f} ± {pzs:.5f}'
              f'   (n={len(sub)}/{len(ps)})')
    differ = [r for r in rows if not r['argmin_same']]
    n_windows = len(windows)
    n_wdiff = sum(1 - w['same'] for w in windows)
    print(f'\nRMSE vs NRMSE argmin on the locked window: same in {len(rows) - len(differ)}'
          f'/{len(rows)} runs')
    for r in differ:
        print(f"  differs: {r['run']}  NRMSE idx {r['nrmse_argmin']}  RMSE idx {r['rmse_argmin']}")
    print(f'Over all online windows: same in {n_windows - n_wdiff}/{n_windows}')
    complete = [r for r in rows if r['csv_skipped_samples'] == 0]
    print(f"Locked omega rebuilt from raw.csv in {sum(r['crane_reproduces'] for r in rows)}"
          f"/{len(rows)} runs ({sum(r['crane_reproduces'] for r in complete)}/{len(complete)}"
          f" runs whose CSV holds every payload sample); max score error "
          f"{max(r['reconstruct_score_err'] for r in rows):.1e}; max offset "
          f"{max(abs(r['crane_minus_locked_steps']) for r in rows):.0f} grid step")
    early = np.array([w['window_end'] for w in windows]) < 3.0
    same = np.array([w['same'] for w in windows], dtype=bool)
    print(f'  windows ending before 3.0 s: same in {np.sum(same & early)}/{np.sum(early)}; '
          f'from 3.0 s: same in {np.sum(same & ~early)}/{np.sum(~early)}')
    print(f"Max |refined - locked| = {max(abs(r['refined_minus_locked_steps']) for r in rows):.2f}"
          f" grid steps; max zeta_hat = {max(r['zeta_hat'] for r in rows):.4f}")
    print('\nCrane-grid omega within 2 % of the run\'s final refined value, by window end:')
    ends = np.array([w['window_end'] for w in windows])
    for lo, hi in ((0.8, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.8)):
        sel = [w for w, e in zip(windows, ends) if lo <= e < hi]
        if sel:
            print(f'  {lo:.1f}-{hi:.1f} s (n={len(sel):3d}):  NRMSE argmin '
                  f"{np.mean([w['nrmse_err_pct'] < 2 for w in sel]):.2f}   RMSE argmin "
                  f"{np.mean([w['rmse_err_pct'] < 2 for w in sel]):.2f}")
    print(f'\nWrote {OUT}/hardware_*_{stamp}.*')


if __name__ == '__main__':
    main()
