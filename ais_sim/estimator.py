"""Natural frequency and damping estimator used in simulation and on the hardware data.

For each candidate natural frequency ``wn`` and damping ratio ``zeta`` the model

    y(t) = c0 + c1*t + exp(-sigma*t) * (a*cos(wd*t) + b*sin(wd*t)),
    sigma = zeta*wn,   wd = wn*sqrt(1 - zeta**2)

is linear in (c0, c1, a, b) and is solved by least squares. Candidates are
ranked by the residual RMSE (``objective='rmse'``, the default) or by the
normalized error ``rmse / hypot(a, b)`` (``objective='nrmse'``), which is the
objective of the crane's online identifier (:func:`crane_zero_zeta_grid`).
NRMSE <= ``score_max`` is used as a gate under either objective. Time is
measured from the first sample of the window.

Search stages:

1. Coarse. (a) zeta = 0: zero-padded periodogram of the detrended window, with
   the strongest peaks scored by the zeta = 0 least-squares fit.
   (b) A log-spaced (wn, zeta) grid over [omega_min, omega_max].
   The better of the two seeds the next stage.
2. Fine 2-D grid: wn linear over +/-25 % of the coarse value, zeta linear over
   [0, zeta_max].
3. Refinement: parabolic interpolation on the fine grid along each axis, then a
   bounded polish of the same objective (variable projection least squares for
   RMSE, Nelder-Mead for NRMSE), started from the parabolic point and from every
   periodogram seed. The lowest valid score wins.

Gates on every candidate: finite fit, amp <= ``amp_max_ratio`` * ptp(y),
NRMSE <= ``score_max``, cond(X^T X) <= ``cond_max``. A step response has
amp = K/sqrt(1 - zeta**2), at most 3.2 K for zeta <= 0.95, while ptp(y) >= K,
so the amplitude gate at 4 does not reject a true fit.

Decision rule: with the estimated overshoot Mp = exp(-pi*zeta/sqrt(1 - zeta**2)),
the mode is ``non-oscillatory`` (no shaper) if Mp < ``mp_min`` and
``oscillatory`` otherwise. It is ``no-estimate`` when the final fit fails the
gates.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize


@dataclass(frozen=True)
class EstimatorConfig:
    """Search ranges, gates and decision thresholds.

    ``omega_min`` defaults to 2*pi/(4*T_window) and ``omega_max`` to the Nyquist
    frequency of the median sample interval, both in rad/s. ``amp_min`` is in the
    units of ``y``.
    """

    omega_min: float | None = None
    omega_max: float | None = None
    zeta_max: float = 0.95
    coarse_peaks: int = 3
    coarse_zeta_count: int = 20
    coarse_log_step: float = 0.03
    coarse_max_omega_count: int = 300
    fine_omega_span: float = 0.25
    fine_omega_count: int = 120
    fine_zeta_count: int = 40
    grid_max_samples: int = 2000
    objective: str = 'rmse'
    polish_max_samples: int = 5000
    amp_min: float = 0.0
    amp_max_ratio: float = 4.0
    score_max: float = 1.5
    cond_max: float = 1.0e8
    mp_min: float = 0.01
    chunk_elems: int = 4_000_000


# Hardware data: swing in mm, with the crane's 1 mm amplitude floor.
CRANE_CONFIG = EstimatorConfig(amp_min=1.0, score_max=1.5)


@dataclass
class EstimateResult:
    valid: bool
    mode: str
    omega_n: float
    zeta: float
    amp: float
    score: float
    rmse: float
    cond: float
    wall_time_s: float


def _decimate(t: np.ndarray, y: np.ndarray, max_samples: int):
    if t.size <= max_samples:
        return t, y
    index = np.unique(np.linspace(0, t.size - 1, max_samples).round().astype(int))
    return t[index], y[index]


def _grid_fit(t, y, omega_n, zeta, t_scale, chunk_elems):
    """Least-squares fit at every (omega_n[i], zeta[i]) pair.

    The normal equations are formed from sums, in chunks over the grid so the
    G x N basis arrays stay bounded in memory. Returns coefficients (G, 4) with
    the drift column per unit ``t/t_scale``, rmse, amplitude and cond(X^T X).
    """
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    zeta = np.asarray(zeta, dtype=float).ravel()
    count = omega_n.size
    n = t.size
    ts = t / t_scale
    s_t = ts.sum()
    s_tt = ts @ ts
    s_y = y.sum()
    s_ty = ts @ y
    s_yy = y @ y

    coefs = np.full((count, 4), np.nan)
    rss = np.full(count, np.nan)
    cond = np.full(count, np.inf)
    step = max(1, int(chunk_elems // max(n, 1)))
    for start in range(0, count, step):
        stop = min(count, start + step)
        wn = omega_n[start:stop, None]
        z = zeta[start:stop, None]
        envelope = np.exp(-(z * wn) * t[None, :])
        phase = wn * np.sqrt(1.0 - z * z) * t[None, :]
        e = envelope * np.cos(phase)
        f = envelope * np.sin(phase)
        g = e.shape[0]

        xtx = np.empty((g, 4, 4))
        xtx[:, 0, 0] = n
        xtx[:, 0, 1] = xtx[:, 1, 0] = s_t
        xtx[:, 0, 2] = xtx[:, 2, 0] = e.sum(axis=1)
        xtx[:, 0, 3] = xtx[:, 3, 0] = f.sum(axis=1)
        xtx[:, 1, 1] = s_tt
        xtx[:, 1, 2] = xtx[:, 2, 1] = e @ ts
        xtx[:, 1, 3] = xtx[:, 3, 1] = f @ ts
        xtx[:, 2, 2] = np.einsum('gn,gn->g', e, e)
        xtx[:, 3, 3] = np.einsum('gn,gn->g', f, f)
        xtx[:, 2, 3] = xtx[:, 3, 2] = np.einsum('gn,gn->g', e, f)
        xty = np.column_stack((np.full(g, s_y), np.full(g, s_ty), e @ y, f @ y))

        with np.errstate(all='ignore'):
            chunk_cond = np.linalg.cond(xtx)
        ok = np.isfinite(chunk_cond) & (chunk_cond < 1.0e15)
        if not np.any(ok):
            continue
        sol = np.linalg.solve(xtx[ok], xty[ok][..., None])[..., 0]
        chunk_rss = (
            s_yy
            - 2.0 * np.einsum('gi,gi->g', sol, xty[ok])
            + np.einsum('gi,gij,gj->g', sol, xtx[ok], sol)
        )
        index = np.arange(start, stop)[ok]
        coefs[index] = sol
        rss[index] = np.maximum(chunk_rss, 0.0)
        cond[index] = chunk_cond[ok]

    rmse = np.sqrt(rss / n)
    amp = np.hypot(coefs[:, 2], coefs[:, 3])
    return coefs, rmse, amp, cond


def _direct_fit(t, y, omega_n, zeta, t_scale):
    """One least-squares solve with the explicit design matrix."""
    envelope = np.exp(-zeta * omega_n * t)
    phase = omega_n * math.sqrt(max(1.0 - zeta * zeta, 0.0)) * t
    x = np.column_stack((
        np.ones_like(t), t / t_scale, envelope * np.cos(phase), envelope * np.sin(phase)))
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coef
    rmse = float(np.sqrt(np.mean(residual * residual)))
    amp = float(math.hypot(coef[2], coef[3]))
    with np.errstate(all='ignore'):
        cond = float(np.linalg.cond(x) ** 2)
    return coef, rmse, amp, cond


class _Scorer:
    def __init__(self, config: EstimatorConfig, y_scale: float, y_p2p: float):
        self.config = config
        self.amp_floor = max(1.0e-12 * y_scale, 1.0e-300)
        self.amp_max = config.amp_max_ratio * y_p2p

    def score(self, rmse, amp):
        return rmse / np.maximum(amp, self.amp_floor)

    def valid(self, score, rmse, amp, cond):
        cfg = self.config
        return (
            np.isfinite(score) & np.isfinite(rmse) & np.isfinite(amp)
            & np.isfinite(cond) & (amp > self.amp_floor) & (amp >= cfg.amp_min)
            & (amp <= self.amp_max)
            & (score <= cfg.score_max) & (cond <= cfg.cond_max)
        )


def _periodogram_seeds(t, y, omega_min, omega_max, n_peaks):
    """Strongest periodogram peaks of the linearly detrended window."""
    n = t.size
    dt = float(np.median(np.diff(t)))
    uniform_t = t[0] + dt * np.arange(n)
    uniform_y = np.interp(uniform_t, t, y)
    trend = np.column_stack((np.ones(n), uniform_t - uniform_t[0]))
    coef, *_ = np.linalg.lstsq(trend, uniform_y, rcond=None)
    detrended = (uniform_y - trend @ coef) * np.hanning(n)
    nfft = min(1 << 22, 1 << int(math.ceil(math.log2(max(8 * n, 1024)))))
    spectrum = np.abs(np.fft.rfft(detrended, nfft))
    omega = 2.0 * math.pi * np.fft.rfftfreq(nfft, dt)
    index = np.flatnonzero((omega >= omega_min) & (omega <= omega_max))
    if index.size == 0:
        return np.array([0.5 * (omega_min + omega_max)])
    if index.size < 3:
        return omega[index[[np.argmax(spectrum[index])]]]
    values = spectrum[index]
    interior = (values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:])
    peaks = index[1:-1][interior]
    if peaks.size == 0:
        peaks = index[[np.argmax(values)]]
    order = peaks[np.argsort(spectrum[peaks])[::-1][:n_peaks]]
    return omega[order]


def _parabolic_offset(left, centre, right):
    """Vertex offset, in grid steps, of the parabola through three scores."""
    denominator = left - 2.0 * centre + right
    if not np.isfinite(denominator) or denominator <= 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def _invalid_result(started):
    nan = math.nan
    return EstimateResult(valid=False, mode='no-estimate', omega_n=nan, zeta=nan, amp=nan,
                          score=nan, rmse=nan, cond=nan,
                          wall_time_s=time.perf_counter() - started)


def estimate(t, y, config: EstimatorConfig = EstimatorConfig()) -> EstimateResult:
    """Identify (wn, zeta) from one window of samples."""
    started = time.perf_counter()
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    finite = np.isfinite(t) & np.isfinite(y)
    t, y = t[finite], y[finite]
    order = np.argsort(t, kind='stable')
    t, y = t[order], y[order]
    n = t.size
    if n < 8:
        return _invalid_result(started)
    t = t - t[0]
    window_s = float(t[-1])
    if window_s <= 0.0:
        return _invalid_result(started)

    cfg = config
    dt = float(np.median(np.diff(t)))
    omega_min = cfg.omega_min or 2.0 * math.pi / (4.0 * window_s)
    omega_max = cfg.omega_max or math.pi / dt
    if omega_max <= omega_min:
        return _invalid_result(started)
    y_scale = float(np.max(np.abs(y))) or 1.0
    scorer = _Scorer(cfg, y_scale, float(np.ptp(y)))
    t_scale = window_s

    tg, yg = _decimate(t, y, cfg.grid_max_samples)

    if cfg.objective not in ('rmse', 'nrmse'):
        raise ValueError(f'unknown objective {cfg.objective!r}')

    def grid_scores(wn, z):
        coefs, rmse, amp, cond = _grid_fit(tg, yg, wn, z, t_scale, cfg.chunk_elems)
        score = scorer.score(rmse, amp)
        rank = rmse if cfg.objective == 'rmse' else score
        return np.where(scorer.valid(score, rmse, amp, cond), rank, np.inf)

    # Stage 1a: periodogram seeds, scored by the zeta = 0 fit.
    bin_width = 2.0 * math.pi / window_s
    seeds = []
    for seed in _periodogram_seeds(t, y, omega_min, omega_max, cfg.coarse_peaks):
        local = np.linspace(max(omega_min, seed - bin_width),
                            min(omega_max, seed + bin_width), 41)
        scores = grid_scores(local, np.zeros_like(local))
        i = int(np.argmin(scores))
        if np.isfinite(scores[i]):
            seeds.append((float(scores[i]), float(local[i])))
    seed_score, seed_omega = min(seeds) if seeds else (math.inf, math.nan)

    # Stage 1b: coarse log grid over (wn, zeta).
    count = int(np.clip(math.ceil(math.log(omega_max / omega_min) / cfg.coarse_log_step),
                        120, cfg.coarse_max_omega_count))
    coarse_w = np.geomspace(omega_min, omega_max, count)
    coarse_z = np.linspace(0.0, cfg.zeta_max, cfg.coarse_zeta_count)
    cw, cz = np.meshgrid(coarse_w, coarse_z, indexing='ij')
    coarse_scores = grid_scores(cw.ravel(), cz.ravel())
    i = int(np.argmin(coarse_scores))
    if np.isfinite(coarse_scores[i]) and coarse_scores[i] < seed_score:
        coarse_omega = float(cw.ravel()[i])
    elif np.isfinite(seed_score):
        coarse_omega = seed_omega
    else:
        return _invalid_result(started)

    # Stage 2: fine 2-D grid around the coarse value.
    fine_w = np.linspace(max(omega_min, (1.0 - cfg.fine_omega_span) * coarse_omega),
                         min(omega_max, (1.0 + cfg.fine_omega_span) * coarse_omega),
                         cfg.fine_omega_count)
    fine_z = np.linspace(0.0, cfg.zeta_max, cfg.fine_zeta_count)
    fw, fz = np.meshgrid(fine_w, fine_z, indexing='ij')
    fine_scores = grid_scores(fw.ravel(), fz.ravel()).reshape(fw.shape)
    step_w = float(fine_w[1] - fine_w[0])
    step_z = float(fine_z[1] - fine_z[0])
    starts = []
    if np.isfinite(fine_scores).any():
        iw, iz = np.unravel_index(int(np.argmin(fine_scores)), fine_scores.shape)
        grid_omega, grid_zeta = float(fine_w[iw]), float(fine_z[iz])
        # Stage 3a: parabolic interpolation along each axis.
        offset_w = (_parabolic_offset(*fine_scores[iw - 1:iw + 2, iz])
                    if 0 < iw < fine_w.size - 1 else 0.0)
        offset_z = (_parabolic_offset(*fine_scores[iw, iz - 1:iz + 2])
                    if 0 < iz < fine_z.size - 1 else 0.0)
        starts.append((grid_omega + offset_w * step_w,
                       float(np.clip(grid_zeta + offset_z * step_z, 0.0, cfg.zeta_max))))
    starts.extend((omega, 0.0) for _, omega in seeds)
    if not starts:
        return _invalid_result(started)

    # Stage 3b: polish the objective over (log wn, zeta).
    tp, yp = _decimate(t, y, cfg.polish_max_samples)

    def objective(x):
        wn, z = math.exp(x[0]), x[1]
        if not (0.0 <= z <= cfg.zeta_max):
            return 1.0e6
        _, rmse, amp, cond = _direct_fit(tp, yp, wn, z, t_scale)
        score = rmse / max(amp, scorer.amp_floor)
        ok = bool(scorer.valid(np.array(score), np.array(rmse), np.array(amp),
                               np.array(cond)))
        rank = rmse / y_scale if cfg.objective == 'rmse' else score
        return rank if ok else 1.0e6

    def projected_residual(x):
        wn, z = math.exp(x[0]), x[1]
        coef, *_ = _direct_fit(tp, yp, wn, z, t_scale)
        envelope = np.exp(-z * wn * tp)
        phase = wn * math.sqrt(max(1.0 - z * z, 0.0)) * tp
        fit = (coef[0] + coef[1] * tp / t_scale
               + envelope * (coef[2] * np.cos(phase) + coef[3] * np.sin(phase)))
        return (yp - fit) / y_scale

    log_bounds = (math.log(omega_min), math.log(omega_max))
    candidates = []
    for wn0, z0 in starts:
        x0 = np.array([math.log(wn0), z0])
        if cfg.objective == 'rmse':
            # Variable projection: the linear coefficients are solved again at
            # every (wn, zeta), so only the two nonlinear parameters iterate.
            # The trust region reflective method respects the zeta = 0 bound.
            x0 = np.clip(x0, [log_bounds[0], 1.0e-9], [log_bounds[1], cfg.zeta_max - 1.0e-9])
            result = least_squares(
                projected_residual, x0, method='trf',
                bounds=([log_bounds[0], 0.0], [log_bounds[1], cfg.zeta_max]),
                x_scale=[min(0.05, 0.25 * bin_width / wn0), 0.02],
                xtol=1.0e-14, ftol=1.0e-15, gtol=1.0e-15, max_nfev=300)
            x0 = result.x
        else:
            dlog = min(0.05, 0.25 * bin_width / wn0)
            dz = 0.02 if z0 + 0.02 <= cfg.zeta_max else -0.02
            simplex = np.array([x0, x0 + [dlog, 0.0], x0 + [0.0, dz]])
            result = minimize(
                objective, x0, method='Nelder-Mead',
                bounds=[(math.log(omega_min), math.log(omega_max)), (0.0, cfg.zeta_max)],
                options=dict(initial_simplex=simplex, xatol=1.0e-11, fatol=1.0e-15,
                             maxiter=2000, maxfev=2000))
            x0 = result.x
        candidates.append((objective(x0), math.exp(x0[0]), float(x0[1])))
    score_best, omega_n, zeta = min(candidates)
    if score_best >= 1.0e6:
        return _invalid_result(started)

    coef, rmse, amp, cond = _direct_fit(t, y, omega_n, zeta, t_scale)
    score = rmse / max(amp, scorer.amp_floor)
    valid = bool(scorer.valid(np.array(score), np.array(rmse), np.array(amp), np.array(cond)))
    overshoot = math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))
    if not valid:
        mode = 'no-estimate'
    elif overshoot < cfg.mp_min:
        mode = 'non-oscillatory'
    else:
        mode = 'oscillatory'
    return EstimateResult(valid=valid, mode=mode, omega_n=omega_n, zeta=zeta, amp=amp,
                          score=score, rmse=rmse, cond=cond,
                          wall_time_s=time.perf_counter() - started)


class OnlineEstimator:
    """Run :func:`estimate` periodically during the identification window.

    Samples are kept for the last ``history_s`` seconds. Updates start at
    ``after_s``, are at least ``update_period_s`` apart, and need ``min_samples``
    samples and ``min_p2p`` peak to peak. The latest valid estimate at or before
    ``deadline_s`` is used. ``max_samples`` keeps only the first samples of each
    window.
    """

    def __init__(self, config: EstimatorConfig = EstimatorConfig(),
                 update_period_s: float = 0.10, after_s: float = 0.8,
                 min_samples: int = 80, min_p2p: float = 0.0,
                 history_s: float = 4.2, deadline_s: float | None = None,
                 max_samples: int | None = None):
        self.config = config
        self.update_period_s = update_period_s
        self.after_s = after_s
        self.min_samples = min_samples
        self.min_p2p = min_p2p
        self.history_s = history_s
        self.deadline_s = deadline_s
        self.max_samples = max_samples

    def update_indices(self, t) -> list[int]:
        """Sample indices at which updates run. The schedule depends only on time."""
        end = t[-1] if self.deadline_s is None else min(t[-1], self.deadline_s)
        indices, last_update = [], -math.inf
        for i, now in enumerate(t):
            if now > end:
                break
            if now < self.after_s or now - last_update < self.update_period_s:
                continue
            last_update = now
            indices.append(i)
        return indices

    def run(self, t, y, prefilter=None):
        """Return (estimate, time) of the latest valid update, or (None, nan).

        Updates are evaluated from the deadline backward and the first valid one
        is returned. ``prefilter(tw, yw) -> yw`` is applied to each update's
        window, so a non-causal filter never sees samples after that update.
        """
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        for i in reversed(self.update_indices(t)):
            now = float(t[i])
            lo = int(np.searchsorted(t, max(now - self.history_s, 0.0), side='left'))
            tw, yw = t[lo:i + 1], y[lo:i + 1]
            if self.max_samples is not None:
                tw, yw = tw[:self.max_samples], yw[:self.max_samples]
            if tw.size < self.min_samples or np.ptp(yw) < self.min_p2p:
                continue
            if prefilter is not None:
                yw = prefilter(tw, yw)
            est = estimate(tw, yw, self.config)
            if est.valid:
                return est, now
        return None, math.nan


def crane_zero_zeta_grid(times, swing, t_min=0.45, t_max=1.45, grid_count=240,
                         min_amp=1.0, max_score=1.5):
    """The crane's online identifier: zeta = 0 least squares on a grid of half periods.

    The grid is over half periods (s), omega = pi/period. Returns a dict, or
    ``None`` when no candidate passes the gates. ``cond`` is sqrt(cond(X^T X)).
    """
    times = np.asarray(times, dtype=float)
    swing = np.asarray(swing, dtype=float)
    tc = times - float(times[0])
    periods = np.linspace(t_min, t_max, max(5, grid_count))
    omegas = math.pi / periods
    phase = omegas[:, None] * tc[None, :]
    design = np.empty((len(periods), len(tc), 4), dtype=float)
    design[:, :, 0] = 1.0
    design[:, :, 1] = tc
    design[:, :, 2] = np.cos(phase)
    design[:, :, 3] = np.sin(phase)
    try:
        xtx = np.einsum('gni,gnj->gij', design, design, optimize=True)
        xty = np.einsum('gni,n->gi', design, swing, optimize=True)
        coefs = np.linalg.solve(xtx, xty)
        pred = np.einsum('gni,gi->gn', design, coefs, optimize=True)
        rmse = np.sqrt(np.mean((swing[None, :] - pred) ** 2, axis=1))
        amp = np.hypot(coefs[:, 2], coefs[:, 3])
        score = rmse / np.maximum(amp, 1.0e-9)
        cond = np.sqrt(np.linalg.cond(xtx))
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return None
    valid = (np.isfinite(score) & np.isfinite(rmse) & np.isfinite(amp)
             & np.isfinite(cond) & (amp >= min_amp) & (score <= max_score))
    if not np.any(valid):
        return None
    best = int(np.flatnonzero(valid)[np.argmin(score[valid])])
    return dict(score=float(score[best]), T=float(periods[best]),
                omega=float(omegas[best]), cond=float(cond[best]),
                rmse=float(rmse[best]), amp=float(amp[best]))
