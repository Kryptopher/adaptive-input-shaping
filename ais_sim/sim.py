"""Simulation: plant, shaped references, identification and residual metrics.

Plant: P(s) = wn^2 / (s^2 + 2 zeta wn s + wn^2) driven by a position reference.
A shaped reference is a sum of steps (the shaper's impulses convolved with a
unit step), so the response is the same sum of shifted analytic step
responses. This is exact at every time, including switch times between
samples, and gives the velocity for the energy metric.

Residual metrics are measured from the last impulse t_last over
[t_last, t_last + cycles * 2 pi / wn] for a unit final reference:

    A_res     = max |x(t) - 1|                   (peak residual, fraction of step)
    overshoot = max(0, max x(t) - 1)             (excursion beyond the final value)
    e_bar     = mean[ (xdot/wn)^2 / 2 + (x - 1)^2 / 2 ]  (normalized residual energy)

After the last reference change, x - 1 is the free response of the plant's
single mode, so A_res is the residual vibration amplitude. When the shaper is
disabled on a non-oscillatory plant that free response is the approach to the
final value, A_res is 1 - K, and the overshoot is the relevant number.
For the unshaped step, t_last = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ais_sim.estimator import EstimateResult, EstimatorConfig, OnlineEstimator, estimate
from ais_sim.shaper import ShaperSolution, design_shaper

# Coarser grids for the in-window updates of identify(..., light=True).
LIGHT_CONFIG = dict(coarse_zeta_count=10, coarse_max_omega_count=150, fine_omega_count=40,
                    fine_zeta_count=20, grid_max_samples=800, polish_max_samples=2000)


@dataclass(frozen=True)
class Plant:
    omega_n: float
    zeta: float

    @property
    def sigma(self) -> float:
        return self.zeta * self.omega_n

    @property
    def omega_d(self) -> float:
        return self.omega_n * math.sqrt(max(1.0 - self.zeta ** 2, 0.0))

    def step(self, t):
        """Unit step response, zero for t < 0. Valid for 0 <= zeta <= 1."""
        t = np.asarray(t, dtype=float)
        tp = np.maximum(t, 0.0)
        wn, z = self.omega_n, self.zeta
        if z < 1.0:
            wd, s = self.omega_d, self.sigma
            y = 1.0 - np.exp(-s * tp) * (np.cos(wd * tp) + s / wd * np.sin(wd * tp))
        else:
            y = 1.0 - (1.0 + wn * tp) * np.exp(-wn * tp)
        return np.where(t >= 0.0, y, 0.0)

    def step_derivative(self, t):
        """Impulse response, the derivative of :meth:`step`."""
        t = np.asarray(t, dtype=float)
        tp = np.maximum(t, 0.0)
        wn, z = self.omega_n, self.zeta
        if z < 1.0:
            y = wn / math.sqrt(1.0 - z * z) * np.exp(-self.sigma * tp) * np.sin(self.omega_d * tp)
        else:
            y = wn * wn * tp * np.exp(-wn * tp)
        return np.where(t >= 0.0, y, 0.0)


Impulses = list[tuple[float, float]]


def response(plant: Plant, impulses: Impulses, t):
    """Position and velocity for a reference that is the sum of steps ``impulses``."""
    t = np.asarray(t, dtype=float)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    for time_s, amplitude in impulses:
        x += amplitude * plant.step(t - time_s)
        v += amplitude * plant.step_derivative(t - time_s)
    return x, v


def residual_metrics(plant: Plant, impulses: Impulses, cycles: float = 10.0,
                     points_per_cycle: int = 400) -> dict:
    final = sum(a for _, a in impulses)
    t_last = max(t for t, _ in impulses)
    window = cycles * 2.0 * math.pi / plant.omega_n
    t = t_last + np.linspace(0.0, window, int(cycles * points_per_cycle) + 1)
    x, v = response(plant, impulses, t)
    error = (x - final) / final
    energy = 0.5 * (v / (plant.omega_n * final)) ** 2 + 0.5 * error ** 2
    return dict(A_res=float(np.max(np.abs(error))), overshoot=float(max(0.0, np.max(error))),
                e_bar=float(np.trapz(energy, t) / window), t_last=float(t_last))


# References: lists of (time in s, step amplitude) that sum to 1.

def unshaped_impulses() -> Impulses:
    return [(0.0, 1.0)]


def disabled_impulses(K: float, tau: float) -> Impulses:
    """Shaper not applied: K at 0, the rest of the step at tau."""
    return [(0.0, K), (tau, 1.0 - K)]


def ais_impulses(solution: ShaperSolution) -> Impulses:
    return solution.impulses()


def ft_amplitude(K: float, sigma: float, t1: float, t2: float) -> float:
    """Amplitude A at t1 for impulses at damped phase pi (t1) and 2 pi (t2).

    The imaginary part of the residual cancels at these phases, and the real
    part gives K - A e^{sigma t1} + (1-K-A) e^{sigma t2} = 0, written here
    without overflow.
    """
    return (K * math.exp(-sigma * t2) + (1.0 - K)) / (math.exp(-sigma * (t2 - t1)) + 1.0)


def ft_impulses(K: float, omega_n: float, zeta: float, tau: float, variant: str) -> Impulses:
    """FT shaper designed from (omega_n, zeta) with the estimate available at tau.

    G_FT = K + A e^{-sT} + (1-K-A) e^{-2sT}, T = pi/wd. If tau <= T the switches
    are at T and 2T.

    variant 'a': a switch whose time has passed at tau is applied at tau, with
                 the amplitudes unchanged (the FT shaper as published).
    variant 'b': both switches are delayed by the smallest whole number of damped
                 periods 2 pi m / wd that puts the first one at or after tau, and
                 the amplitudes are solved again for the shifted times.
    """
    omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)
    sigma = zeta * omega_n
    T = math.pi / omega_d
    if tau <= T:
        t1, t2, a1 = T, 2.0 * T, ft_amplitude(K, sigma, T, 2.0 * T)
    elif variant == 'a':
        t1, t2, a1 = tau, max(2.0 * T, tau), ft_amplitude(K, sigma, T, 2.0 * T)
    elif variant == 'b':
        period = 2.0 * math.pi / omega_d
        shift = math.ceil((tau - T) / period - 1e-12)
        t1, t2 = T + shift * period, 2.0 * T + shift * period
        a1 = ft_amplitude(K, sigma, t1, t2)
    else:
        raise ValueError(f'unknown FT variant {variant!r}')
    return [(0.0, K), (t1, a1), (t2, 1.0 - K - a1)]


# Identification

def online_schedule(tau: float) -> dict:
    """Online estimator timing for a window of tau seconds.

    Updates start at 0.8 s, run every 0.1 s and stop 0.1 s before tau for
    tau >= 2 s. For shorter windows every interval shrinks in proportion.
    """
    return dict(after_s=min(0.8, 0.4 * tau), update_period_s=min(0.1, tau / 20.0),
                deadline_s=tau - min(0.1, 0.05 * tau))


def sample_interval(plant: Plant, tau: float) -> float:
    """Simulation sample interval in s: at most 1 ms, 200 per period and tau/400."""
    return min(1.0e-3, 2.0 * math.pi / (plant.omega_n * 200.0), tau / 400.0)


def identify(plant: Plant, K: float, tau: float, light: bool = False,
             noise_std: float = 0.0, rng: np.random.Generator | None = None,
             max_samples: int = 20_000, prefilter=None) -> EstimateResult | None:
    """Run the online estimator on the response to a step of K over [0, tau].

    ``light`` runs the in-window updates with coarser grids and then the full
    estimator once on the window of the latest valid update. ``noise_std`` is
    in units of the response (the step is K). ``prefilter(t, y) -> y`` is
    applied to each update's window.
    """
    dt = sample_interval(plant, tau)
    t = np.arange(0.0, tau + 0.5 * dt, dt)
    y = K * plant.step(t)
    if noise_std > 0.0:
        rng = rng or np.random.default_rng()
        y = y + noise_std * rng.standard_normal(t.size)
    full = EstimatorConfig()
    config = EstimatorConfig(**LIGHT_CONFIG) if light else full
    online = OnlineEstimator(config, min_samples=80, min_p2p=0.0, history_s=max(4.2, tau),
                             max_samples=max_samples, **online_schedule(tau))
    final, final_time = online.run(t, y, prefilter=prefilter)
    if light and final is not None:
        end = int(np.searchsorted(t, final_time, side='right'))
        tw, yw = t[:end][:max_samples], y[:end][:max_samples]
        if prefilter is not None:
            yw = prefilter(tw, yw)
        final = estimate(tw, yw, full)
    return final


def ais_from_estimate(est: EstimateResult | None, K: float, tau: float):
    """Design the AIS from an estimate. Returns (impulses, solution or None, mode)."""
    if est is None or not est.valid:
        return disabled_impulses(K, tau), None, 'no-estimate'
    if est.mode == 'non-oscillatory':
        return disabled_impulses(K, tau), None, est.mode
    solution = design_shaper(K, tau, est.omega_n, est.zeta)
    if not solution.feasible:
        return disabled_impulses(K, tau), solution, 'infeasible'
    return ais_impulses(solution), solution, est.mode
