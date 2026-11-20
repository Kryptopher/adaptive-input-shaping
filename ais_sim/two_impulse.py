"""AIS-min: the two-impulse shaper with the earliest end time.

Shaper: K at 0, A at t1, B = 1-K-A at t2, with tau <= t1 < t2 (the estimate is
available at tau). The AIS is the member t1 = tau + T, t2 = tau + 2T. AIS-min
drops the equal spacing and minimizes t2.

The optimum fires A at t1 = tau. With phi = wd tau, b = wd t2,
g1 = e^{sigma tau}, g2 = e^{sigma t2}, u = g1 e^{-j phi} and v = g2 e^{-j b},
the cancellation K + A u + B v = 0 is linear in A for fixed b,
A = -(K + (1-K) v) / (u - v), and A real gives

    F(b) = K g1 sin(phi) - K g2 sin(b) + (1-K) g1 g2 sin(phi - b) = 0.

t2 is the smallest root b > phi with 0 < A < 1-K. Undamped (g = 1) this is
explicit: A = (1-2K) / (2 (1 - K + K cos phi)), B = 1-K-A and
wn t2 = arg(-(K + A e^{+j phi})) lifted by 2 pi to the first value > phi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

TWO_PI = 2.0 * math.pi
N_SCAN = 20000


@dataclass
class TwoImpulse:
    feasible: bool
    K: float
    A: float
    B: float
    t1: float
    t2: float
    residual: float

    def impulses(self):
        return [(0.0, self.K), (self.t1, self.A), (self.t2, self.B)]


def _residual(K, A, t1, t2, omega_n, zeta):
    """Magnitude of the residual vector divided by the sum of the term magnitudes."""
    sigma = zeta * omega_n
    omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)
    B = 1.0 - K - A
    z = (K + A * np.exp(sigma * t1 - 1j * omega_d * t1)
         + B * np.exp(sigma * t2 - 1j * omega_d * t2))
    scale = K + A * math.exp(sigma * t1) + B * math.exp(sigma * t2)
    return float(abs(z) / scale)


def min_end_two_impulse(K, tau, omega_n, zeta) -> TwoImpulse:
    """AIS-min with A at t1 = tau. Explicit when undamped, one scalar root when damped."""
    infeasible = TwoImpulse(False, K, math.nan, math.nan, math.nan, math.nan, math.nan)
    if not 0.0 < K < 0.5:
        return infeasible
    if zeta <= 0.0:
        phi = omega_n * tau
        a = (1.0 - 2.0 * K) / (2.0 * (1.0 - K + K * math.cos(phi)))
        theta = float(np.angle(-(K + a * np.exp(1j * phi))))
        theta += TWO_PI * math.ceil((phi - theta) / TWO_PI)
        if theta <= phi:
            theta += TWO_PI
        t2 = theta / omega_n
        return TwoImpulse(True, K, a, 1.0 - K - a, tau, t2,
                          _residual(K, a, tau, t2, omega_n, 0.0))
    sigma = zeta * omega_n
    omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)
    phi = omega_d * tau
    g1 = math.exp(sigma * tau)
    u = g1 * np.exp(-1j * phi)

    def F(b):
        g2 = np.exp(sigma * b / omega_d)
        return K * g1 * math.sin(phi) - K * g2 * np.sin(b) + (1.0 - K) * g1 * g2 * np.sin(phi - b)

    def amplitude(b):
        v = math.exp(sigma * b / omega_d) * np.exp(-1j * b)
        return float((-(K + (1.0 - K) * v) / (u - v)).real)

    # b = phi is a spurious root (u = v). Every feasible t2 lies within two periods.
    b = phi + np.linspace(1e-9, 2.0 * TWO_PI, N_SCAN + 1)[1:]
    f = F(b)
    for i in np.flatnonzero(f[:-1] * f[1:] <= 0.0):
        root = brentq(F, b[i], b[i + 1], xtol=1e-15, rtol=1e-15)
        a = amplitude(root)
        if 0.0 < a < 1.0 - K:
            t2 = root / omega_d
            return TwoImpulse(True, K, a, 1.0 - K - a, tau, t2,
                              _residual(K, a, tau, t2, omega_n, zeta))
    return infeasible
