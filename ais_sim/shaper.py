"""AIS shaper solvers: closed form for zeta = 0 and a numerical solver for zeta >= 0.

The adaptive input shaper is

    G(s) = K + A exp(-s(tau+T)) + B exp(-s(tau+2T)),    B = 1 - K - A.

Cancelling the pole s = -sigma + j*wd gives two real equations. Scaled by
exp(-sigma*tau) so that long damped windows do not overflow:

    R1 = K e^{-sigma tau} + A e^{sigma T} cos(wd(tau+T)) + B e^{2 sigma T} cos(wd(tau+2T)) = 0
    R2 =                    A e^{sigma T} sin(wd(tau+T)) + B e^{2 sigma T} sin(wd(tau+2T)) = 0

Undamped (sigma = 0, wd = wn), with phi = wn*tau and theta = wn*T, eliminating
A from R2 and substituting into R1 gives

    F(theta) = K [sin(phi+theta) - sin(phi+2 theta)] - (1-K) sin(theta) = 0.

With beta = tan(theta/2), F * (1+beta^2)^2 = 2 beta * P(beta) where

    P(beta) = -K sin(phi) beta^3 + (3K cos(phi) - (1-K)) beta^2
              + 3K sin(phi) beta - (K cos(phi) + (1-K)).

The beta = 0 factor is the trivial solution theta = 0. P degenerates when
sin(phi) -> 0, where one root runs to beta = infinity (theta = pi, the half
period branch). F has no such singularity and is the bracketed alternative.
A then follows from R1:

    A = -(K + (1-K) cos(phi+2 theta)) / (cos(phi+theta) - cos(phi+2 theta)).

Units: tau and T in s, omega_n in rad/s, K, A and B as fractions of the step.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, fsolve

SIN_PHI_GUARD = 1.0e-6
REAL_ROOT_TOL = 1.0e-9
UNDAMPED_RESIDUAL_TOL = 1.0e-9
DAMPED_RESIDUAL_TOL = 1.0e-10
TWO_PI = 2.0 * math.pi


@dataclass
class Candidate:
    theta: float
    T: float
    A: float
    B: float
    source: str
    residual: float
    accepted: bool
    nfev: int = 0


@dataclass
class ShaperSolution:
    feasible: bool
    K: float
    tau: float
    omega_n: float
    zeta: float
    A: float
    B: float
    T: float
    method: str
    residual: float
    nfev: int = 0
    nfev_total: int = 0
    n_distinct: int = 0
    wall_time_s: float = 0.0

    def impulses(self) -> list[tuple[float, float]]:
        """(time, amplitude) of the three shaper impulses."""
        return [(0.0, self.K), (self.tau + self.T, self.A), (self.tau + 2.0 * self.T, self.B)]


# Residuals

def undamped_residuals(K, tau, omega_n, A, T):
    """Real and imaginary cancellation residuals at s = j*wn."""
    B = 1.0 - K - A
    p1 = omega_n * (tau + T)
    p2 = omega_n * (tau + 2.0 * T)
    return (K + A * math.cos(p1) + B * math.cos(p2), A * math.sin(p1) + B * math.sin(p2))


def damped_residuals(K, tau, omega_n, zeta, A, T, normalized=False):
    """R1 and R2 above, scaled by exp(-sigma*tau).

    With ``normalized`` the residuals are divided by the sum of the term
    magnitudes, the scale the convergence tolerance is applied to.
    """
    sigma = zeta * omega_n
    omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)
    B = 1.0 - K - A
    with np.errstate(over='ignore', invalid='ignore'):
        k_term = K * math.exp(-sigma * tau)
        e1 = math.exp(min(sigma * T, 700.0))
        e2 = math.exp(min(2.0 * sigma * T, 700.0))
    p1 = omega_d * (tau + T)
    p2 = omega_d * (tau + 2.0 * T)
    r1 = k_term + A * e1 * math.cos(p1) + B * e2 * math.cos(p2)
    r2 = A * e1 * math.sin(p1) + B * e2 * math.sin(p2)
    if normalized:
        scale = abs(k_term) + abs(A) * e1 + abs(B) * e2
        scale = scale if scale > 0.0 else 1.0
        return r1 / scale, r2 / scale
    return r1, r2


def transcendental(theta, K, phi):
    """F(theta), the undamped condition with A eliminated."""
    return K * (np.sin(phi + theta) - np.sin(phi + 2.0 * theta)) - (1.0 - K) * np.sin(theta)


def transcendental_derivative(theta, K, phi):
    return (K * (np.cos(phi + theta) - 2.0 * np.cos(phi + 2.0 * theta))
            - (1.0 - K) * np.cos(theta))


def cubic_coefficients(K, phi):
    """Coefficients (a, b, c, d) of P(beta), highest power first."""
    s, c = math.sin(phi), math.cos(phi)
    return (-K * s, 3.0 * K * c - (1.0 - K), 3.0 * K * s, -(K * c + (1.0 - K)))


# Closed form

def cardano_roots(a, b, c, d) -> np.ndarray:
    """All three roots of a*x^3 + b*x^2 + c*x + d in complex arithmetic.

    Depressed form x^3 - 3 B x - 2 C = 0 (x = beta + b/(3a)). With
    U^3 = C + sqrt(C^2 - B^3) and V = B/U the roots are w^k U + w^-k V.
    Complex arithmetic covers the case of three real roots, C^2 < B^3. Each
    root gets two Newton steps on the original cubic.
    """
    bn, cn, dn = b / a, c / a, d / a
    p = cn - bn * bn / 3.0
    q = 2.0 * bn ** 3 / 27.0 - bn * cn / 3.0 + dn
    big_b = complex(-p / 3.0)
    big_c = complex(-q / 2.0)
    root = np.sqrt(big_c * big_c - big_b ** 3)
    if abs(big_c - root) > abs(big_c + root):
        root = -root
    u = (big_c + root) ** (1.0 / 3.0)
    if u == 0:
        x = np.zeros(3, dtype=complex)
    else:
        v = big_b / u
        w = complex(-0.5, math.sqrt(3.0) / 2.0)
        x = np.array([u + v, w * u + w.conjugate() * v, w.conjugate() * u + w * v])
    roots = x - bn / 3.0
    for _ in range(2):
        value = ((a * roots + b) * roots + c) * roots + d
        slope = (3.0 * a * roots + 2.0 * b) * roots + c
        step = np.where(np.abs(slope) > 0.0, value / np.where(slope == 0, 1, slope), 0.0)
        roots = roots - step
    return roots


def cubic_discriminant_sign(K, phi) -> float:
    """C^2 - B^3 of the depressed cubic. Negative means three real roots.

    Returns +inf inside the sin(phi) guard, where the cubic degenerates.
    """
    a, b, c, d = cubic_coefficients(K, phi)
    if abs(math.sin(phi)) < SIN_PHI_GUARD:
        return math.inf
    bn, cn, dn = b / a, c / a, d / a
    p = cn - bn * bn / 3.0
    q = 2.0 * bn ** 3 / 27.0 - bn * cn / 3.0 + dn
    return (q / 2.0) ** 2 + (p / 3.0) ** 3


def _guard_branch_roots(theta, K, phi):
    """Refine a root on the half period branch, bracketing locally near a multiple root.

    At phi = 2 pi j with K = 1/4, F = -sin(theta)(1 + cos(theta))/2 has a
    triple root at theta = pi (F' = F'' = 0). An offset delta in phi splits it
    into roots about delta**(1/3) apart, which Newton from pi cannot reach.
    """
    if abs(transcendental_derivative(theta, K, phi)) > 1.0e-6:
        return [_newton_theta(theta, K, phi)]
    local = transcendental_roots(K, phi, n_scan=400, lo=theta - 0.05, hi=theta + 0.05)
    return local or [theta]


def _newton_theta(theta, K, phi, iterations=3):
    for _ in range(iterations):
        slope = transcendental_derivative(theta, K, phi)
        if abs(slope) < 1.0e-14:
            break
        theta = theta - transcendental(theta, K, phi) / slope
    return float(theta)


def _evaluate_theta(theta, K, tau, omega_n, source) -> list[Candidate]:
    """Expand one root theta into T = (theta + 2 pi k)/wn, k = -3..3, and screen each."""
    phi = omega_n * tau
    out = []
    for k in range(-3, 4):
        T = (theta + TWO_PI * k) / omega_n
        if T <= 0.0:
            continue
        c1 = math.cos(phi + omega_n * T)
        c2 = math.cos(phi + 2.0 * omega_n * T)
        denominator = c1 - c2
        if abs(denominator) < 1.0e-12:
            out.append(Candidate(theta, T, math.nan, math.nan, source, math.inf, False))
            continue
        A = -(K + (1.0 - K) * c2) / denominator
        B = 1.0 - K - A
        residual = max(abs(r) for r in undamped_residuals(K, tau, omega_n, A, T))
        accepted = A > 0.0 and B > 0.0 and residual < UNDAMPED_RESIDUAL_TOL
        out.append(Candidate(theta, T, A, B, source, residual, accepted))
    return out


def closed_form_thetas(K, phi) -> list[tuple[float, str]]:
    """Real roots theta from the cubic, or from the half period branch when sin(phi) ~ 0.

    Returns the roots with their source label.
    """
    sin_phi = math.sin(phi)
    if abs(sin_phi) < SIN_PHI_GUARD:
        # At sin(phi) = 0, F = sin(theta) * [K c_phi (1 - 2 cos theta) - (1-K)]:
        # theta = pi, plus cos(theta) = (1 - (1-K)/(K c_phi))/2 when that is real.
        c_phi = math.copysign(1.0, math.cos(phi))
        thetas = [(root, 'half-period') for root in _guard_branch_roots(math.pi, K, phi)]
        target = 0.5 * (1.0 - (1.0 - K) / (K * c_phi))
        if -1.0 <= target <= 1.0:
            for value in (math.acos(target), -math.acos(target)):
                thetas.extend((root, 'half-period-pair')
                              for root in _guard_branch_roots(value, K, phi))
        return thetas
    thetas = []
    for beta in cardano_roots(*cubic_coefficients(K, phi)):
        if abs(beta.imag) > REAL_ROOT_TOL * max(1.0, abs(beta.real)):
            continue
        thetas.append((_newton_theta(2.0 * math.atan(beta.real), K, phi), 'cubic'))
    return thetas


def transcendental_roots(K, phi, n_scan=2000, lo=1.0e-9, hi=TWO_PI - 1.0e-9) -> list[float]:
    """All roots of F on (lo, hi), by a sign change scan plus brentq.

    Two roots inside one scan interval show up as an extremum of F without a
    sign change. Those are split at the extremum (found by brentq on F') and
    bracketed separately. A root of multiplicity k is determined only to about
    eps**(1/k) by any method.
    """
    grid = np.linspace(lo, hi, n_scan + 1)
    f = transcendental(grid, K, phi)
    d = transcendental_derivative(grid, K, phi)
    roots = [float(x) for x in grid[:-1][f[:-1] == 0.0]]
    sign_change = f[:-1] * f[1:] < 0.0
    hidden_pair = (~sign_change) & (d[:-1] * d[1:] < 0.0) & (f[:-1] != 0.0)
    for i in np.flatnonzero(sign_change | hidden_pair):
        a, b = grid[i], grid[i + 1]
        if sign_change[i]:
            roots.append(brentq(transcendental, a, b, args=(K, phi), xtol=1e-15, rtol=1e-15))
        else:
            extremum = brentq(transcendental_derivative, a, b, args=(K, phi),
                              xtol=1e-15, rtol=1e-15)
            fe = transcendental(extremum, K, phi)
            if fe * f[i] < 0.0:
                roots.append(brentq(transcendental, a, extremum, args=(K, phi),
                                    xtol=1e-15, rtol=1e-15))
                roots.append(brentq(transcendental, extremum, b, args=(K, phi),
                                    xtol=1e-15, rtol=1e-15))
            elif abs(fe) < 1.0e-14:
                roots.append(float(extremum))
    return sorted(roots)


def _pick(candidates):
    accepted = [c for c in candidates if c.accepted]
    return min(accepted, key=lambda c: c.T) if accepted else None


def _solution(K, tau, omega_n, best, method, started):
    if best is None:
        return ShaperSolution(
            feasible=False, K=K, tau=tau, omega_n=omega_n, zeta=0.0, A=math.nan,
            B=math.nan, T=math.nan, method=method, residual=math.nan,
            wall_time_s=time.perf_counter() - started)
    return ShaperSolution(
        feasible=True, K=K, tau=tau, omega_n=omega_n, zeta=0.0, A=best.A, B=best.B,
        T=best.T, method=method, residual=best.residual,
        wall_time_s=time.perf_counter() - started)


def solve_undamped_brentq(K, tau, omega_n) -> ShaperSolution:
    """Undamped AIS from the bracketed roots of F(theta), smallest feasible T."""
    started = time.perf_counter()
    candidates = []
    for theta in transcendental_roots(K, omega_n * tau):
        candidates.extend(_evaluate_theta(theta, K, tau, omega_n, 'brentq'))
    return _solution(K, tau, omega_n, _pick(candidates), 'brentq', started)


def solve_undamped(K, tau, omega_n) -> ShaperSolution:
    """Closed form undamped AIS, smallest feasible T.

    Real cubic roots (or the half period branch when |sin phi| < SIN_PHI_GUARD),
    then T = (2 atan(beta) + 2 pi k)/wn for k = -3..3. A candidate is kept if
    T > 0, A > 0, 1-K-A > 0 and the residual of R1, R2 is below 1e-9. The
    smallest such T is returned.
    """
    started = time.perf_counter()
    if not (0.0 < K < 1.0) or omega_n <= 0.0 or tau < 0.0:
        raise ValueError('need 0 < K < 1, omega_n > 0, tau >= 0')
    candidates = []
    for theta, source in closed_form_thetas(K, omega_n * tau):
        candidates.extend(_evaluate_theta(theta, K, tau, omega_n, source))
    best = _pick(candidates)
    method = 'half-period' if best is not None and best.source.startswith('half') \
        else 'closed-form'
    return _solution(K, tau, omega_n, best, method, started)


# Damped numerical solver

def solve_damped(K, tau, omega_n, zeta) -> ShaperSolution:
    """Damped AIS from R1 = R2 = 0 with scipy's MINPACK hybrd (fsolve).

    Initial points: the undamped closed form for the same (K, tau, wn), then
    T0 = m*pi/(2 wd), A0 = (1-K)/2 for m = 1..8. A converged point is accepted
    when 0 < A < 1-K, T > 0 and the normalized residual is below 1e-10. Every
    initial point is run, distinct solutions (within 1e-6 in A and T) are
    counted, and the one with the smallest T is returned.
    """
    started = time.perf_counter()
    if not (0.0 < K < 1.0) or omega_n <= 0.0 or tau < 0.0 or not (0.0 <= zeta < 1.0):
        raise ValueError('need 0 < K < 1, omega_n > 0, tau >= 0, 0 <= zeta < 1')
    omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)

    def equations(x):
        return damped_residuals(K, tau, omega_n, zeta, x[0], x[1])

    inits = []
    warm = solve_undamped(K, tau, omega_n)
    if warm.feasible:
        inits.append(('undamped closed form', warm.A, warm.T))
    inits.extend((f'm={m}', 0.5 * (1.0 - K), m * math.pi / (2.0 * omega_d)) for m in range(1, 9))

    distinct = []
    nfev_total = 0
    for label, a0, t0 in inits:
        with np.errstate(all='ignore'):
            x, info, ier, message = fsolve(equations, [a0, t0], full_output=True, xtol=1.0e-12)
        nfev_total += int(info['nfev'])
        A, T = float(x[0]), float(x[1])
        residual = (max(abs(r) for r in damped_residuals(K, tau, omega_n, zeta, A, T, True))
                    if np.all(np.isfinite(x)) else math.inf)
        accepted = 0.0 < A < 1.0 - K and T > 0.0 and residual < DAMPED_RESIDUAL_TOL
        candidate = Candidate(omega_d * T, T, A, 1.0 - K - A, f'fsolve from {label}',
                              residual, accepted, int(info['nfev']))
        if candidate.accepted and not any(
                abs(c.T - T) < 1.0e-6 and abs(c.A - A) < 1.0e-6 for c in distinct):
            distinct.append(candidate)

    best = min(distinct, key=lambda c: c.T) if distinct else None
    if best is None:
        return ShaperSolution(
            feasible=False, K=K, tau=tau, omega_n=omega_n, zeta=zeta, A=math.nan,
            B=math.nan, T=math.nan, method='fsolve', residual=math.nan,
            nfev_total=nfev_total, wall_time_s=time.perf_counter() - started)
    return ShaperSolution(
        feasible=True, K=K, tau=tau, omega_n=omega_n, zeta=zeta, A=best.A, B=best.B,
        T=best.T, method='fsolve', residual=best.residual, nfev=best.nfev,
        nfev_total=nfev_total, n_distinct=len(distinct),
        wall_time_s=time.perf_counter() - started)


def design_shaper(K, tau, omega_n, zeta) -> ShaperSolution:
    """Design rule: the numerical solver for every zeta >= 0, started from the
    undamped closed form (see :func:`solve_damped`)."""
    return solve_damped(K, tau, omega_n, zeta)


def solver_statistics(solutions) -> dict:
    """Success rate, median nfev, median wall time and max residual over runs."""
    solutions = list(solutions)
    ok = [s for s in solutions if s.feasible]
    return dict(
        runs=len(solutions),
        success_rate=len(ok) / len(solutions) if solutions else math.nan,
        median_nfev=float(np.median([s.nfev for s in ok])) if ok else math.nan,
        median_time_s=float(np.median([s.wall_time_s for s in ok])) if ok else math.nan,
        max_residual=float(max(s.residual for s in ok)) if ok else math.nan,
        max_distinct=max((s.n_distinct for s in ok), default=0),
    )
