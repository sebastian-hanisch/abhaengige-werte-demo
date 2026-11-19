"""Schiefe bei unabhängigen Werten (Tab „Schiefe“): dieselben Karten auf standardisierten Werten (Mittel 0, Standardabweichung 1, bekannte Parameter), zero-state.

* Shewhart 3σ zweiseitig: exakt als 1/P(|Z| > 3); Exponential e⁴, Gamma(4) e¹⁰·3/683 geschlossen, Lognormal(0; 0,5) und t(5) aus der Verteilungsfunktion.
* CUSUM einseitig (k = 0,5, h = 4,0965): Brook-Evans-Kette mit Richardson (zweiter Weg: Simulation, `tests/test_oracle_skew.py`).
* Vorzeichen-CUSUM: Schritt +1 bei einem Wert über dem Median, −2 sonst, Alarm bei C ≥ H. Bei stetiger Verteilung ist die Schrittwahrscheinlichkeit ½: die ARL₀ ist
  **exakt als Bruch** und verteilungsfrei (H = 3: 14; H = 8: 266; H = 9: 442).
Zähleinheit: Beobachtungen bis zum Alarm, unabhängige Werte, bekannte Parameter. Das Maß der Verzögerung (ARL₁) ist ν = 0, zero-state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import stats
from scipy.optimize import brentq

import abw_chain as CH

H_ONE = 4.096494807          # einseitiger CUSUM, k = 0,5, ARL₀ = 370,398 bei normalen unabhängigen Werten (abw_chain.cusum_h_for_arl)
K = 0.5


@dataclass(frozen=True)
class Dist:
    key: str
    name: str
    rv: object

    @property
    def mean(self) -> float:
        return float(self.rv.mean())

    @property
    def sd(self) -> float:
        return float(self.rv.std())

    def cdf_z(self, t):
        """Verteilungsfunktion der standardisierten Werte (Z = (X − μ)/σ)."""
        return self.rv.cdf(self.mean + self.sd * np.asarray(t, float))

    def sf_z(self, t: float) -> float:
        return float(self.rv.sf(self.mean + self.sd * t))


DISTS = {
    "normal": Dist("normal", "Normal", stats.norm()),
    "expon": Dist("expon", "Exponential", stats.expon()),
    "gamma4": Dist("gamma4", "Gamma(4)", stats.gamma(4)),
    "lognorm": Dist("lognorm", "Lognormal(0; 0,5)", stats.lognorm(s=0.5)),
    "t5": Dist("t5", "t(5)", stats.t(5)),
}


# ---------------------------------------------------------------------------------------------------------------- Shewhart
def shewhart_two_sided(d: Dist, limit: float = 3.0) -> float:
    """ARL₀ der Shewhart-Karte ±limit (standardisiert, zweiseitig) = 1/(P(Z > limit) + P(Z < −limit))."""
    p = d.sf_z(limit) + float(d.cdf_z(-limit))
    return 1.0 / p


def shewhart_one_sided(d: Dist, c: float | None = None) -> float:
    """ARL₀ der einseitigen Karte (Alarm bei Z > c, c = 2,7822 wie bei normalen Werten)."""
    return 1.0 / d.sf_z(CH.one_sided_limit() if c is None else c)


def exponential_closed() -> float:
    """e⁴: für die Exponentialverteilung (Mittel 1, Standardabweichung 1) liegt die untere Grenze −2 außerhalb des Trägers, die obere ist 4: P(X > 4) = e^{−4}."""
    return math.exp(4.0)


def gamma4_closed() -> float:
    """Gamma(4, 1): Mittel 4, Standardabweichung 2, obere Grenze 10 (untere −2 nie erreicht): P(X > 10) = e^{−10}·Σ_{j<4} 10^j/j! = e^{−10}·683/3."""
    return math.exp(10.0) * 3.0 / 683.0


def shewhart_limit_for(d: Dist, gamma: float = CH.NOM) -> float:
    """Zweiseitige Grenze ±L (in σ), bei der die ARL₀ gleich gamma ist (nachgeeicht)."""
    return float(brentq(lambda lim: shewhart_two_sided(d, lim) - gamma, 1.0, 12.0, xtol=1e-10))


# ---------------------------------------------------------------------------------------------------------------- CUSUM einseitig
def cusum_arl(d: Dist, h: float = H_ONE, k: float = K, n: int = 800, shift: float = 0.0) -> float:
    """ARL₀ (oder ARL₁ bei Lageverschiebung um `shift` σ; die Verschiebung addiert sich auf die standardisierten Werte) des einseitigen CUSUM (Brook-Evans-Kette, Richardson)."""
    return CH.cusum_chain_arl(lambda t: d.cdf_z(np.asarray(t, float) - shift), k, h, n, richardson=True)


def cusum_h_for(d: Dist, gamma: float = CH.NOM, n: int = 400) -> float:
    """Nachgeeichte Schwelle h (k = 0,5), bei der die ARL₀ des einseitigen CUSUM gleich gamma ist."""
    return float(brentq(lambda h: cusum_arl(d, h, K, n) - gamma, 0.5, 30.0, xtol=1e-6))


# ---------------------------------------------------------------------------------------------------------------- Vorzeichen-CUSUM
def _solve_fraction(a, b):
    n = len(a)
    m = [[Fraction(v) for v in row] + [Fraction(bb)] for row, bb in zip(a, b)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        m[c] = [v / m[c][c] for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] for i in range(n)]


def sign_cusum_arl(h: int, p_up=Fraction(1, 2)):
    """ARL des Vorzeichen-CUSUM aus C = 0: C' = max(0, C + 1) mit Wahrscheinlichkeit p_up, sonst max(0, C − 2); Alarm bei C' ≥ H. p_up als Fraction gibt einen Bruch, als float eine Zahl.
    Bei stetiger Verteilung und Median als Bezugswert ist p_up = ½ (verteilungsfrei)."""
    exact = isinstance(p_up, Fraction)
    one = Fraction(1) if exact else 1.0
    zero = Fraction(0) if exact else 0.0
    a = [[zero] * h for _ in range(h)]
    for s in range(h):
        a[s][s] += one
        for step, pr in ((1, p_up), (-2, one - p_up)):
            ns = max(0, s + step)
            if ns < h:
                a[s][ns] -= pr
    if exact:
        return _solve_fraction(a, [Fraction(1)] * h)[0]
    return float(np.linalg.solve(np.array(a, float), np.ones(h))[0])


def sign_up_probability(d: Dist, shift: float) -> float:
    """P(X + shift·σ liegt über dem alten Median) nach einer Lageverschiebung um `shift` Standardabweichungen (Werte in Einheiten von X)."""
    return float(d.rv.sf(float(d.rv.median()) - shift * d.sd))


def sign_cusum_shift_arl(d: Dist, shift: float, h: int) -> float:
    """ARL₁ des Vorzeichen-CUSUM nach Lageverschiebung um `shift`·σ (ν = 0, zero-state). Ist P(Schritt +1) = 1, meldet er nach genau H Beobachtungen (untere Stützstelle der Verteilung)."""
    p = sign_up_probability(d, shift)
    if p >= 1.0 - 1e-15:
        return float(h)
    return float(sign_cusum_arl(h, p))


def sign_cusum_note(d: Dist, shift: float) -> bool:
    """Wahr, wenn jeder verschobene Wert über dem alten Median liegt (Stützstelle): dann ist die Meldung nach H Beobachtungen ein Artefakt der Verteilung, keine Stärke der Karte."""
    return sign_up_probability(d, shift) >= 1.0 - 1e-15
