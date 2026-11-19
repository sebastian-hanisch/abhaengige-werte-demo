"""Spielzeuge zum Nachrechnen von Hand (exakte Brüche, `fractions`): dieselbe Häufigkeit, zwei Regeln, zwei Richtungen.

* **Zwei-Zustände-Kette** (kein AR(1)): Zustand 0 = ruhig, 1 = Spitze, P(Spitze) = p, Übergänge a = p(1 − φ) (0 → 1) und b = (1 − p)(1 − φ) (1 → 0); stationär a/(a + b) = p,
  zweiter Eigenwert 1 − a − b = φ. Regel A: Alarm bei der ersten Spitze (Karte ohne Gedächtnis); Regel B: Alarm bei zwei Spitzen in Folge (drei Zustände: nach 0, nach 1, Alarm).
* **Diskrete Wartezeit** W' = W + 1 mit Wahrscheinlichkeit 1/3, sonst max(0, W − 1): stationär π_n = (1/2)^{n+1}; Lag 1 = 5/6, Lag 2 = 13/18; Alarm bei W ≥ 3: ARL 221/8.
* **Varianz einer Summe** von m AR(1)-Werten: m + 2·Σ_j (m − j)·φ^j.
Zähleinheit: Beobachtungen bis zum ersten Alarm, Start stationär, bekannte Parameter."""

from __future__ import annotations

from fractions import Fraction

import abw_skew as SK


def two_state(phi: Fraction, p: Fraction = Fraction(1, 10)) -> dict:
    """Regel A und Regel B der Zwei-Zustände-Kette (exakte Brüche)."""
    phi, p = Fraction(phi), Fraction(p)
    a = p * (1 - phi)
    b = (1 - p) * (1 - phi)
    arl_a = 1 + (1 - p) / a                      # erste Beobachtung Spitze mit p, sonst Warten mit der Rate a
    e0 = (1 + a) / (a * (1 - b))                 # weitere Beobachtungen aus dem Zustand „ruhig“; aus „Spitze“: e1 = 1 + b·e0
    e1 = 1 + b * e0
    arl_b = 1 + p * e1 + (1 - p) * e0
    return dict(phi=phi, p=p, a=a, b=b, arl_a=arl_a, arl_b=arl_b, stationary=a / (a + b), eigenvalue=1 - a - b)


def discrete_wait(threshold: int = 3) -> dict:
    """Diskrete Wartezeit (W' = W + 1 mit 1/3, sonst max(0, W − 1)): stationär geometrisch mit r = 1/2; Alarm bei W ≥ threshold (Zustände 0 … threshold − 1 ohne Alarm)."""
    pu, pd = Fraction(1, 3), Fraction(2, 3)
    r = pu / pd
    pi = [(1 - r) * r ** n for n in range(threshold)]
    a = [[Fraction(0)] * threshold for _ in range(threshold)]
    for s in range(threshold):
        a[s][s] += 1
        for step, pr in ((1, pu), (-1, pd)):
            ns = max(0, s + step)
            if ns < threshold:
                a[s][ns] -= pr
    e = SK._solve_fraction(a, [Fraction(1)] * threshold)
    arl = 1 + sum(pi[s] * e[s] for s in range(threshold))
    mean, ew2 = r / (1 - r), r * (1 + r) / (1 - r) ** 2
    var = ew2 - mean ** 2
    lag1 = (var - mean / 3) / var                                  # E[W'|n] = n − 1/3 für n ≥ 1; der Term n = 0 verschwindet im Produkt n·E[W'|n]
    pi1 = (1 - r) * r
    eww2 = pi1 * Fraction(7, 9) + (ew2 - pi1) - Fraction(2, 3) * (mean - pi1)   # E[W''|n] = n − 2/3 für n ≥ 2, 7/9 für n = 1, 4/9 für n = 0
    lag2 = (eww2 - mean ** 2) / var
    return dict(r=r, p_exceed=r ** threshold, iid_arl=1 / r ** threshold, further=e, arl=arl, ratio=arl * r ** threshold, mean=mean, var=var, lag1=lag1, lag2=lag2)


def sum_variance(m: int, phi: Fraction) -> Fraction:
    """Var(Summe von m AR(1)-Werten mit Varianz 1) = m + 2·Σ_{j=1}^{m−1} (m − j)·φ^j."""
    phi = Fraction(phi)
    return m + 2 * sum((m - j) * phi ** j for j in range(1, m))


def sum_variance_limit(phi: Fraction) -> Fraction:
    """Grenzwert von Var(Summe)/m für m → ∞: (1 + φ)/(1 − φ)."""
    phi = Fraction(phi)
    return (1 + phi) / (1 - phi)
