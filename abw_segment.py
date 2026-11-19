"""Rückblickende Segmentierung bei autokorreliertem Rauschen ohne jeden Bruch (Expander „Und rückblickend?“; Brücke zu Stück 7 der Reihe).

Optimale Zerlegung (dynamische Programmierung ohne Abkürzung, Jackson u. a.) für Mittelwertsprünge: Kosten einer Strecke = Summe der Abweichungsquadrate um ihr Mittel
(Gauß-Kosten mit Standardabweichung 1, bekannt), Strafe β = c·ln n je Bruch, Mindestlänge `minlen`. Gezählt werden **Reihen** (n = 200 Werte), auf denen die
Zerlegung mindestens einen Bruch findet, obwohl es keinen gibt. Zähleinheit: Reihen; Standardfehler der Anteile höchstens 2,5 Prozentpunkte bei 400 Reihen."""

from __future__ import annotations

import math

import numpy as np

import abw_normal as N

N_SERIES = 200
MIN_LEN = 5
PENALTY_FACTORS = (2, 4, 6, 10)


def _cumulative(x: np.ndarray):
    z = np.zeros((x.shape[0], 1))
    return np.concatenate([z, np.cumsum(x, 1)], 1), np.concatenate([z, np.cumsum(x * x, 1)], 1)


def optimal_partition(x: np.ndarray, betas, minlen: int):
    """Optimale Zerlegung für alle Zeilen von x (R, n) und alle Strafen `betas` zugleich.
    Rückgabe: (breaks[r][p] = sortierte Bruchstellen (Bruch bei s: neue Strecke beginnt mit dem Wert s, 0-basiert), Zielwerte (R, P))."""
    x = np.atleast_2d(x)
    runs, n = x.shape
    betas = np.asarray(betas, float)
    s1, s2 = _cumulative(x)
    f = np.full((runs, betas.size, n + 1), np.inf)
    f[:, :, 0] = -betas[None, :]
    last = np.zeros((runs, betas.size, n + 1), dtype=np.int32)
    for t in range(minlen, n + 1):
        smax = t - minlen
        s = np.arange(smax + 1)
        cost = (s2[:, t, None] - s2[:, : smax + 1]) - (s1[:, t, None] - s1[:, : smax + 1]) ** 2 / (t - s)[None, :]
        tot = f[:, :, : smax + 1] + cost[:, None, :]
        am = tot.argmin(2)
        f[:, :, t] = np.take_along_axis(tot, am[:, :, None], 2)[:, :, 0] + betas[None, :]
        last[:, :, t] = am
    out = []
    for r in range(runs):
        row = []
        for p in range(betas.size):
            t, cuts = n, []
            while t > 0:
                s_ = int(last[r, p, t])
                if s_ > 0:
                    cuts.append(s_)
                t = s_
            row.append(sorted(cuts))
        out.append(row)
    return out, f[:, :, n]


def false_breaks(phi: float, runs: int, seed: int, residuals: bool, n: int = N_SERIES, minlen: int = MIN_LEN, cs=PENALTY_FACTORS) -> dict:
    """Anteil der Reihen (in %) mit mindestens einem Bruch und mittlere Bruchzahl je Strafe c, für Rohdaten oder Residuen (φ bekannt), AR(1) ohne Bruch."""
    s = math.sqrt(1.0 - phi * phi)
    st = N.Stream(seed)
    u = np.empty((runs, n + 1))
    u[:, 0] = st.normal(runs)
    eps = st.normal((runs, n))
    for t in range(1, n + 1):
        u[:, t] = phi * u[:, t - 1] + s * eps[:, t - 1]
    x = (u[:, 1:] - phi * u[:, :-1]) / s if residuals else u[:, 1:]
    betas = [c * math.log(n) for c in cs]
    cuts, _ = optimal_partition(x, betas, minlen)
    share = [100.0 * float(np.mean([len(cuts[r][p]) >= 1 for r in range(runs)])) for p in range(len(cs))]
    mean_breaks = [float(np.mean([len(cuts[r][p]) for r in range(runs)])) for p in range(len(cs))]
    return dict(share=share, mean_breaks=mean_breaks, runs=runs, seed=seed)


def brute_force(x, beta: float, minlen: int):
    """Alle Zerlegungen mit Mindestlänge `minlen` aufzählen; Rückgabe (kleinster Zielwert, Bruchstellen). Nur für kleine n (Orakel im Test)."""
    x = np.asarray(x, float)
    n = x.size
    best = [math.inf, None]

    def objective(cuts):
        b = [0] + list(cuts) + [n]
        tot = 0.0
        for a, c in zip(b[:-1], b[1:]):
            seg = x[a:c]
            tot += float(((seg - seg.mean()) ** 2).sum())
        return tot + beta * len(cuts)

    def rec(start, chosen):
        val = objective(chosen)
        if val < best[0] - 1e-12:
            best[0], best[1] = val, list(chosen)
        for p in range(start, n - minlen + 1):
            if not chosen and p >= minlen or chosen and p - chosen[-1] >= minlen:
                rec(p + minlen, chosen + [p])
    rec(minlen, [])
    return best[0], best[1]
