"""Einseitiger CUSUM auf einer abhängigen Reihe: zweidimensionale Zellenkette (Zustand = (CUSUM-Zelle, letzter Wert der Reihe)).

S_t = max(0, S_{t-1} + z_t − k), Alarm bei S_t ≥ h, z_t = a·(latenter Wert) + b (a > 0). CUSUM-Zellen nach Brook/Evans: Zelle 0 = [0, w/2) mit Vertreter 0, Zelle i =
[(i − ½)w, (i + ½)w) mit Vertreter i·w, w = h/(m − ½). Latente Zellen j (Kanten `edges`, Vertreter `centers`): die Masse des nächsten Wertes wird **exakt** auf die Schnittmengen
(latente Zelle) × (CUSUM-Zelle) verteilt; nur der Vertreter des *letzten* Wertes ist eine Mittelpunktsnäherung. Fehler der Ordnung (Zellbreite)²: je zwei Gitter (m = 16 und 32)
und Richardson (4·A(32) − A(16))/3. Rechenzeit und Genauigkeit stehen in der README; bei großem h (≥ 10) ist der Gitterfehler größer (bis rund 2 %).

Die Kerne: AR(1) (auch mit Verschiebung des Prozessmittels) und die Lindley-Kette der M/M/1-Wartezeit (`abw_queue.py`)."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.special import ndtr

LAT_RANGE = 6.0       # latenter Bereich ±6 (Randzellen reichen bis ±∞)
N_LAT = 128           # latente Zellen der Tabellenrechnung
M_PAIR = (16, 32)     # CUSUM-Zellen der beiden Gitter (Richardson)


# ---------------------------------------------------------------------------------------------------------------- Kerne
def ar1_kernel(phi: float, drift: float = 0.0):
    """Verteilungsfunktion des nächsten Wertes y an den Stellen `y` (Vektor), gegeben die letzten Werte `c` (Vektor): Matrix [j, r] = F(y_r | c_j)."""
    s = math.sqrt(1.0 - phi * phi)

    def f(y, centers):
        y = np.asarray(y, float)
        yy = np.where(np.isfinite(y), y, 0.0)
        out = ndtr((yy[None, :] - phi * np.asarray(centers)[:, None] - drift) / s)
        out = np.where(np.isneginf(y)[None, :], 0.0, out)
        return np.where(np.isposinf(y)[None, :], 1.0, out)
    return f


def lat_grid(n: int = N_LAT, lr: float = LAT_RANGE):
    """Kanten (Randzellen offen bis ±∞) und Vertreter (Mittelpunkte der nominellen Zellen) des latenten Gitters."""
    e = np.linspace(-lr, lr, n + 1)
    c = 0.5 * (e[:-1] + e[1:])
    e = e.copy()
    e[0], e[-1] = -np.inf, np.inf
    return e, c


# ---------------------------------------------------------------------------------------------------------------- Kette
def build_chain(kernel, stat_cdf, k: float, h: float, m: int, edges: np.ndarray, centers: np.ndarray, a_obs: float = 1.0, b_obs: float = 0.0,
                prune: float = 1e-17):
    """Übergangsmatrix Q (dünn besetzt, (m·n)×(m·n), Zustand = i·n + j) ohne Alarm; pi0 = stationäre Masse je latente Zelle (Verteilung des Wertes vor der ersten Beobachtung)."""
    n = len(centers)
    w = h / (m - 0.5)
    v = w * np.arange(m)
    rows_all, cols_all, vals_all = [], [], []
    row_j = np.arange(n)
    qs = np.arange(1, m + 1)
    for i in range(m):
        thr_lat = ((qs - 0.5) * w - v[i] + k - b_obs) / a_obs           # Schwellen der CUSUM-Zellen q = 1 … m (q = m: Alarm), aufsteigend
        pts = np.unique(np.concatenate([edges, thr_lat]))
        lo_r, hi_r = pts[:-1], pts[1:]
        mid = np.where(np.isfinite(lo_r) & np.isfinite(hi_r), 0.5 * (lo_r + hi_r), np.where(np.isfinite(lo_r), lo_r + 1.0, hi_r - 1.0))
        l_idx = np.searchsorted(edges, mid, side="right") - 1
        s_next = v[i] + a_obs * mid + b_obs - k
        q_idx = np.where(s_next < 0.5 * w, 0, np.floor(s_next / w + 0.5).astype(int))
        keep = (l_idx >= 0) & (l_idx < n) & (s_next < h - 1e-12)
        mass = np.maximum(np.diff(kernel(pts, centers), axis=1), 0.0)     # n × (Zahl der Intervalle)
        for r in np.nonzero(keep)[0]:
            col = int(q_idx[r]) * n + int(l_idx[r])
            mr = mass[:, r]
            nz = mr > prune
            if nz.any():
                rows_all.append(i * n + row_j[nz])
                cols_all.append(np.full(int(nz.sum()), col))
                vals_all.append(mr[nz])
    q = sp.coo_matrix((np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))), shape=(m * n, m * n)).tocsr()
    ee = np.where(np.isneginf(edges), -1e300, np.where(np.isposinf(edges), 1e300, edges))
    cdf = np.asarray(stat_cdf(ee))
    cdf = np.where(np.isneginf(edges), 0.0, np.where(np.isposinf(edges), 1.0, cdf))
    return q, np.diff(cdf)


def chain_arl(q: sp.csr_matrix, pi0: np.ndarray, n: int) -> float:
    """ARL aus S = 0 (Zelle 0), erster Wert stationär: π₀ · g[Zelle 0], g = (I − Q)⁻¹ 1."""
    size = q.shape[0]
    g = spl.spsolve((sp.identity(size, format="csc") - q.tocsc()), np.ones(size))
    arl = float(pi0 @ g[:n])
    return arl if arl >= 1.0 else float("nan")        # eine ARL unter 1 ist unmöglich: bei riesigen ARL (h sehr groß, grobes Gitter) verliert die Lösung ihre Genauigkeit


def chain_distribution(q: sp.csr_matrix, pi0: np.ndarray, n: int, nmax: int, stop_below: float | None = None, min_steps: int = 0) -> np.ndarray:
    """P(T > t), t = 0 … nmax (bricht ab, wenn der Wert unter `stop_below` fällt und mindestens `min_steps` Schritte gerechnet sind; der Rest bleibt NaN)."""
    v = np.ones(q.shape[0])
    surv = np.full(nmax + 1, np.nan)
    surv[0] = 1.0
    for t in range(1, nmax + 1):
        v = q @ v
        surv[t] = float(pi0 @ v[:n])
        if stop_below is not None and surv[t] < stop_below and t >= min_steps:
            break
    return surv


# ---------------------------------------------------------------------------------------------------------------- AR(1)
def ar1_arl(phi: float, h: float, k: float = 0.5, kind: str = "none", delta: float = 0.0, ms=M_PAIR, n: int = N_LAT, lr: float = LAT_RANGE,
            richardson: bool = True) -> float:
    """ARL des einseitigen CUSUM auf AR(1) (zweidimensionale Kette, Richardson über ms). kind: "none", "offset" (Messversatz δ ab t = 1; die erste
    Beobachtung ist u₁ + δ) oder "process" (Verschiebung des Prozessmittels um δ: x_t = φ·x_{t-1} + (1 − φ)·δ + s·ε_t, erste Beobachtung mit Mittel (1 − φ)·δ)."""
    drift = (1.0 - phi) * delta if kind == "process" else 0.0
    kern = ar1_kernel(phi, drift)
    edges, cen = lat_grid(n, lr)
    # Zustand = letzter Wert vor der nächsten Beobachtung; sein Startwert (vor der ersten Beobachtung) ist in allen drei Fällen der alte stationäre Wert N(0, 1):
    # bei "offset" liegt der Versatz im Beobachtungsmaß (b_obs), bei "process" im Übergang (Drift).
    b_obs = delta if kind == "offset" else 0.0
    arls = []
    for m in (ms if richardson else ms[:1]):
        q, pi0 = build_chain(kern, ndtr, k, h, m, edges, cen, 1.0, b_obs)
        arls.append(chain_arl(q, pi0, len(cen)))
    val = (4.0 * arls[1] - arls[0]) / 3.0 if richardson else arls[0]
    return val if val >= 1.0 else float("nan")      # unmöglich kleine Zahl (grobes Gitter bei riesiger ARL): als nicht gerechnet melden


def ar1_distribution(phi: float, h: float, k: float = 0.5, m: int = 24, n: int = N_LAT, nmax: int = 3000) -> dict:
    """Lauflängenverteilung aus der Kette (m, n): P(T ≤ 100), Median, 5-%-Quantil (Gitterfehler rund 0,001 in der Verteilungsfunktion)."""
    edges, cen = lat_grid(n)
    q, pi0 = build_chain(ar1_kernel(phi), ndtr, k, h, m, edges, cen)
    surv = chain_distribution(q, pi0, len(cen), nmax, stop_below=0.02, min_steps=101)
    cdf = 1.0 - surv
    return dict(p100=float(cdf[100]), median=first_reach(cdf, 0.5), q05=first_reach(cdf, 0.05), cdf=cdf)


def first_reach(cdf: np.ndarray, level: float):
    """Kleinstes t mit P(T ≤ t) ≥ level, oder None, wenn die Kurve das Niveau (im gerechneten Bereich) nicht erreicht."""
    hit = np.nan_to_num(cdf, nan=-1.0) >= level
    return int(hit.argmax()) if hit.any() else None
