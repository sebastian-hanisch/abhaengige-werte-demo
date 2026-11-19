"""Wartezeit-Reihe am Gate: M/M/1 (exakt) und das Gate M/M/2/6 (Simulation, Randverteilung exakt).

M/M/1: Lindley-Kette W' = max(0, W + S − A), Abfertigung μ = 1, Ankünfte λ = ρ, stationär (Atom 1 − ρ bei 0, sonst Exponential mit Rate 1 − ρ). Mittel ρ/(1 − ρ), Varianz ρ(2 − ρ)/(1 − ρ)².
Zähleinheit: Beobachtungen (aufeinanderfolgende Wartezeiten) bis zum Alarm, zero-state, bekannte Parameter (Mittel, Standardabweichung der Randverteilung).
* Lag-Autokorrelation: Lag 1 geschlossen (und als Bruch), Lag 1 bis 5 aus einer Zellenkette; zweiter Weg Quadratur der Dichte (`tests/test_oracle_queue.py`).
* Shewhart nach oben bei μ + 3σ: **geschlossene Lösung** der Integralgleichung (zweimaliges Ableiten gibt g'' − (1 − ρ)g' = −ρ); zweite Wege: Zellenkette, Simulation.
* CUSUM einseitig auf z = (W − μ)/σ: iid-Sicht (eindimensionale Kette mit der Randverteilung) gegen wirkliche Reihe (zweidimensionale Kette, `abw_cusum.py`).
Das Gate M/M/2/6 (zwei Spuren, 6 Plätze insgesamt, angenommene Lkw, Zeiteinheit = mittlere Abfertigungsdauer): Randverteilung der Wartezeit exakt, Karten als Simulation."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

import abw_chain as CH
import abw_cusum as CU


# ---------------------------------------------------------------------------------------------------------------- M/M/1: Randverteilung, Kern, Lag
def mm1_moments(rho: float) -> tuple[float, float]:
    return rho / (1.0 - rho), rho * (2.0 - rho) / (1.0 - rho) ** 2


def mm1_stationary_cdf(rho: float):
    def g(y):
        y = np.asarray(y, float)
        return np.where(y < 0, 0.0, 1.0 - rho * np.exp(-(1.0 - rho) * np.where(y < 0, 0.0, y)))
    return g


def mm1_kernel(rho: float):
    """Matrix [j, r] = P(W' ≤ y_r | W = c_j) (Atom bei 0 steckt in F(0 | c))."""
    def f(y, centers):
        y = np.asarray(y, float)
        c = np.asarray(centers, float)[:, None]
        yy = np.where(np.isfinite(y), y, 0.0)[None, :]
        up = yy >= c
        out = np.where(up, 1.0 - rho / (1.0 + rho) * np.exp(-(yy - c)), np.exp(rho * (yy - c)) / (1.0 + rho))
        out = np.where(yy < 0, 0.0, out)
        out = np.where(np.isneginf(y)[None, :], 0.0, out)
        return np.where(np.isposinf(y)[None, :], 1.0, out)
    return f


def mm1_cell_stats(rho: float, edges: np.ndarray):
    """Masse und bedingtes Mittel der stationären Wartezeit je Zelle (erste Kante −∞: das Atom bei 0 steckt in der ersten Zelle; letzte Kante +∞ erlaubt)."""
    r = 1.0 - rho
    e = np.asarray(edges, float)
    lo = np.where(np.isneginf(e[:-1]), 0.0, e[:-1])
    hi = e[1:]
    g = mm1_stationary_cdf(rho)
    ee = np.where(np.isposinf(e), 1e300, np.where(np.isneginf(e), -1.0, e))
    cdf = np.where(np.isposinf(e), 1.0, np.where(np.isneginf(e), 0.0, g(ee)))
    mass = np.diff(cdf)

    def prim(w):
        return -rho * (w + 1.0 / r) * np.exp(-r * w)
    mom = np.array([(0.0 if np.isposinf(hi[i]) else prim(hi[i])) - prim(lo[i]) for i in range(len(hi))])
    return mass, mom / mass


def next_mean(w, rho: float):
    """E[W' | W = w] = w − (1 − ρ)/ρ + e^{−ρw}/(ρ(1 + ρ))."""
    w = np.asarray(w, float)
    return w - (1.0 - rho) / rho + np.exp(-rho * w) / (rho * (1.0 + rho))


def lag1_exact(rho) -> Fraction | float:
    """Lag-1-Autokorrelation: (E[W·W'] − E[W]²)/Var mit E[W·W'] = 2ρ/(1 − ρ)² − 1 + (1 − ρ)/(1 + ρ); für Fraction als Bruch, sonst Gleitkomma."""
    mean = rho / (1 - rho)
    var = rho * (2 - rho) / (1 - rho) ** 2
    eww = 2 * rho / (1 - rho) ** 2 - 1 + (1 - rho) / (1 + rho)
    return (eww - mean ** 2) / var


def lag_chain(rho: float, nlags: int = 5, n: int = 1500) -> np.ndarray:
    """Lag 1 … nlags aus der Zellenkette (n Zellen, bedingte Mittel als Vertreter), Richardson über n und 2n (Fehler ∝ 1/n)."""
    def one(nn):
        r = 1.0 - rho
        mean, var = mm1_moments(rho)
        top = 28.0 / r + mean
        edges = np.concatenate([[-np.inf], np.linspace(0, top, nn)[1:], [np.inf]])
        mass, cen = mm1_cell_stats(rho, edges)
        q = np.diff(mm1_kernel(rho)(edges, cen), axis=1)
        mk = next_mean(cen, rho)
        vals = []
        for lag in range(1, nlags + 1):
            if lag > 1:
                mk = q @ mk
            vals.append((float((mass * cen * mk).sum()) - mean ** 2) / var)
        return np.array(vals)
    return 2.0 * one(2 * n) - one(n)


# ---------------------------------------------------------------------------------------------------------------- M/M/1: Shewhart nach oben (geschlossen)
def shewhart_threshold(rho: float) -> float:
    mean, var = mm1_moments(rho)
    return mean + 3.0 * math.sqrt(var)


def shewhart_iid_arl(rho: float, c: float | None = None) -> float:
    """iid-Sicht: 1/P(W > c) mit P(W > c) = ρ·e^{−(1−ρ)c}."""
    c = shewhart_threshold(rho) if c is None else c
    return 1.0 / (rho * math.exp(-(1.0 - rho) * c))


def _int_terms(rho: float, w: float, c: float):
    """Integrale I1_b(w) = ∫₀^w e^{ρ(y−w)} b(y) dy und I2_b(w) = ∫_w^c e^{−(y−w)} b(y) dy für b ∈ {1, e^{(1−ρ)y}, y} (analytisch)."""
    f = rho / (1.0 - rho)                    # die dritte Basisfunktion ist ρ·y/(1 − ρ), die Integrale unten gelten für y
    i1 = (
        (1.0 - math.exp(-rho * w)) / rho,
        math.exp(-rho * w) * (math.exp(w) - 1.0),
        f * ((w / rho - 1.0 / rho ** 2) + math.exp(-rho * w) / rho ** 2),
    )
    i2 = (
        1.0 - math.exp(-(c - w)),
        math.exp(w) * (math.exp(-rho * w) - math.exp(-rho * c)) / rho,
        f * ((w + 1.0) - (c + 1.0) * math.exp(w - c)),
    )
    return i1, i2


def _basis(rho: float, w: float):
    r = 1.0 - rho
    return (1.0, math.exp(r * w), rho * w / r)


def shewhart_closed(rho: float, c: float | None = None) -> dict:
    """Geschlossene Lösung g(w) = C₀ + C₁·e^{(1−ρ)w} + ρ·w/(1 − ρ) (erwartete weitere Beobachtungen bis zum Alarm aus der Wartezeit w ≤ c) und die ARL bei stationärem Start:
    ARL = 1 + (1 − ρ)·g(0) + ∫₀^c ρ(1 − ρ)e^{−(1−ρ)w}·g(w) dw. C₀ und C₁ aus der Integralgleichung bei w = 0 und w = c. Rückgabe: dict(arl, c0, c1, c)."""
    c = shewhart_threshold(rho) if c is None else c
    kap = rho / (1.0 + rho)
    r = 1.0 - rho

    def row(w):
        i1, i2 = _int_terms(rho, w, c)
        b_w, b_0 = _basis(rho, w), _basis(rho, 0.0)
        return [b_w[k] - math.exp(-rho * w) / (1.0 + rho) * b_0[k] - kap * (i1[k] + i2[k]) for k in range(3)]
    m = np.array([row(0.0), row(c)])
    c0, c1 = np.linalg.solve(m[:, :2], np.array([1.0, 1.0]) - m[:, 2])
    g0 = c0 + c1
    # ∫₀^c ρ r e^{−r w} g(w) dw, g(w) = C₀ + C₁ e^{rw} + ρw/r
    e = math.exp(-r * c)
    integral = rho * r * (c0 * (1.0 - e) / r + c1 * c + (rho / r) * (1.0 - (1.0 + r * c) * e) / r ** 2)
    return dict(arl=1.0 + r * g0 + integral, c0=float(c0), c1=float(c1), c=c)


def shewhart_closed_residual(rho: float, c: float | None = None, points=(0.2, 0.5, 0.8)) -> float:
    """Größter Restfehler der Integralgleichung g(w) = 1 + e^{−ρw}/(1 + ρ)·g(0) + ∫ f(y|w) g(y) dy der geschlossenen Lösung an inneren Punkten w = Anteil · c."""
    sol = shewhart_closed(rho, c)
    c, c0, c1 = sol["c"], sol["c0"], sol["c1"]
    kap = rho / (1.0 + rho)
    worst = 0.0
    for a in points:
        w = a * c
        i1, i2 = _int_terms(rho, w, c)
        b_w, b_0 = _basis(rho, w), _basis(rho, 0.0)
        coef = (c0, c1, 1.0)
        lhs = sum(coef[k] * b_w[k] for k in range(3))
        rhs = 1.0 + math.exp(-rho * w) / (1.0 + rho) * sum(coef[k] * b_0[k] for k in range(3)) + kap * sum(coef[k] * (i1[k] + i2[k]) for k in range(3))
        worst = max(worst, abs(lhs - rhs))
    return worst


def shewhart_chain(rho: float, c: float | None = None, n: int = 800):
    """Zellenkette der Shewhart-Karte nach oben: Zellen über [0, c] (Atom bei 0 in der ersten Zelle, Vertreter = bedingte Mittel). Rückgabe (Q, pin)."""
    c = shewhart_threshold(rho) if c is None else c
    edges = np.concatenate([[-np.inf], np.linspace(0, c, n + 1)[1:]])
    mass, cen = mm1_cell_stats(rho, edges)
    q = np.maximum(np.diff(mm1_kernel(rho)(edges, cen), axis=1), 0.0)
    return q, mass


def shewhart_chain_arl(rho: float, c: float | None = None, n: int = 800) -> float:
    return CH.chain_arl(*shewhart_chain(rho, c, n))


def shewhart_real_p100(rho: float, c: float | None = None, n: int = 800) -> float:
    q, pin = shewhart_chain(rho, c, n)
    return float(1.0 - CH.chain_survival(q, pin, 100)[100])


# ---------------------------------------------------------------------------------------------------------------- M/M/1: CUSUM auf z = (W − μ)/σ
def _cdf_z(rho: float):
    mean, var = mm1_moments(rho)
    sd = math.sqrt(var)
    g = mm1_stationary_cdf(rho)
    return lambda t: g(mean + sd * np.asarray(t, float))


def cusum_iid_arl(rho: float, h: float = 4.096494807, k: float = 0.5, n: int = 800) -> float:
    return CH.cusum_chain_arl(_cdf_z(rho), k, h, n, richardson=True)


def cusum_iid_distribution(rho: float, h: float = 4.096494807, k: float = 0.5, n: int = 800, nmax: int = 3000) -> dict:
    surv = CH.cusum_chain_survival(_cdf_z(rho), k, h, n, nmax)
    cdf = 1.0 - surv
    return dict(p100=float(cdf[100]), median=int(np.argmax(cdf >= 0.5)))


def cusum_real_chain(rho: float, h: float, k: float, m: int, n: int):
    mean, var = mm1_moments(rho)
    sd = math.sqrt(var)
    wmax = mean + sd * (h + k)
    edges = np.concatenate([[-np.inf], np.linspace(0, wmax, n)[1:], [np.inf]])
    mass, cen = mm1_cell_stats(rho, edges)
    q, pi0 = CU.build_chain(mm1_kernel(rho), mm1_stationary_cdf(rho), k, h, m, edges, cen, 1.0 / sd, -mean / sd)
    return q, pi0, len(cen)


def cusum_real_arl(rho: float, h: float = 4.096494807, k: float = 0.5, ms=CU.M_PAIR, n: int = 192) -> float:
    vals = []
    for m in ms:
        q, pi0, nn = cusum_real_chain(rho, h, k, m, n)
        vals.append(CU.chain_arl(q, pi0, nn))
    return (4.0 * vals[1] - vals[0]) / 3.0


def cusum_real_distribution(rho: float, h: float = 4.096494807, k: float = 0.5, m: int = 24, n: int = 192, nmax: int = 3000) -> dict:
    q, pi0, nn = cusum_real_chain(rho, h, k, m, n)
    surv = CU.chain_distribution(q, pi0, nn, nmax, stop_below=0.45, min_steps=101)
    cdf = 1.0 - surv
    ok = ~np.isnan(cdf)
    return dict(p100=float(cdf[100]), median=int(np.argmax(cdf[ok] >= 0.5)))


# ---------------------------------------------------------------------------------------------------------------- Gate M/M/c/K
def gate_exact(lam: float, c: int = 2, kk: int = 6, mu: float = 1.0) -> dict:
    """Randverteilung der Wartezeit eines angenommenen Lkw (PASTA: ein Ankommender sieht n mit π_n, angenommen bei n < K): Verlust, P(W = 0), Mittel, Standardabweichung, Schiefe;
    Wartezeit = Erlang(n − c + 1, c·μ) bei n ≥ c im System."""
    a = lam / mu
    pi = np.empty(kk + 1)
    for n in range(kk + 1):
        pi[n] = a ** n / math.factorial(n) if n <= c else a ** n / (math.factorial(c) * c ** (n - c))
    pi /= pi.sum()
    acc = pi[:kk] / (1.0 - pi[kk])
    wts = {n: float(acc[n]) for n in range(c, kk)}
    rate = c * mu
    m1 = sum(w * (n - c + 1) / rate for n, w in wts.items())
    m2 = sum(w * (n - c + 1) * (n - c + 2) / rate ** 2 for n, w in wts.items())
    m3 = sum(w * (n - c + 1) * (n - c + 2) * (n - c + 3) / rate ** 3 for n, w in wts.items())
    var = m2 - m1 ** 2
    return dict(c=c, k=kk, lam=lam, pi=pi, loss=float(pi[kk]), p0=float(acc[:c].sum()), wts=wts, rate=rate, mean=m1, sd=math.sqrt(var),
                skew=(m3 - 3 * m1 * m2 + 2 * m1 ** 3) / var ** 1.5)


def gate_marginal_sf(ex: dict, x: float) -> float:
    """P(W > x) der Randverteilung (x ≥ 0)."""
    from scipy.stats import gamma as gamma_d
    return float(sum(w * gamma_d.sf(x, n - ex["c"] + 1, scale=1.0 / ex["rate"]) for n, w in ex["wts"].items()))


def gate_shewhart_iid_arl(ex: dict) -> float:
    """iid-Sicht der Shewhart-Karte 3σ (nach oben; die untere Grenze μ − 3σ ist negativ und wird nie erreicht): 1/P(W > μ + 3σ), exakt."""
    return 1.0 / gate_marginal_sf(ex, ex["mean"] + 3.0 * ex["sd"])


def gate_waits(stream, runs: int, nobs: int, lam: float, burn: int = 300, c: int = 2, kk: int = 6, mu: float = 1.0) -> np.ndarray:
    """Wartezeiten der angenommenen Lkw eines M/M/c/K-Gates (FIFO; stationär nach `burn` verworfenen Lkw), `runs` unabhängige Läufe (spaltenweise ein Ankunftsschritt)."""
    need = nobs + burn
    narr = int(need / max(1e-9, (1.0 - gate_exact(lam, c, kk, mu)["loss"])) * 1.15) + 200
    ia = stream.exponential((narr, runs), lam)
    sv = stream.exponential((narr, runs), mu)
    ends = np.zeros((runs, kk))
    free = np.zeros((runs, c))
    t = np.zeros(runs)
    out = np.empty((runs, need))
    cnt = np.zeros(runs, dtype=np.int64)
    ar = np.arange(runs)
    for i in range(narr):
        t = t + ia[i]
        busy = (ends > t[:, None]).sum(axis=1)
        acc = (busy < kk) & (cnt < need)
        rows = ar[acc]
        if rows.size:
            j = np.argmin(free[rows], axis=1)
            start = np.maximum(t[rows], free[rows, j])
            endt = start + sv[i, rows]
            free[rows, j] = endt
            slot = np.argmin(ends[rows], axis=1)
            ends[rows, slot] = endt
            out[rows, cnt[rows]] = start - t[rows]
            cnt[rows] += 1
        if (cnt >= need).all():
            break
    if (cnt < need).any():
        raise RuntimeError("zu wenige Ankünfte erzeugt")
    return out[:, burn:]


def acf_rows(mat: np.ndarray, lags: int = 5):
    """Mittlere Autokorrelation je Lag über die Läufe (Zeilen), mit Standardfehler über die Läufe."""
    x = mat - mat.mean(axis=1, keepdims=True)
    v = (x * x).mean(axis=1)
    res = []
    for k in range(1, lags + 1):
        r = (x[:, :-k] * x[:, k:]).mean(axis=1) / v
        res.append((float(r.mean()), float(r.std(ddof=1) / math.sqrt(mat.shape[0]))))
    return res

