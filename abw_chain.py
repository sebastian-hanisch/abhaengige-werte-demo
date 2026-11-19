"""Exakte Rechnung für Karten auf einer AR(1)-Reihe und auf unabhängigen Werten (Zähleinheit: Beobachtungen bis zum Alarm, bekannte Parameter, zero-state).

Reihe: u_t = φ·u_{t-1} + s·ε_t mit s = √(1 − φ²), Randverteilung N(0, 1), Start stationär. Alle Zahlen dieses Moduls sind Rechnung (kein Zufall):
* Shewhart-Karte auf AR(1): **Zellenkette** (Zustand = letzter Wert, Zellen genau zwischen den Grenzen) mit Richardson-Extrapolation; **zweiter Weg:**
  Integralgleichung (Gauß-Legendre-Nyström). Beide Wege stimmen auf 5·10⁻⁴ überein (`tests/test_oracle_shewhart.py`).
* Alarm-Klumpen: P(zwei Alarmbeobachtungen in Folge) durch Quadratur (zweiter Weg: Einschluss-Ausschluss aus der Kette).
* CUSUM einseitig bei unabhängigen Werten (φ = 0): Integralgleichung (Page; Nyström) und Brook-Evans-Kette; Residuenkarte mit erstem Schritt.
Der CUSUM auf AR(1) (zweidimensionale Kette) steht in `abw_cusum.py`, die Wartezeit-Reihe in `abw_queue.py`."""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr

NOM = 1.0 / (2.0 * float(ndtr(-3.0)))        # 370,398: ARL₀ der Shewhart-Karte 3σ zweiseitig bei unabhängigen normalen Werten
K_CUSUM = 0.5                                  # Referenzwert des CUSUM
SQ2PI = math.sqrt(2.0 * math.pi)


def pdf(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / SQ2PI


def sf(x):
    return ndtr(-np.asarray(x, float))


# ---------------------------------------------------------------------------------------------------------------- Shewhart auf AR(1): Zellenkette
def shewhart_chain(phi: float, lo: float, hi: float, n: int, zshift: float = 0.0, drift: float = 0.0, stat_mean: float = 0.0):
    """Zellenkette der Shewhart-Karte (Alarm, wenn der beobachtete Wert außerhalb [lo, hi] liegt).

    Zustand = Zelle des letzten (nicht alarmierenden) latenten Wertes; die n Zellen liegen genau zwischen den Grenzen (im latenten Maß: [lo - zshift, hi - zshift]).
    zshift: additiver Messversatz (beobachtet wird latent + zshift); drift: Zusatz im Reihenmodell (Verschiebung des Prozessmittels: drift = (1 − φ)·δ);
    stat_mean: Mittel der ersten Beobachtung. Rückgabe: Q (Übergänge innerhalb), pin (Masse der ersten Beobachtung je Zelle)."""
    s = math.sqrt(1.0 - phi * phi)
    e = np.linspace(lo - zshift, hi - zshift, n + 1)
    c = 0.5 * (e[:-1] + e[1:])
    k = ndtr((e[None, :] - phi * c[:, None] - drift) / s)
    q = np.maximum(np.diff(k, axis=1), 0.0)
    pin = np.diff(ndtr(e - stat_mean))
    return q, pin


def chain_arl(q: np.ndarray, pin: np.ndarray) -> float:
    """ARL = 1 + pin · (I − Q)⁻¹ 1: eine Beobachtung, dann die weiteren aus dem Zustand der ersten (falls sie nicht alarmiert)."""
    g = np.linalg.solve(np.eye(q.shape[0]) - q, np.ones(q.shape[0]))
    return 1.0 + float(pin @ g)


def chain_survival(q: np.ndarray, pin: np.ndarray, nmax: int) -> np.ndarray:
    """P(T > t) für t = 0 … nmax (t = 0: 1)."""
    surv = np.empty(nmax + 1)
    surv[0] = 1.0
    v = pin.copy()
    surv[1] = v.sum()
    for t in range(2, nmax + 1):
        v = v @ q
        surv[t] = v.sum()
    return surv


def shewhart_arl(phi: float, lo: float, hi: float, n: int = 480, richardson: bool = True, **kw) -> float:
    """ARL der Shewhart-Karte; mit Richardson: (4·ARL(2n) − ARL(n)) / 3 (Fehler der Ordnung 1/n²)."""
    a1 = chain_arl(*shewhart_chain(phi, lo, hi, n, **kw))
    if not richardson:
        return a1
    a2 = chain_arl(*shewhart_chain(phi, lo, hi, 2 * n, **kw))
    return (4.0 * a2 - a1) / 3.0


def one_sided_limit(gamma: float = NOM) -> float:
    """Grenze c der einseitigen Shewhart-Karte (Alarm bei x > c) mit ARL₀ = gamma bei unabhängigen Werten: c = 2,7822 für 370,398."""
    return float(brentq(lambda c: 1.0 / float(ndtr(-c)) - gamma, 1.0, 5.0, xtol=1e-12))


def shewhart_arl_two_sided(phi: float, limit: float = 3.0, n: int = 480) -> float:
    return shewhart_arl(phi, -limit, limit, n)


def shewhart_arl_one_sided(phi: float, c: float | None = None, n: int = 480) -> float:
    return shewhart_arl(phi, -9.0, one_sided_limit() if c is None else c, n)


def shewhart_distribution(phi: float, limit: float = 3.0, n: int = 960, nmax: int = 16000) -> dict:
    """Lauflängenverteilung der zweiseitigen Shewhart-Karte (Zellenkette n): P(T ≤ 100), Median, 5-%- und 95-%-Quantil."""
    q, pin = shewhart_chain(phi, -limit, limit, n)
    surv = chain_survival(q, pin, nmax)
    cdf = 1.0 - surv
    return dict(p100=float(cdf[100]), median=int(np.argmax(cdf >= 0.5)), q05=int(np.argmax(cdf >= 0.05)),
                q95=int(np.argmax(cdf >= 0.95)) if cdf[-1] >= 0.95 else None)


# ---------------------------------------------------------------------------------------------------------------- Shewhart auf AR(1): zweiter Weg
def shewhart_nystrom(phi: float, lo: float, hi: float, n: int = 220, nx: int = 240, xr: float = 9.0, drift: float = 0.0,
                     start_sd: float = 1.0, start_mean: float = 0.0) -> float:
    """Integralgleichung g(x) = 1 + ∫_lo^hi f(y|x) g(y) dy (Gauß-Legendre-Nyström); ARL = ∫ p₀(x) g(x) dx mit p₀ = Dichte des (unbeobachteten) Wertes vor der ersten Beobachtung."""
    s = math.sqrt(1.0 - phi * phi)
    x, wq = leggauss(n)
    y = 0.5 * (hi - lo) * (x + 1) + lo
    wy = 0.5 * (hi - lo) * wq

    def dens(yv, xv):
        return pdf((yv - phi * xv - drift) / s) / s

    a = dens(y[None, :], y[:, None]) * wy[None, :]
    g = np.linalg.solve(np.eye(n) - a, np.ones(n))
    xs, ws = leggauss(nx)
    xx = xr * (xs + 1) - xr + start_mean
    wx = xr * ws
    gx = 1.0 + (dens(y[None, :], xx[:, None]) * wy[None, :] * g[None, :]).sum(axis=1)
    p0 = pdf((xx - start_mean) / start_sd) / start_sd
    return float((p0 * gx * wx).sum())


def alarm_pair_probability(phi: float, c: float = 3.0) -> float:
    """P(|x_t| > c und |x_{t+1}| > c) im stationären AR(1) (Quadratur; Symmetrie: Faktor 2 für x_t > c)."""
    s = math.sqrt(1.0 - phi * phi)

    def f(x):
        return float(pdf(x)) * (float(ndtr((-c - phi * x) / s)) + float(ndtr(-(c - phi * x) / s)))
    return 2.0 * quad(f, c, 12.0, epsabs=1e-15, epsrel=1e-12)[0]


def clusters(phi: float, c: float = 3.0, n: int = 480) -> dict:
    """Alarm-Klumpen bei Weiterlaufen ohne Anhalten: Rate p der Alarmbeobachtungen (jedes φ gleich), q = P(zweiter Alarm in Folge | Alarm),
    mittlere Klumpenlänge 1/(1 − q), Kehrwert der Rate der Klumpenanfänge 1/(p(1 − q)) neben der ARL₀ (zweiseitig)."""
    p = 2.0 * float(ndtr(-c))
    q = alarm_pair_probability(phi, c) / p
    arl = shewhart_arl(phi, -c, c, n)
    start_rate_inv = 1.0 / (p * (1.0 - q))
    share = (start_rate_inv - 1.0 / p) / (arl - 1.0 / p) if arl > 1.0 / p + 1e-9 else float("nan")
    return dict(p=p, q=q, cluster_length=1.0 / (1.0 - q), start_rate_inv=start_rate_inv, arl=arl, share=share)


def calibrated_limit(phi: float, gamma: float = NOM, n: int = 480) -> float:
    """Nachgeeichte Shewhart-Grenze L_φ (zweiseitig): ARL₀ der Rohdatenkarte auf AR(1) gleich gamma."""
    return float(brentq(lambda lim: shewhart_arl(phi, -lim, lim, n) - gamma, 2.0, 3.2, xtol=1e-8))


# ---------------------------------------------------------------------------------------------------------------- Residuenkarte (bekanntes φ)
def residual_innovation_means(phi: float, delta: float, art: str) -> tuple[float, float]:
    """Mittel der standardisierten Innovation im ersten und in den späteren Schritten. art = "offset" (Messversatz; die Karte kennt x₀ vom Versatz unbetroffen)
    oder "process" (Verschiebung des Prozessmittels: konstant (1 − φ)·δ/s)."""
    s = math.sqrt(1.0 - phi * phi)
    later = (1.0 - phi) * delta / s
    return (delta / s if art == "offset" else later), later


def alarm_prob(mu: float, lim: float = 3.0) -> float:
    return float(ndtr(mu - lim) + ndtr(-lim - mu))


def residual_shewhart_arl(phi: float, delta: float, art: str = "offset", lim: float = 3.0) -> float:
    """ARL₁ der Residuenkarte (Shewhart, Grenze ±lim): 1 + (1 − p₁)/p₂ mit den Alarmwahrscheinlichkeiten des ersten und der späteren Schritte."""
    m1, m2 = residual_innovation_means(phi, delta, art)
    p1, p2 = alarm_prob(m1, lim), alarm_prob(m2, lim)
    return 1.0 + (1.0 - p1) / p2


# ---------------------------------------------------------------------------------------------------------------- CUSUM bei unabhängigen Werten (eindimensional)
def cusum_iid_L(k: float, h: float, mu: float, n: int = 100):
    """Einseitiger CUSUM S_t = max(0, S_{t-1} + x_t − k), x_t ~ N(mu, 1): Integralgleichung (Page; Nyström mit Gauß-Legendre).
    Rückgabe: L(0) und die Funktion L(u) (ARL aus Startwert u)."""
    x, wq = leggauss(n)
    y = 0.5 * h * (x + 1)
    wq = 0.5 * h * wq
    u = np.concatenate([[0.0], y])
    a = np.zeros((n + 1, n + 1))
    a[:, 0] = ndtr(k - u - mu)
    a[:, 1:] = pdf(y[None, :] - u[:, None] + k - mu) * wq[None, :]
    lv = np.linalg.solve(np.eye(n + 1) - a, np.ones(n + 1))

    def lfun(uu):
        uu = np.asarray(uu, float)
        return 1.0 + lv[0] * ndtr(k - uu - mu) + (pdf(y - uu[..., None] + k - mu) * wq * lv[1:]).sum(-1)
    return float(lv[0]), lfun


def cusum_iid_arl(k: float, h: float, mu: float = 0.0, n: int = 100) -> float:
    return cusum_iid_L(k, h, mu, n)[0]


def cusum_h_for_arl(k: float, gamma: float = NOM, n: int = 100) -> float:
    """Schwelle h des einseitigen CUSUM mit ARL₀ = gamma bei unabhängigen normalen Werten: 4,0965 für k = 0,5 und 370,398."""
    return float(brentq(lambda h: cusum_iid_arl(k, h, 0.0, n) - gamma, 0.3, 25.0, xtol=1e-12))


def residual_cusum_arl(h: float, k: float, m1: float, m2: float, n: int = 160) -> float:
    """CUSUM auf den Innovationen: erster Schritt N(m1, 1), danach N(m2, 1); ARL aus S₀ = 0."""
    l0, lf = cusum_iid_L(k, h, m2, n)
    if abs(m1 - m2) < 1e-15:
        return l0
    x, w = leggauss(n)
    y = 0.5 * h * (x + 1)
    w = 0.5 * h * w
    return 1.0 + float(ndtr(k - m1)) * l0 + float((lf(y) * pdf(y + k - m1) * w).sum())


# ---------------------------------------------------------------------------------------------------------------- CUSUM bei unabhängigen Werten: Brook-Evans-Kette
def cusum_chain(cdf, k: float, h: float, n: int) -> np.ndarray:
    """Brook-Evans-Kette für C = max(0, C + x − k), x mit Verteilungsfunktion cdf: Zustand 0 und n Zellen über (0, h]; Übergangsmatrix (n+1)×(n+1) ohne Alarm."""
    wdt = h / n
    reps = np.concatenate([[0.0], (np.arange(n) + 0.5) * wdt])
    edges = np.arange(n + 1) * wdt
    f = cdf(k - reps[:, None] + edges[None, :])
    q = np.empty((n + 1, n + 1))
    q[:, 0] = cdf(k - reps)
    q[:, 1:] = f[:, 1:] - f[:, :-1]
    return np.maximum(q, 0.0)


def cusum_chain_arl(cdf, k: float, h: float, n: int = 800, richardson: bool = True) -> float:
    """ARL₀ des einseitigen CUSUM bei unabhängigen Werten mit Verteilungsfunktion cdf; mit Richardson über n/2 und n (Fehler ∝ 1/n)."""
    def one(nn):
        q = cusum_chain(cdf, k, h, nn)
        return float(np.linalg.solve(np.eye(nn + 1) - q, np.ones(nn + 1))[0])
    if not richardson:
        return one(n)
    return 2.0 * one(n) - one(n // 2)


def cusum_chain_survival(cdf, k: float, h: float, n: int, nmax: int) -> np.ndarray:
    """P(T > t), t = 0 … nmax, Start S = 0."""
    q = cusum_chain(cdf, k, h, n)
    out = np.empty(nmax + 1)
    v = np.ones(n + 1)
    out[0] = 1.0
    for t in range(1, nmax + 1):
        v = q @ v
        out[t] = v[0]
    return out
