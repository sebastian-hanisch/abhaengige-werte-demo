"""Simulationen (SplitMix64-Strom, `abw_normal.py`): Reihen, Läufe bis zum Alarm, Alarm-Klumpen, M/M/1 und Gate.

Alle Läufe sind vektorisiert über die Läufe **und** die Zeit: die AR(1)-Reihe entsteht mit `scipy.signal.lfilter`, der CUSUM und die Lindley-Kette mit der Spiegelungsformel
(S_t = C_t − min(0, min_{u ≤ t} C_u) mit C = Summe der Schritte); `tests/test_oracle_*.py` vergleicht beides mit der Schritt-für-Schritt-Schleife. Läufe ohne Alarm bis `tmax` heißen
**zensiert** (Lauflänge tmax + 1 im Feld, in der Anzeige als „zensiert“ gezählt). Zähleinheit: Beobachtungen bis zum Alarm je Lauf (ein Lauf = eine Reihe)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

import abw_chain as CH
import abw_normal as N
import abw_queue as Q

H_ONE = 4.096494807
K = 0.5
H_TWO = 4.77                      # zweiseitiger CUSUM, gerundet wie in V2 (ARL₀ = 368,6 bei unabhängigen Werten; nicht dieselbe Karte wie h = 4,7749)
TMAX = 10_000
LIVE_LIMIT = 5_000_000            # Obergrenze für R · tmax (Zufallszahlen eines Live-Laufs)


class LiveLimitError(ValueError):
    """Der gewünschte Live-Lauf überschreitet die Obergrenze R · tmax <= 5 · 10^6."""


def check_limit(runs: int, tmax: int, factor: int = 1) -> None:
    if runs * tmax * factor > LIVE_LIMIT:
        raise LiveLimitError(f"R · tmax = {runs} · {tmax} = {runs * tmax * factor:,} liegt über der Obergrenze {LIVE_LIMIT:,} für Live-Läufe".replace(",", " "))


# ---------------------------------------------------------------------------------------------------------------- Reihen
def ar1_series(stream: N.Stream, runs: int, length: int, phi: float, drift: float = 0.0):
    """Stationäre AR(1)-Reihen u_t = φ·u_{t-1} + s·ε_t (+ drift): Rückgabe (u₀, u₁ … u_T) mit u₀ ~ N(0, 1)."""
    s = math.sqrt(1.0 - phi * phi)
    u0 = stream.normal(runs)
    eps = stream.normal((runs, length))
    if drift:
        eps = eps + drift / s
    u = lfilter([s], [1.0, -phi], eps, axis=1, zi=(phi * u0)[:, None])[0]
    return u0, u


def first_alarm(mask: np.ndarray) -> np.ndarray:
    """Erste Beobachtung mit Alarm je Zeile (1-basiert); ohne Alarm: Länge der Zeile + 1 (zensiert)."""
    t = mask.shape[1]
    idx = np.where(mask.any(axis=1), mask.argmax(axis=1) + 1, t + 1)
    return idx.astype(np.int64)


def cusum_paths(z: np.ndarray, k: float = K, lower: bool = False) -> np.ndarray:
    """CUSUM-Pfade S_t = max(0, S_{t-1} ± z_t − k) aus S₀ = 0 (Spiegelungsformel); lower=True: S_t = max(0, S_{t-1} − z_t − k)."""
    c = np.cumsum((-z if lower else z) - k, axis=1)
    return c - np.minimum(0.0, np.minimum.accumulate(c, axis=1))


def cusum_paths_loop(z: np.ndarray, k: float = K) -> np.ndarray:
    """Dieselbe Rekursion als Schleife über die Zeit (Referenz im Test)."""
    s = np.zeros(z.shape[0])
    out = np.empty_like(z)
    for t in range(z.shape[1]):
        s = np.maximum(0.0, s + z[:, t] - k)
        out[:, t] = s
    return out


def lindley_paths(x: np.ndarray, w0: np.ndarray) -> np.ndarray:
    """W_t = max(0, W_{t-1} + X_t) aus W₀ = w0: W_t = C_t − min(−W₀, min_{1 ≤ u ≤ t} C_u), C = Summe der Schritte."""
    c = np.cumsum(x, axis=1)
    return c - np.minimum(-w0[:, None], np.minimum.accumulate(c, axis=1))


# ---------------------------------------------------------------------------------------------------------------- Karten auf AR(1): Läufe ohne Änderung
@dataclass(frozen=True)
class Ar1Run:
    phi: float
    runs: int
    tmax: int
    seed: int
    lengths: dict            # Karte -> Lauflängen je Lauf (tmax + 1 = zensiert)
    summary: dict            # Karte -> dict(mean, se, p100, censored)
    seconds: float


def summarize(lengths: np.ndarray, tmax: int) -> dict:
    ok = lengths <= tmax
    n = lengths.size
    mean = float(lengths.mean())
    se = float(lengths.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return dict(mean=mean, se=se, p100=float((lengths <= 100).mean()), censored=int((~ok).sum()), median=float(np.median(lengths)))


def ar1_runs(phi: float, runs: int, seed: int, tmax: int = TMAX, h: float = H_ONE, c1: float | None = None, check: bool = True) -> Ar1Run:
    """Läufe bis zum Alarm für Shewhart zweiseitig (|x| > 3), Shewhart einseitig (x > c) und CUSUM einseitig (k = 0,5, h) auf derselben AR(1)-Reihe (zero-state, stationärer Start)."""
    if check:
        check_limit(runs, tmax)
    t0 = time.perf_counter()
    c1 = CH.one_sided_limit() if c1 is None else c1
    _, u = ar1_series(N.Stream(seed), runs, tmax, phi)
    lengths = {
        "shewhart2": first_alarm(np.abs(u) > 3.0),
        "shewhart1": first_alarm(u > c1),
        "cusum": first_alarm(cusum_paths(u) >= h),
    }
    return Ar1Run(phi, runs, tmax, seed, lengths, {k: summarize(v, tmax) for k, v in lengths.items()}, time.perf_counter() - t0)


def cusum_two_sided_runs(phi: float, runs: int, seed: int, h: float = H_TWO, tmax: int = 20_000) -> np.ndarray:
    """Lauflängen des zweiseitigen CUSUM (k = 0,5, h) auf AR(1)."""
    _, u = ar1_series(N.Stream(seed), runs, tmax, phi)
    hit = (cusum_paths(u) >= h) | (cusum_paths(u, lower=True) >= h)
    return first_alarm(hit)


# ---------------------------------------------------------------------------------------------------------------- Karten auf AR(1): Läufe nach der Änderung
def observed_series(stream: N.Stream, runs: int, tmax: int, phi: float, delta: float, art: str):
    """(o₀, o₁ … o_T): beobachtete Werte. art = "offset": o_t = u_t + δ für t ≥ 1 (o₀ = u₀ ohne Versatz); art = "process": x_t = φ·x_{t-1} + (1 − φ)·δ + s·ε_t mit x₀ ~ N(0, 1)."""
    if art == "process":
        u0, x = ar1_series(stream, runs, tmax, phi, drift=(1.0 - phi) * delta)
        return u0, x
    u0, u = ar1_series(stream, runs, tmax, phi)
    return u0, u + delta


def change_runs(phi: float, delta: float, art: str, scheme: str, card: str, runs: int, seed: int, tmax: int = 3000,
                limit: float = 3.0, h: float = H_ONE, check: bool = True) -> np.ndarray:
    """Lauflängen nach der Änderung (ν = 0). scheme: "roh" (Karte auf den beobachteten Werten) oder "residuum" (Karte auf e_t = (o_t − φ·o_{t-1})/s, φ bekannt, o₀ wie oben);
    card: "shewhart" (zweiseitig, Grenze ±limit) oder "cusum" (einseitig oben, k = 0,5, Schwelle h)."""
    if check:
        check_limit(runs, tmax)
    s = math.sqrt(1.0 - phi * phi)
    o0, o = observed_series(N.Stream(seed), runs, tmax, phi, delta, art)
    z = o if scheme == "roh" else (o - phi * np.concatenate([o0[:, None], o[:, :-1]], axis=1)) / s
    if card == "shewhart":
        return first_alarm(np.abs(z) > limit)
    return first_alarm(cusum_paths(z) >= h)


def change_all(phi: float, delta: float, art: str, runs: int, seed: int, tmax: int, limit_cal: float, h_cal: float, h_nom: float = H_ONE) -> dict:
    """Alle sechs Läufe der Tabelle von Experiment 3 auf **derselben** Reihe: Shewhart (zweiseitig) und CUSUM (einseitig), je Rohdaten nominell, Rohdaten nachgeeicht (Grenze limit_cal,
    Schwelle h_cal) und Residuenkarte (φ bekannt). Schlüssel "karte|schema"; Lauflängen ab der Änderung (ν = 0), tmax + 1 = zensiert."""
    check_limit(runs, tmax)
    s = math.sqrt(1.0 - phi * phi)
    o0, o = observed_series(N.Stream(seed), runs, tmax, phi, delta, art)
    e = (o - phi * np.concatenate([o0[:, None], o[:, :-1]], axis=1)) / s
    return {
        "shewhart|nominal": first_alarm(np.abs(o) > 3.0),
        "shewhart|calibrated": first_alarm(np.abs(o) > limit_cal),
        "shewhart|residual": first_alarm(np.abs(e) > 3.0),
        "cusum|nominal": first_alarm(cusum_paths(o) >= h_nom),
        "cusum|calibrated": first_alarm(cusum_paths(o) >= h_cal),
        "cusum|residual": first_alarm(cusum_paths(e) >= h_nom),
    }


# ---------------------------------------------------------------------------------------------------------------- Alarm-Klumpen (eine Reihe ohne Anhalten)
@dataclass(frozen=True)
class ClusterSeries:
    phi: float
    length: int
    values: tuple
    alarms: tuple             # Indizes (1-basiert) der Alarmbeobachtungen |x| > 3
    clusters: tuple           # Längen der Folgen aufeinanderfolgender Alarmbeobachtungen
    seconds: float


def cluster_series(phi: float, length: int, seed: int) -> ClusterSeries:
    t0 = time.perf_counter()
    _, u = ar1_series(N.Stream(seed), 1, length, phi)
    x = u[0]
    alarm = np.abs(x) > 3.0
    idx = np.nonzero(alarm)[0] + 1
    runs_, cur = [], 0
    for a in alarm:
        if a:
            cur += 1
        elif cur:
            runs_.append(cur)
            cur = 0
    if cur:
        runs_.append(cur)
    return ClusterSeries(phi, length, tuple(float(v) for v in x), tuple(int(i) for i in idx), tuple(runs_), time.perf_counter() - t0)


def cluster_counts(phi: float, runs: int, length: int, seed: int) -> dict:
    """Über viele Reihen (Länge `length`): Zahl der Alarmbeobachtungen, der Folgenanfänge und der Paare in Folge (Schätzer für p, p(1 − q), q)."""
    _, u = ar1_series(N.Stream(seed), runs, length, phi)
    a = np.abs(u) > 3.0
    n_alarm = int(a.sum())
    pairs = int((a[:, 1:] & a[:, :-1]).sum())
    starts = n_alarm - pairs
    return dict(alarms=n_alarm, starts=starts, pairs=pairs, observations=runs * length,
                p=n_alarm / (runs * length), q=pairs / n_alarm if n_alarm else float("nan"))


# ---------------------------------------------------------------------------------------------------------------- M/M/1
def mm1_runs(rho: float, runs: int, seed: int, tmax: int = 5000, shewhart: bool = True, cusum: bool = True, h: float = H_ONE, check: bool = True) -> dict:
    """Läufe bis zum Alarm auf der M/M/1-Wartezeit (stationärer Start): Shewhart nach oben bei μ + 3σ und CUSUM einseitig auf z = (W − μ)/σ. Rückgabe: Karte -> Lauflängen."""
    if check:
        check_limit(runs, tmax, 2)
    st = N.Stream(seed)
    r = 1.0 - rho
    mean, var = Q.mm1_moments(rho)
    sd = math.sqrt(var)
    w0 = np.where(st.uniform(runs) < rho, st.exponential(runs, r), 0.0)
    x = st.exponential((runs, tmax), 1.0) - st.exponential((runs, tmax), rho)
    w = lindley_paths(x, w0)
    out = {}
    if shewhart:
        out["shewhart"] = first_alarm(w > mean + 3.0 * sd)
    if cusum:
        out["cusum"] = first_alarm(cusum_paths((w - mean) / sd) >= h)
    return out


# ---------------------------------------------------------------------------------------------------------------- Gate M/M/2/6
@dataclass(frozen=True)
class GateRun:
    lam: float
    runs: int
    seed: int
    shewhart: dict
    cusum2: dict
    lag: tuple
    seconds: float


def gate_iid_draws(stream: N.Stream, runs: int, length: int, ex: dict) -> np.ndarray:
    """Unabhängige Ziehungen aus der exakten Randverteilung der Wartezeit (Atom bei 0 plus Mischung von Erlang-Verteilungen)."""
    u = stream.uniform((runs, length))
    pick = stream.uniform((runs, length))
    ks = sorted(ex["wts"])
    weights = np.array([ex["wts"][n] for n in ks])
    p_wait = weights.sum()
    cum = np.cumsum(weights) / p_wait
    which = np.searchsorted(cum, pick)                 # Index in ks
    shape = np.array([n - ex["c"] + 1 for n in ks])[np.minimum(which, len(ks) - 1)]
    prod = np.ones((runs, length))
    for j in range(1, int(shape.max()) + 1):
        prod = np.where(shape >= j, prod * stream.uniform((runs, length)), prod)
    wait = -np.log(prod) / ex["rate"]
    return np.where(u < p_wait, wait, 0.0)


def gate_cards(w: np.ndarray, ex: dict, tmax: int, h: float = H_TWO) -> dict:
    """Shewhart 3σ (zweiseitig auf z; die untere Grenze wird nie erreicht) und CUSUM zweiseitig (k = 0,5, h) auf z = (W − μ)/σ der Randverteilung; Lauflängen je Reihe (Zeile)."""
    z = (w[:, :tmax] - ex["mean"]) / ex["sd"]
    return dict(shewhart=first_alarm(np.abs(z) > 3.0), cusum2=first_alarm((cusum_paths(z) >= h) | (cusum_paths(z, lower=True) >= h)))


def gate_run(lam: float, runs: int, seed: int, nobs: int = 1500, burn: int = 300) -> GateRun:
    """Läufe am Gate (M/M/2/6, zwei Spuren, 6 Plätze insgesamt, angenommene Lkw): Karten auf der wirklichen Reihe und Lag 1 bis 5 der Wartezeiten."""
    t0 = time.perf_counter()
    ex = Q.gate_exact(lam)
    w = Q.gate_waits(N.Stream(seed), runs, nobs, lam, burn)
    res = gate_cards(w, ex, nobs)
    lag = tuple(Q.acf_rows(w[:, : min(nobs, w.shape[1])])[:5]) if runs > 1 else ()
    return GateRun(lam, runs, seed, summarize(res["shewhart"], nobs), summarize(res["cusum2"], nobs), lag, time.perf_counter() - t0)
