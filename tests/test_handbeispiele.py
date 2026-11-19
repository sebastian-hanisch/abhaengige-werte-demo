"""Handbeispiele des Plans (H1 bis H6) als exakte Tests: Brüche, Ganzzahlen, Tafelwerte. Die Handwerte entstehen hier **ohne den Code der Demo** (nur `fractions`, `math`, numpy, scipy);
danach prüft jeweils ein Test, dass die Funktionen der Demo dieselben Zahlen liefern."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

import abw_chain as CH
import abw_queue as Q
import abw_skew as SK
import abw_toy as T


def _solve(a, b):
    """Gauß-Elimination mit Brüchen (eigene Fassung, nicht die der Demo)."""
    n = len(a)
    m = [[Fr(v) for v in row] + [Fr(bb)] for row, bb in zip(a, b)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        m[c] = [v / m[c][c] for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] for i in range(n)]


# ----------------------------------------------------------------------------------------------------------------------------------------- H1
@pytest.mark.parametrize("phi,arl_a,arl_b", [(Fr(0), 10, 110), (Fr(1, 2), 19, Fr(409, 11)), (Fr(4, 5), 46, Fr(2386, 41))])
def test_h1_zwei_zustaende_kette_von_hand(phi, arl_a, arl_b):
    p = Fr(1, 10)
    a, b = p * (1 - phi), (1 - p) * (1 - phi)
    assert a / (a + b) == p and 1 - a - b == phi                 # stationär P(Spitze) = p, zweiter Eigenwert = φ
    assert 1 + (1 - p) / a == arl_a                               # Regel A: erste Spitze
    e0, e1 = _solve([[a, -a], [-b, 1]], [1, 1])                   # Regel B: E0 = 1 + a·E1 + (1 − a)·E0, E1 = 1 + b·E0
    assert e0 == (1 + a) / (a * (1 - b))
    assert 1 + p * e1 + (1 - p) * e0 == arl_b


def test_h1_zweiter_weg_ueberlebenssumme_in_gleitkomma():
    p = 0.1
    for phi, arl_b in ((0.0, 110.0), (0.5, 409 / 11), (0.8, 2386 / 41)):
        a, b = p * (1 - phi), (1 - p) * (1 - phi)
        q = np.array([[1 - a, a], [b, 0.0]])
        cur, tot = np.array([1 - p, p]), 1.0
        for _ in range(40000):
            tot += cur.sum()
            cur = cur @ q
        assert abs(tot - arl_b) < 1e-6


def test_h1_regel_b_ist_nicht_monoton_regel_a_steigt():
    a = [T.two_state(Fr(x))["arl_a"] for x in (Fr(0), Fr(1, 2), Fr(4, 5))]
    b = [T.two_state(Fr(x))["arl_b"] for x in (Fr(0), Fr(1, 2), Fr(4, 5))]
    assert a == [10, 19, 46] and b == [110, Fr(409, 11), Fr(2386, 41)]
    assert b[1] < b[2] < b[0]                                     # fällt, steigt wieder, bleibt unter 110


# ----------------------------------------------------------------------------------------------------------------------------------------- H2
def test_h2_diskrete_wartezeit_von_hand():
    pu, pd = Fr(1, 3), Fr(2, 3)
    r = pu / pd
    pi = lambda n: (1 - r) * r ** n  # noqa: E731
    mean, ew2 = r / (1 - r), r * (1 + r) / (1 - r) ** 2
    assert mean == 1 and ew2 == 3 and ew2 - mean ** 2 == 2
    # Lag 1: E[W'|n] = n − 1/3 für n ≥ 1; Lag 2: 7/9 bei n = 1, n − 2/3 ab n = 2
    var = ew2 - mean ** 2
    assert (ew2 - Fr(1, 3) * mean - mean ** 2) / var == Fr(5, 6)
    eww2 = pi(1) * Fr(7, 9) + (ew2 - pi(1)) - Fr(2, 3) * (mean - pi(1))
    assert (eww2 - mean ** 2) / var == Fr(13, 18)
    e0, e1, e2 = _solve([[1 - pd, -pu, 0], [-pd, 1, -pu], [0, -pd, 1]], [1, 1, 1])
    assert (e0, e1, e2) == (33, 30, 21)
    arl = 1 + pi(0) * e0 + pi(1) * e1 + pi(2) * e2
    assert arl == Fr(221, 8) and r ** 3 == Fr(1, 8) and arl / 8 == Fr(221, 64)    # 3,45faches der iid-Sicht 8


def test_h2_zweiter_weg_abgeschnittene_kette_und_ueberlebenssumme():
    n = 80
    p = np.zeros((n, n))
    for s in range(n):
        p[s, min(s + 1, n - 1)] += 1 / 3
        p[s, max(s - 1, 0)] += 2 / 3
    w = np.array([0.5 ** (s + 1) for s in range(n)])
    w /= w.sum()
    st = np.arange(n, dtype=float)
    mu = float(w @ st)
    va = float(w @ st ** 2) - mu ** 2
    for lag, soll in ((1, 5 / 6), (2, 13 / 18)):
        c = float(w @ (st * (np.linalg.matrix_power(p, lag) @ st))) - mu ** 2
        assert abs(c / va - soll) < 1e-9
    qs = np.array([[2 / 3, 1 / 3, 0], [2 / 3, 0, 1 / 3], [0, 2 / 3, 0]])
    cur, tot = np.array([1 / 2, 1 / 4, 1 / 8]), 1.0
    for _ in range(5000):
        tot += cur.sum()
        cur = cur @ qs
    assert abs(tot - 221 / 8) < 1e-9


def test_h2_demo_liefert_dieselben_zahlen():
    d = T.discrete_wait()
    assert d["arl"] == Fr(221, 8) and d["further"] == [33, 30, 21] and d["lag1"] == Fr(5, 6) and d["lag2"] == Fr(13, 18) and d["iid_arl"] == 8 and d["ratio"] == Fr(221, 64)


# ----------------------------------------------------------------------------------------------------------------------------------------- H3
def test_h3_residuenkarte_phi_06_messversatz_1():
    phi = 0.6
    s = math.sqrt(1 - phi ** 2)
    assert abs(s - 0.8) < 1e-15
    pa = lambda m: norm.sf(3 - m) + norm.cdf(-3 - m)  # noqa: E731
    m1, m2 = 1 / s, (1 - phi) / s
    assert abs(m1 - 1.25) < 1e-12 and abs(m2 - 0.5) < 1e-12
    p1, p2 = pa(m1), pa(m2)
    assert abs(p1 - 0.04007) < 5e-6 and abs(p2 - 0.006442) < 5e-7
    assert abs(1 + (1 - p1) / p2 - 150.00) < 0.01
    ohne = 1 + (1 - pa(1.0)) / p2                                   # e₁ = x₁ ohne x₀ (Randverteilung): p₁ = P(|N(1; 1)| > 3) = 0,02278
    assert abs(pa(1.0) - 0.022782) < 1e-6 and abs(ohne - 152.69) < 0.01
    assert abs(1 / (2 * norm.sf(3)) - 370.3983) < 1e-4              # ARL₀ der Innovationskarte, für jedes φ


def test_h3_demo_liefert_dieselben_zahlen():
    assert abs(CH.residual_shewhart_arl(0.6, 1.0, "offset") - 150.00) < 0.01
    m1, m2 = CH.residual_innovation_means(0.6, 1.0, "offset")
    assert abs(m1 - 1.25) < 1e-12 and abs(m2 - 0.5) < 1e-12
    assert CH.residual_innovation_means(0.6, 1.0, "process") == (pytest.approx(0.5), pytest.approx(0.5))


# ----------------------------------------------------------------------------------------------------------------------------------------- H4
def test_h4_schiefe_geschlossen_und_vorzeichen_cusum_von_hand():
    assert abs(math.exp(4) - 54.598) < 1e-3
    assert sum(Fr(10 ** j, math.factorial(j)) for j in range(4)) == Fr(683, 3)
    assert abs(math.exp(10) * 3 / 683 - 96.75) < 0.01
    assert abs(1 / (2 * norm.sf(3.0)) / math.exp(4) - 6.784) < 0.01
    # Vorzeichen-CUSUM H = 3 von Hand: E2 = 1 + E0/2, E1 = 1 + (E2 + E0)/2, E0 = 1 + (E1 + E0)/2  =>  E0 = 14
    e0 = Fr(14)
    e2 = 1 + e0 / 2
    e1 = 1 + (e2 + e0) / 2
    assert e0 == 1 + (e1 + e0) / 2

    def vorz(h):
        a = [[Fr(0)] * h for _ in range(h)]
        for s in range(h):
            a[s][s] += 1
            for step in (1, -2):
                ns = max(0, s + step)
                if ns < h:
                    a[s][ns] -= Fr(1, 2)
        return _solve(a, [1] * h)[0]
    assert (vorz(3), vorz(8), vorz(9)) == (14, 266, 442)


def test_h4_demo_liefert_dieselben_zahlen():
    assert [SK.sign_cusum_arl(h) for h in (3, 8, 9)] == [14, 266, 442]
    assert abs(SK.exponential_closed() - math.exp(4)) < 1e-12 and abs(SK.gamma4_closed() - math.exp(10) * 3 / 683) < 1e-12


# ----------------------------------------------------------------------------------------------------------------------------------------- H5
@pytest.mark.parametrize("phi,m", [(Fr(1, 2), 2), (Fr(4, 5), 10), (Fr(-1, 2), 6)])
def test_h5_varianz_einer_summe_gegen_kovarianzmatrix(phi, m):
    brute = sum(phi ** abs(i - j) for i in range(m) for j in range(m))
    assert m + 2 * sum((m - j) * phi ** j for j in range(1, m)) == brute
    assert T.sum_variance(m, phi) == brute


def test_h5_beispiele_und_grenzwert():
    assert T.sum_variance(2, Fr(1, 2)) == 3                         # statt 2
    assert T.sum_variance_limit(Fr(4, 5)) == 9
    assert abs(float(T.sum_variance(4000, Fr(4, 5))) / 4000 - 9) < 0.01


# ----------------------------------------------------------------------------------------------------------------------------------------- H6
@pytest.mark.parametrize("rho,soll", [(Fr(1, 2), Fr(7, 9)), (Fr(4, 5), Fr(26, 27)), (Fr(9, 10), Fr(207, 209))])
def test_h6_mm1_lag1_als_bruch_und_durch_quadratur(rho, soll):
    mean = rho / (1 - rho)
    var = rho * (2 - rho) / (1 - rho) ** 2
    eww = 2 * rho / (1 - rho) ** 2 - 1 + (1 - rho) / (1 + rho)
    assert (eww - mean ** 2) / var == soll and Q.lag1_exact(rho) == soll
    r = float(rho)
    kap = r / (1 + r)

    def next_mean(w):
        if w == 0:
            return integrate.quad(lambda y: y * kap * math.exp(-y), 0, math.inf)[0]
        return (integrate.quad(lambda y: y * kap * math.exp(r * (y - w)), 0, w)[0] + integrate.quad(lambda y: y * kap * math.exp(-(y - w)), w, math.inf)[0])
    for w in (0.0, 0.7, 3.0):
        assert abs(next_mean(w) - (w - (1 - r) / r + math.exp(-r * w) / (r * (1 + r)))) < 1e-9
    q = integrate.quad(lambda w: w * r * (1 - r) * math.exp(-(1 - r) * w) * next_mean(w), 0, 40 / (1 - r), limit=200)[0]
    assert abs(q - float(eww)) < 1e-5 * float(eww)


def test_h6_mm1_mittel_varianz_als_bruch():
    assert (Fr(1, 2) / (1 - Fr(1, 2)), Fr(1, 2) * (2 - Fr(1, 2)) / (1 - Fr(1, 2)) ** 2) == (1, 3)
    mean, var = Q.mm1_moments(0.8)
    assert (round(mean, 12), round(var, 12)) == (4.0, 24.0)
