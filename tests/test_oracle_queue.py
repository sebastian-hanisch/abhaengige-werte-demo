"""Orakel: M/M/1-Wartezeit. Prüflinge: Lag 1 (geschlossen), Lag 1 bis 5 (Zellenkette), Shewhart nach oben (geschlossene Lösung), CUSUM (iid-Sicht eindimensional, wirkliche Reihe zweidimensional).
Zweite Wege: Quadratur der Dichte, Zellenkette, Simulation der Lindley-Rekursion als Schleife und vektorisiert (SplitMix64, Band von 4 Standardfehlern), Randbedingungen der Integralgleichung."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import integrate

import abw_normal as N
import abw_queue as Q
import abw_scenario as SC

H = 4.096494807
V2_LAGS = {0.5: (0.777, 0.616, 0.495, 0.402, 0.329), 0.8: (0.962, 0.927, 0.894, 0.864, 0.836), 0.9: (0.989, 0.979, 0.969, 0.959, 0.950)}   # V2: Simulation, 400 × 30 000


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.9])
def test_lag1_geschlossen_gleich_quadratur_der_dichte(rho):
    r = 1 - rho
    kap = rho / (1 + rho)

    def nxt(w):
        if w == 0:
            return integrate.quad(lambda y: y * kap * math.exp(-y), 0, math.inf, epsabs=1e-13)[0]
        return (integrate.quad(lambda y: y * kap * math.exp(rho * (y - w)), 0, w, epsabs=1e-13)[0]
                + integrate.quad(lambda y: y * kap * math.exp(-(y - w)), w, math.inf, epsabs=1e-13)[0])
    for w in (0.0, 1.3, 7.0):
        assert abs(float(Q.next_mean(w, rho)) - nxt(w)) < 1e-9
    eww = integrate.quad(lambda w: w * rho * r * math.exp(-r * w) * nxt(w), 0, 60 / r, epsabs=1e-12, epsrel=1e-12, limit=400)[0]
    mean, var = Q.mm1_moments(rho)
    assert abs((eww - mean ** 2) / var - Q.lag1_exact(rho)) < 1e-6


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.9])
def test_lag_1_bis_5_aus_der_kette_gleich_lag1_geschlossen_und_v2_simulation(rho):
    lags = Q.lag_chain(rho)
    assert abs(lags[0] - Q.lag1_exact(rho)) < 2e-3
    assert all(abs(a - b) < 0.01 for a, b in zip(lags, V2_LAGS[rho]))
    assert all(a > b for a, b in zip(lags, lags[1:]))                    # Lag fällt in k


def test_lag_waechst_in_rho_und_p_w_gleich_0_ist_1_minus_rho():
    l1 = [Q.lag1_exact(r) for r in (0.3, 0.5, 0.7, 0.9)]
    assert all(a < b for a, b in zip(l1, l1[1:]))
    for rho in (0.3, 0.8):
        assert abs(float(Q.mm1_stationary_cdf(rho)(0.0)) - (1 - rho)) < 1e-15


def test_lag1_als_bruch_gleich_gleitkomma():
    for rho in (Fr(1, 2), Fr(4, 5), Fr(9, 10)):
        assert abs(float(Q.lag1_exact(rho)) - Q.lag1_exact(float(rho))) < 1e-12
    assert (Q.lag1_exact(Fr(1, 2)), Q.lag1_exact(Fr(4, 5)), Q.lag1_exact(Fr(9, 10))) == (Fr(7, 9), Fr(26, 27), Fr(207, 209))


@pytest.mark.parametrize("rho,soll", [(0.5, 159.83), (0.8, 1147.88), (0.9, 4669.44)])
def test_shewhart_geschlossene_loesung_gleich_kette_und_tafel(rho, soll):
    cl = Q.shewhart_closed(rho)
    assert abs(cl["arl"] - soll) < 0.01
    assert abs(Q.shewhart_chain_arl(rho, n=800) - cl["arl"]) <= 1e-3 * cl["arl"]
    assert Q.shewhart_closed_residual(rho) < 1e-9                         # Integralgleichung an drei inneren Punkten erfüllt


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.9])
def test_shewhart_p100_gegen_simulation_im_band(rho):
    runs = 60000
    out = SC.mm1_runs(rho, runs, 31, 100, True, False, check=False)["shewhart"]
    hit = float((out <= 100).mean())
    exakt = Q.shewhart_real_p100(rho)
    assert abs(hit - exakt) <= 4 * math.sqrt(exakt * (1 - exakt) / runs)


def test_shewhart_iid_sicht_gleich_kehrwert_der_ueberschreitung():
    for rho, soll in ((0.5, 44.31), (0.8, 52.59), (0.9, 54.07)):
        assert abs(Q.shewhart_iid_arl(rho) - soll) < 0.01


def test_shewhart_mittel_gegen_simulation_rho_05():
    out = SC.mm1_runs(0.5, 3000, 32, 3000, True, False, check=False)["shewhart"]
    se = out.std(ddof=1) / math.sqrt(out.size)
    assert (out > 3000).sum() == 0 and abs(out.mean() - Q.shewhart_closed(0.5)["arl"]) <= 4 * se


def test_lindley_vektorisiert_gleich_schleife():
    st = N.Stream(3)
    x = st.exponential((30, 400), 1.0) - st.exponential((30, 400), 0.8)
    w0 = np.where(st.uniform(30) < 0.8, st.exponential(30, 0.2), 0.0)
    ref, w = np.empty_like(x), w0.copy()
    for t in range(400):
        w = np.maximum(0.0, w + x[:, t])
        ref[:, t] = w
    assert np.max(np.abs(SC.lindley_paths(x, w0) - ref)) < 1e-9


@pytest.mark.parametrize("rho,soll", [(0.5, (87.31, 70.22)), (0.8, (101.43, 150.14)), (0.9, (104.02, 359.89))])
def test_cusum_iid_sicht_und_wirkliche_reihe_gegen_tafel(rho, soll):
    assert abs(Q.cusum_iid_arl(rho) - soll[0]) <= 2e-3 * soll[0]
    assert abs(Q.cusum_real_arl(rho, ms=(16, 32), n=192) - soll[1]) <= 3e-3 * soll[1]


@pytest.mark.parametrize("rho,seed", [(0.5, 41), (0.8, 42)])
def test_cusum_wirkliche_reihe_gegen_simulation_im_band(rho, seed):
    exakt = Q.cusum_real_arl(rho, ms=(16, 32), n=192)
    out = SC.mm1_runs(rho, 4000, seed, 3000, False, True, check=False)["cusum"]
    se = out.std(ddof=1) / math.sqrt(out.size)
    assert (out > 3000).sum() == 0 and abs(out.mean() - exakt) <= 4 * se


def test_cusum_iid_sicht_gegen_simulation_unabhaengiger_ziehungen():
    """Die iid-Sicht ist ein CUSUM auf unabhängigen Ziehungen aus der Randverteilung (Atom 1 − ρ bei 0, sonst Exponential mit Rate 1 − ρ)."""
    rho, runs, t = 0.8, 6000, 2000
    st = N.Stream(51)
    mean, var = Q.mm1_moments(rho)
    sd = math.sqrt(var)
    w = np.where(st.uniform((runs, t)) < rho, st.exponential((runs, t), 1 - rho), 0.0)
    rl = SC.first_alarm(SC.cusum_paths((w - mean) / sd) >= H)
    se = rl.std(ddof=1) / math.sqrt(runs)
    assert abs(rl.mean() - Q.cusum_iid_arl(rho)) <= 4 * se


def test_cusum_verteilung_p100_gegen_simulation():
    rho = 0.8
    d = Q.cusum_real_distribution(rho)
    out = SC.mm1_runs(rho, 40000, 52, 100, False, True, check=False)["cusum"]
    hit = float((out <= 100).mean())
    assert abs(hit - d["p100"]) <= 4 * math.sqrt(hit * (1 - hit) / 40000) + 1.5e-3


def test_schiefe_allein_senkt_abhaengigkeit_hebt_und_senkt_keine_feste_richtung():
    iid = {r: Q.cusum_iid_arl(r) for r in (0.5, 0.8, 0.9)}
    real = {r: Q.cusum_real_arl(r, ms=(16, 32), n=192) for r in (0.5, 0.8, 0.9)}
    assert all(80 < v < 110 for v in iid.values())                          # die Schiefe allein: 87 bis 104
    assert real[0.5] < iid[0.5] and real[0.8] > iid[0.8] and real[0.9] > iid[0.9]


# ----------------------------------------------------------------------------------------------------------------------------------------- Gate
def _birth_death(lam, c=2, k=6, mu=1.0):
    pi = np.ones(k + 1)
    for n in range(1, k + 1):
        pi[n] = pi[n - 1] * lam / (mu * min(n, c))
    return pi / pi.sum()


def test_gate_randverteilung_gegen_geburts_und_sterbeprozess_und_little():
    """Zweiter Weg ohne die Erlang-Mischung: Gleichgewicht des Geburts-und-Sterbeprozesses; mittlere Wartezeit aus Little (W_q = L_q/λ_eff)."""
    lam = 1.6
    ex = Q.gate_exact(lam)
    pi = _birth_death(lam)
    assert abs(ex["loss"] - pi[6]) < 1e-12 and abs(ex["loss"] - 0.0760) < 5e-5
    lq = sum((n - 2) * pi[n] for n in range(3, 7))
    wq = lq / (lam * (1 - pi[6]))
    assert abs(ex["mean"] - wq) < 1e-12 and abs(ex["mean"] - 0.659) < 5e-4
    assert abs(ex["sd"] - 0.898) < 5e-4 and abs(ex["skew"] - 1.738) < 5e-4
    acc = pi[:6] / (1 - pi[6])
    assert abs(ex["p0"] - acc[:2].sum()) < 1e-12


def test_gate_simulation_gegen_exakte_randverteilung_im_band():
    lam = 1.6
    ex = Q.gate_exact(lam)
    w = Q.gate_waits(N.Stream(61), 300, 3000, lam, 300)
    je_lauf = w.mean(axis=1)
    se = je_lauf.std(ddof=1) / math.sqrt(je_lauf.size)
    assert abs(je_lauf.mean() - ex["mean"]) <= 4 * se
    p0 = float((w == 0).mean())
    assert abs(p0 - ex["p0"]) < 0.01
    assert w.max() > 0 and np.all(w >= 0)


def test_gate_iid_ziehungen_haben_die_exakte_randverteilung():
    ex = Q.gate_exact(1.6)
    x = SC.gate_iid_draws(N.Stream(62), 100, 4000, ex)
    assert abs(x.mean() - ex["mean"]) < 4 * ex["sd"] / math.sqrt(x.size) and abs(x.std() - ex["sd"]) < 0.01
    assert abs(float((x == 0).mean()) - ex["p0"]) < 0.005
    p_sf = float((x > ex["mean"] + 3 * ex["sd"]).mean())
    assert abs(p_sf - 1 / Q.gate_shewhart_iid_arl(ex)) < 4 * math.sqrt(p_sf / x.size) + 1e-4


def test_gate_shewhart_iid_sicht_exakt_60_1():
    assert abs(Q.gate_shewhart_iid_arl(Q.gate_exact(1.6)) - 60.1) < 0.05
