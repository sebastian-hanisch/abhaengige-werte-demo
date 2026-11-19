"""Orakel: Schiefe bei unabhängigen Werten. Prüflinge: Shewhart aus der Verteilungsfunktion, CUSUM-Kette, Vorzeichen-CUSUM mit Brüchen. Zweite Wege: Quadratur der Dichte, Simulation
(SplitMix64-Gleichverteilung, Umkehrfunktion; Band von 4 Standardfehlern), Zufallsbewegung für den Vorzeichen-CUSUM, Handwerte."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import integrate, stats

import abw_chain as CH
import abw_normal as N
import abw_scenario as SC
import abw_skew as SK

D = SK.DISTS


def test_exponential_e_hoch_4_und_gamma4_geschlossen_gegen_quadratur():
    assert abs(SK.shewhart_two_sided(D["expon"]) - math.exp(4)) < 1e-9
    assert abs(SK.shewhart_two_sided(D["gamma4"]) - math.exp(10) * 3 / 683) < 1e-9
    p, _ = integrate.quad(lambda x: x ** 3 * math.exp(-x) / 6, 10, math.inf, epsabs=1e-16, epsrel=1e-12)    # P(X > 10) für Gamma(4, 1): zweiter Weg per Quadratur
    assert abs(1 / p - SK.gamma4_closed()) < 1e-6
    p, _ = integrate.quad(lambda x: math.exp(-x), 4, math.inf)
    assert abs(1 / p - SK.exponential_closed()) < 1e-9


def test_lognormal_und_t5_gegen_quadratur_der_dichte():
    ln = stats.lognorm(s=0.5)
    mu, sd = ln.mean(), ln.std()
    lo, hi = mu - 3 * sd, mu + 3 * sd
    assert lo < 0                                                        # die untere Grenze liegt außerhalb des Trägers
    p, _ = integrate.quad(lambda x: math.exp(-0.5 * (math.log(x) / 0.5) ** 2) / (x * 0.5 * math.sqrt(2 * math.pi)), hi, math.inf, epsabs=1e-14)
    assert abs(1 / p - SK.shewhart_two_sided(D["lognorm"])) < 1e-3
    t5 = stats.t(5)
    s5 = t5.std()
    dens = lambda x: math.gamma(3) / (math.sqrt(5 * math.pi) * math.gamma(2.5)) * (1 + x * x / 5) ** -3  # noqa: E731
    p, _ = integrate.quad(dens, 3 * s5, math.inf, epsabs=1e-14)
    assert abs(1 / (2 * p) - SK.shewhart_two_sided(D["t5"])) < 1e-3


@pytest.mark.parametrize("key,soll", [("expon", 54.598), ("gamma4", 96.749), ("lognorm", 65.01), ("t5", 85.29), ("normal", 370.398)])
def test_shewhart_tafelwerte(key, soll):
    assert abs(SK.shewhart_two_sided(D[key]) - soll) < 0.01


@pytest.mark.parametrize("key,soll", [("expon", 104.95), ("gamma4", 159.01), ("lognorm", 120.08), ("t5", 275.98)])
def test_cusum_kette_tafelwerte(key, soll):
    assert abs(SK.cusum_arl(D[key]) - soll) <= 1e-3 * soll


@pytest.mark.parametrize("key,seed,runs", [("expon", 71, 8000), ("lognorm", 72, 8000), ("t5", 73, 3000)])
def test_cusum_kette_gleich_simulation_im_band(key, seed, runs):
    d = D[key]
    t = 3000
    u = N.Stream(seed).uniform((runs, t))
    x = (d.rv.ppf(u) - d.mean) / d.sd
    rl = SC.first_alarm(SC.cusum_paths(x) >= SK.H_ONE)
    se = rl.std(ddof=1) / math.sqrt(runs)
    assert (rl > t).sum() == 0 and abs(rl.mean() - SK.cusum_arl(d)) <= 4 * se


def test_vorzeichen_cusum_brueche_und_gleichungssystem_aus_drei_zeilen():
    assert [SK.sign_cusum_arl(h) for h in (3, 8, 9)] == [14, 266, 442]
    assert all(isinstance(SK.sign_cusum_arl(h), Fr) for h in (3, 8, 9))
    # H = 3 von Hand: E2 = 1 + E0/2, E1 = 1 + (E2 + E0)/2, E0 = 1 + (E1 + E0)/2
    e0 = Fr(14)
    e2 = 1 + e0 / 2
    e1 = 1 + (e2 + e0) / 2
    assert e0 == 1 + (e1 + e0) / 2


def test_vorzeichen_cusum_gleitkomma_gleich_bruch_und_gegen_zufallsbewegung():
    assert abs(SK.sign_cusum_arl(8, 0.5) - 266.0) < 1e-6
    runs, t = 60000, 600
    st = N.Stream(74)
    step = np.where(st.uniform((runs, t)) < 0.5, 1.0, -2.0)
    c = np.zeros(runs)
    rl = np.full(runs, t + 1)
    alive = np.ones(runs, bool)
    for i in range(t):
        c = np.maximum(0.0, c + step[:, i])
        hit = alive & (c >= 3)
        rl[hit] = i + 1
        alive &= ~hit
    se = rl.std(ddof=1) / math.sqrt(runs)
    assert abs(rl.mean() - 14) <= 4 * se


def test_vorzeichen_cusum_ist_verteilungsfrei_gleiche_zahl_bei_jeder_stetigen_verteilung():
    """Der Median halbiert jede stetige Verteilung: P(Schritt +1) = ½ für Normal, Exponential, Gamma, Lognormal, t."""
    for d in D.values():
        assert abs(d.rv.sf(float(d.rv.median())) - 0.5) < 1e-12


def test_artefakt_exponential_nach_lageverschiebung_meldet_nach_genau_h():
    d = D["expon"]
    assert SK.sign_cusum_note(d, 2.0) and SK.sign_cusum_shift_arl(d, 2.0, 9) == 9.0
    assert d.rv.median() < 2.0                                             # alle verschobenen Werte liegen über dem alten Median ln 2
    # bei normalen Werten ist es kein Artefakt: ARL₁ des Vorzeichen-CUSUM (1 σ) 15,45 gegen CUSUM 8,58
    nrm = D["normal"]
    assert not SK.sign_cusum_note(nrm, 1.0)
    assert abs(SK.sign_cusum_shift_arl(nrm, 1.0, 9) - 15.45) < 0.01 and abs(SK.cusum_arl(nrm, shift=1.0) - 8.58) < 0.01


def test_nacheichung_gibt_arl0_370_398():
    for key in ("expon", "lognorm"):
        d = D[key]
        assert abs(SK.shewhart_two_sided(d, SK.shewhart_limit_for(d)) - CH.NOM) < 1e-6
    d = D["gamma4"]
    assert abs(SK.cusum_arl(d, SK.cusum_h_for(d)) - CH.NOM) < 0.5
