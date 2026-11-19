"""Orakel: Shewhart-Karte auf AR(1). Der Prüfling ist die Zellenkette mit Richardson (`abw_chain`); zweite Wege: Integralgleichung (Nyström), Geometrie bei φ = 0,
Quadratur gegen Einschluss-Ausschluss für die Alarm-Klumpen, Simulation (SplitMix64, Band von 4 Standardfehlern) und Symmetrie in φ."""

import math

import numpy as np
import pytest
from scipy.special import ndtr

import abw_chain as CH
import abw_normal as N
import abw_scenario as SC

P = 2 * float(ndtr(-3.0))


@pytest.mark.parametrize("phi,soll", [(0.0, 370.398), (0.3, 376.383), (0.6, 419.377), (0.9, 831.783), (-0.6, 419.377)])
def test_zellenkette_gleich_integralgleichung_auf_5e_4(phi, soll):
    kette = CH.shewhart_arl_two_sided(phi)
    nystrom = CH.shewhart_nystrom(phi, -3.0, 3.0, n=140)
    assert abs(kette - nystrom) <= 5e-4 * nystrom
    assert abs(kette - soll) < 5e-3                                   # Tafelwerte der Vormessung (V14 §0, §1)


def test_phi_null_gleich_kehrwert_der_alarmrate_und_median_geometrisch():
    assert abs(CH.shewhart_arl_two_sided(0.0) - 1 / P) < 1e-7
    d = CH.shewhart_distribution(0.0, n=240, nmax=4000)
    assert d["median"] == math.ceil(math.log(0.5) / math.log(1 - P))   # geometrische Verteilung: kleinstes t mit 1 − (1 − p)^t ≥ ½
    assert abs(d["p100"] - (1 - (1 - P) ** 100)) < 1e-6


def test_symmetrie_in_phi_gleichheit_auf_1e_9_nur_zweiseitig():
    for phi in (0.2, 0.55, 0.9):
        assert abs(CH.shewhart_arl_two_sided(phi, n=240) - CH.shewhart_arl_two_sided(-phi, n=240)) < 1e-9 * CH.shewhart_arl_two_sided(phi, n=240)
    assert abs(CH.shewhart_arl_one_sided(0.6, n=240) - CH.shewhart_arl_one_sided(-0.6, n=240)) > 10     # einseitig nicht (436,6 gegen 383,2)


def test_einseitig_grenze_und_werte():
    assert abs(CH.one_sided_limit() - 2.7822) < 5e-5
    assert abs(CH.shewhart_arl_one_sided(0.0) - 370.398) < 5e-3
    assert abs(CH.shewhart_arl_one_sided(0.9) - 924.029) < 0.05 and abs(CH.shewhart_arl_one_sided(-0.9) - 603.037) < 0.05


@pytest.mark.parametrize("phi", [0.0, 0.3, 0.6, 0.9, -0.6])
def test_alarmpaar_quadratur_gleich_einschluss_ausschluss_der_kette(phi):
    q, pin = CH.shewhart_chain(phi, -3.0, 3.0, 960)
    beide_innen = float(pin @ q.sum(axis=1))
    beide_aussen = 1 - 2 * (1 - P) + beide_innen
    quad = CH.alarm_pair_probability(phi)
    assert abs(quad - beide_aussen) <= 3e-4 * quad + 1e-9


def test_klumpen_kennzahlen_gegen_tafel():
    c9, c6, c0 = CH.clusters(0.9), CH.clusters(0.6), CH.clusters(0.0)
    assert abs(c9["q"] - 0.4522) < 5e-5 and abs(c6["q"] - 0.1035) < 5e-5 and abs(c0["q"] - 0.0027) < 5e-5
    assert abs(c9["cluster_length"] - 1.825) < 5e-4 and abs(c9["start_rate_inv"] - 676.1) < 0.05 and abs(c9["share"] - 0.66) < 0.005
    assert abs(c6["start_rate_inv"] - 413.1) < 0.05
    assert abs(c0["q"] - P) < 1e-9                                      # φ = 0: q = p (Alarme unabhängig)


def test_lauflaengenverteilung_monoton_und_endet_bei_eins():
    q, pin = CH.shewhart_chain(0.9, -3.0, 3.0, 240)
    surv = CH.chain_survival(q, pin, 30000)
    assert surv[0] == 1.0 and np.all(np.diff(surv) <= 1e-15) and surv[-1] < 1e-6


def test_verteilung_tafelwerte_phi_09():
    d = CH.shewhart_distribution(0.9)
    assert abs(d["p100"] - 0.1151) < 5e-5 and (d["median"], d["q05"], d["q95"]) == (576, 41, 2494)


@pytest.mark.parametrize("phi,seed", [(0.0, 1), (0.6, 2), (0.9, 3), (-0.6, 4)])
def test_simulation_p100_im_band(phi, seed):
    """P(T ≤ 100) exakt gegen Simulation (100 000 Reihen der Länge 100, SplitMix64): Band 4 Standardfehler."""
    runs = 100_000
    _, u = SC.ar1_series(N.Stream(seed), runs, 100, phi)
    hit = float((np.abs(u) > 3.0).any(axis=1).mean())
    exakt = CH.shewhart_distribution(phi, n=480, nmax=200)["p100"]
    se = math.sqrt(exakt * (1 - exakt) / runs)
    assert abs(hit - exakt) <= 4 * se


@pytest.mark.parametrize("phi,seed", [(0.9, 7), (0.6, 8)])
def test_simulation_mittel_im_band(phi, seed):
    res = SC.ar1_runs(phi, 2000, seed, 12000, check=False)
    s = res.summary["shewhart2"]
    exakt = CH.shewhart_arl_two_sided(phi)
    assert s["censored"] <= 2 and abs(s["mean"] - exakt) <= 4 * s["se"]


def test_arl_steigt_in_der_grenze_zufallsinstanzen():
    rng = np.random.default_rng(20261119)
    for _ in range(20):
        phi = float(rng.uniform(-0.9, 0.9))
        lims = np.sort(rng.uniform(1.5, 3.5, 3))
        v = [CH.shewhart_arl(phi, -lim, lim, 120) for lim in lims]
        assert v[0] < v[1] < v[2]


def test_kette_aendert_sich_bei_halber_zellbreite_um_weniger_als_das_band():
    for phi in (0.3, 0.9):
        a = CH.shewhart_arl(phi, -3.0, 3.0, 240, richardson=False)
        b = CH.shewhart_arl(phi, -3.0, 3.0, 480, richardson=False)
        assert abs(a - b) < 2e-3 * b


def test_gleicher_seed_gleiche_ausgabe():
    a = SC.ar1_runs(0.6, 200, 5, 3000).lengths
    b = SC.ar1_runs(0.6, 200, 5, 3000).lengths
    assert all(np.array_equal(a[k], b[k]) for k in a)
    c = SC.ar1_runs(0.6, 200, 6, 3000).lengths
    assert not np.array_equal(a["shewhart2"], c["shewhart2"])
