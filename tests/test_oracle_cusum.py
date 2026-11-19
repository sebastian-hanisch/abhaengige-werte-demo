"""Orakel: einseitiger CUSUM auf AR(1). Der Prüfling ist die zweidimensionale Zellenkette (`abw_cusum`, Richardson über zwei Gitter); zweite Wege: bei φ = 0 die eindimensionale
Integralgleichung (Page, Nyström) auf 0,1 %, bei φ ≠ 0 die Simulation (SplitMix64, Band von 4 Standardfehlern); dazu die Spiegelungsformel gegen die Schleife und Eigenschaften."""

import math

import numpy as np
import pytest

import abw_chain as CH
import abw_cusum as CU
import abw_normal as N
import abw_scenario as SC

H = CH.cusum_h_for_arl(0.5)


def test_schwelle_h_fuer_arl_370_398():
    assert abs(H - 4.0965) < 5e-5
    assert abs(CH.cusum_iid_arl(0.5, H) - CH.NOM) < 1e-6
    assert abs(CH.cusum_iid_arl(0.5, 4.0965) - 370.4) < 0.05      # die in Text und README gerundete Schwelle


def test_zweidimensionale_kette_bei_phi_0_gleich_eindimensionaler_integralgleichung_auf_0_1_prozent():
    ein = CH.cusum_iid_arl(0.5, H)
    zwei = CU.ar1_arl(0.0, H)                                       # m = 16 und 32, n = 128, Richardson
    assert abs(zwei - ein) <= 1e-3 * ein
    assert abs(zwei - 370.431) < 0.01                               # Tafelwert der Vormessung (V14 0.5)


def test_brook_evans_kette_gleich_integralgleichung_bei_phi_0():
    from scipy.special import ndtr
    kette = CH.cusum_chain_arl(lambda t: ndtr(np.asarray(t, float)), 0.5, H, 400)
    assert abs(kette - CH.NOM) <= 2e-4 * CH.NOM


@pytest.mark.parametrize("phi,soll", [(0.3, 114.225), (0.6, 62.171), (0.9, 63.399), (0.1, 229.798)])
def test_tafelwerte_der_vormessung(phi, soll):
    assert abs(CU.ar1_arl(phi, H) - soll) <= 1.5e-3 * soll


@pytest.mark.parametrize("phi,seed,runs", [(0.3, 11, 12000), (0.6, 12, 12000), (0.9, 13, 12000)])
def test_kette_gleich_simulation_im_band(phi, seed, runs):
    exakt = CU.ar1_arl(phi, H)
    _, u = SC.ar1_series(N.Stream(seed), runs, 2500, phi)
    rl = SC.first_alarm(SC.cusum_paths(u) >= H)
    assert (rl > 2500).sum() == 0
    se = rl.std(ddof=1) / math.sqrt(runs)
    assert abs(rl.mean() - exakt) <= 4 * se


def test_negatives_phi_minus_03_kette_gleich_simulation_im_band():
    exakt = CU.ar1_arl(-0.3, H)
    assert abs(exakt - 3798.2) < 1.0
    runs = 1200
    _, u = SC.ar1_series(N.Stream(14), runs, 30000, -0.3)
    rl = SC.first_alarm(SC.cusum_paths(u) >= H)
    se = rl.std(ddof=1) / math.sqrt(runs)
    assert abs(rl.mean() - exakt) <= 4 * se


def test_phi_minus_06_ist_nur_kette_nicht_durch_simulation_geprueft():
    """Die Kette gibt 284 222 (767fach); eine Simulation dieser Länge ist nicht möglich: die Zahl steht mit „nur Kette“ in App und README."""
    assert CU.ar1_arl(-0.6, H) > 2.5e5


def test_nicht_monoton_minimum_bei_0_8_im_gitter_und_anstieg_danach():
    werte = {p: CU.ar1_arl(p, H) for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
    ks = sorted(werte)
    assert all(werte[a] > werte[b] for a, b in zip(ks[:-2], ks[1:-1]))      # fällt bis 0,8
    assert werte[0.9] > werte[0.8]                                          # steigt danach wieder
    assert min(werte, key=werte.get) == 0.8
    assert all(werte[p] < CH.NOM for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))   # 0 < φ ≤ 0,6: unter 370,4


def test_gitter_aenderung_unter_dem_genannten_band():
    grob = CU.ar1_arl(0.6, H, ms=(8, 16), n=64)
    mittel = CU.ar1_arl(0.6, H)
    fein = CU.ar1_arl(0.6, H, ms=(24, 48), n=192)
    assert abs(grob - fein) < 0.03 * fein and abs(mittel - fein) < 0.005 * fein


def test_arl0_steigt_in_h_und_in_k_zufallsinstanzen():
    """Der Plan nennt „fällt in k“; für die ARL₀ gilt das Gegenteil (ein größerer Referenzwert lässt die Summe stärker zurücklaufen): sie steigt in k."""
    rng = np.random.default_rng(1119)
    for _ in range(6):
        phi = float(rng.uniform(0.1, 0.7))
        hs = np.sort(rng.uniform(2.5, 5.0, 2))
        a, b = (CU.ar1_arl(phi, float(h), ms=(8, 16), n=64) for h in hs)
        assert a < b
        k1, k2 = np.sort(rng.uniform(0.2, 0.8, 2))
        a, b = (CU.ar1_arl(phi, 4.0, k=float(k), ms=(8, 16), n=64) for k in (k1, k2))
        assert a < b


def test_spiegelungsformel_gleich_schleife_und_start_null():
    _, u = SC.ar1_series(N.Stream(5), 60, 500, 0.8)
    assert np.max(np.abs(SC.cusum_paths(u) - SC.cusum_paths_loop(u))) < 1e-9
    assert np.all(SC.cusum_paths(u) >= 0)
    unten = SC.cusum_paths(u, lower=True)
    assert np.max(np.abs(unten - SC.cusum_paths_loop(-u))) < 1e-9


def test_ar1_series_gleich_schleife():
    st = N.Stream(8)
    u0, u = SC.ar1_series(st, 40, 300, 0.7)
    st2 = N.Stream(8)
    z0 = st2.normal(40)
    eps = st2.normal((40, 300))
    assert np.array_equal(z0, u0)
    s = math.sqrt(1 - 0.49)
    prev, ref = z0.copy(), np.empty((40, 300))
    for t in range(300):
        prev = 0.7 * prev + s * eps[:, t]
        ref[:, t] = prev
    assert np.max(np.abs(ref - u)) < 1e-12


def test_zweiseitiger_cusum_ist_simulation_naeherung_gilt_nur_bis_phi_03():
    """Näherung „zweiseitig = einseitig/2“ bei h = 4,77: Abweichung gegen die Simulation wächst mit φ (kleine Probe: 6000 Läufe, tmax 4000)."""
    abw = {}
    for i, phi in enumerate((0.3, 0.9)):
        up = CU.ar1_arl(phi, 4.77)
        rl = SC.cusum_two_sided_runs(phi, 6000, 50 + i, 4.77, 4000)
        abw[phi] = (up / 2 - rl.mean()) / rl.mean()
    assert abs(abw[0.3]) < 0.04 and abw[0.9] > 0.2


def test_verteilung_der_kette_gegen_simulation_p100():
    d = CU.ar1_distribution(0.6, H)
    runs = 40000
    _, u = SC.ar1_series(N.Stream(21), runs, 100, 0.6)
    hit = float((SC.cusum_paths(u) >= H).any(axis=1).mean())
    se = math.sqrt(hit * (1 - hit) / runs)
    assert abs(hit - d["p100"]) <= 4 * se + 1.2e-3                        # Gitterfehler der Verteilungsfunktion rund 0,001 (m = 24, n = 128)


def test_gitterfehler_der_verteilung_bei_phi_0_erklaert_die_abweichung_2312_gegen_2302():
    """Die zweidimensionale Kette (m = 24, n = 128) gibt P(T ≤ 100) = 0,2312 bei φ = 0, die eindimensionale 0,2302: die 2D-Zahl führt den Gitterfehler (< 0,002)."""
    zwei = CU.ar1_distribution(0.0, H)["p100"]
    sf = CH.cusum_chain_survival(lambda t: __import__("scipy.special", fromlist=["ndtr"]).ndtr(np.asarray(t, float)), 0.5, H, 800, 100)
    ein = 1 - sf[100]
    assert abs(ein - 0.2302) < 3e-4 and 5e-4 < abs(zwei - ein) < 2e-3
