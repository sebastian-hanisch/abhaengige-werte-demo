"""Orakel: Residuenkarte und Nacheichung. Prüflinge: ARL₀ und ARL₁ der Residuenkarte (Formel und Integralgleichung), Nacheichung (Grenze L_φ, Schwelle h_φ). Zweite Wege: Zellenkette der
Residuenkarte auf der AR(1)-Reihe (Alarm |u_t − φ·u_{t-1}| > 3s), Nyström-Integralgleichung für L_φ, Simulation der Reihe (SplitMix64, Band von 4 Standardfehlern)."""

import math

import numpy as np
import pytest
from scipy.special import ndtr

import abw_chain as CH
import abw_reference as REF
import abw_scenario as SC

H = CH.cusum_h_for_arl(0.5)


def _residual_chain_arl0(phi, n=240, half_range=8.0):
    """Zweiter Weg zur ARL₀ der Residuenkarte: Zellenkette auf dem LETZTEN WERT; Alarm, wenn |u_t − φ·u_{t-1}| > 3s (ohne die Innovation zu standardisieren)."""
    s = math.sqrt(1 - phi ** 2)
    e = np.linspace(-half_range, half_range, n + 1)
    c = 0.5 * (e[:-1] + e[1:])
    lo = np.maximum(e[None, :-1], (phi * c - 3 * s)[:, None])
    hi = np.minimum(e[None, 1:], (phi * c + 3 * s)[:, None])
    q = np.where(hi > lo, ndtr((hi - phi * c[:, None]) / s) - ndtr((lo - phi * c[:, None]) / s), 0.0)
    pin = np.diff(ndtr(e))
    return 1 + float(pin @ np.linalg.solve(np.eye(n) - q, np.ones(n)))


@pytest.mark.parametrize("phi", [0.3, 0.6, 0.9, -0.5])
def test_residuenkarte_arl0_ist_370_398_fuer_jedes_phi_zellenkette(phi):
    assert abs(_residual_chain_arl0(phi) - CH.NOM) <= 3e-3 * CH.NOM
    assert abs(CH.residual_shewhart_arl(phi, 0.0) - CH.NOM) < 1e-9      # δ = 0: ARL₁ = ARL₀ = 1/(2Φ(−3))


@pytest.mark.parametrize("phi,seed", [(0.0, 1), (0.3, 2), (0.6, 3), (0.9, 4)])
def test_residuenkarte_arl0_aus_der_simulierten_reihe_im_band(phi, seed):
    runs = 5000
    rl = SC.change_runs(phi, 0.0, "offset", "residuum", "shewhart", runs, 8000 + seed, 3500, check=False)
    se = rl.std(ddof=1) / math.sqrt(runs)
    assert (rl > 3500).sum() <= 3 and abs(rl.mean() - CH.NOM) <= 4 * se


def test_residuen_cusum_arl0_gleich_370_398_integralgleichung():
    assert abs(CH.residual_cusum_arl(H, 0.5, 0.0, 0.0) - CH.NOM) < 1e-4


@pytest.mark.parametrize("art,phi,delta,seed", [("offset", 0.6, 1.0, 11), ("offset", 0.9, 2.0, 12), ("process", 0.9, 1.0, 13)])
def test_arl1_der_residuenkarte_formel_gegen_simulation(art, phi, delta, seed):
    runs = 4000
    for card, exakt in (("shewhart", CH.residual_shewhart_arl(phi, delta, art)),
                        ("cusum", CH.residual_cusum_arl(H, 0.5, *CH.residual_innovation_means(phi, delta, art)))):
        rl = SC.change_runs(phi, delta, art, "residuum", card, runs, 100 * seed, 3000, check=False)
        se = rl.std(ddof=1) / math.sqrt(runs)
        assert (rl > 3000).sum() <= 2 and abs(rl.mean() - exakt) <= 4 * se, (art, phi, delta, card)


@pytest.mark.parametrize("art,phi,delta,seed", [("offset", 0.6, 1.0, 21), ("process", 0.9, 1.0, 22)])
def test_rohdatenkarte_nominell_ketten_gegen_simulation(art, phi, delta, seed):
    kw = dict(zshift=delta) if art == "offset" else dict(drift=(1 - phi) * delta, stat_mean=(1 - phi) * delta)
    exakt = CH.shewhart_arl(phi, -3.0, 3.0, 480, **kw)
    runs = 4000
    rl = SC.change_runs(phi, delta, art, "roh", "shewhart", runs, 100 * seed, 3000, check=False)
    se = rl.std(ddof=1) / math.sqrt(runs)
    assert abs(rl.mean() - exakt) <= 4 * se


def test_konvention_des_ersten_residuums_150_00_gegen_152_69():
    assert abs(CH.residual_shewhart_arl(0.6, 1.0, "offset") - 150.00) < 0.01
    m1, m2 = CH.residual_innovation_means(0.6, 1.0, "offset")
    ohne = 1 + (1 - CH.alarm_prob(1.0)) / CH.alarm_prob(m2)
    assert abs(ohne - 152.69) < 0.01 and ohne > CH.residual_shewhart_arl(0.6, 1.0, "offset")


def test_nacheichung_grenze_gegen_integralgleichung_und_tafel():
    for phi, soll in ((0.3, 2.9951), (0.6, 2.9605), (0.9, 2.7112)):
        lim = CH.calibrated_limit(phi)
        assert abs(lim - soll) < 5e-4
        assert abs(CH.shewhart_nystrom(phi, -lim, lim, n=220) - CH.NOM) < 2e-3          # zweiter Weg
        assert abs(REF.calibration(phi)["limit"] - lim) < 2e-4                          # gefrorene Zeile


def test_nacheichung_schwelle_gefroren_gegen_kette_und_simulation_aus_der_tabelle():
    for phi in (0.3, 0.6, 0.9):
        c = REF.calibration(phi)
        assert abs(c["arl0_at_h_fine_fine_grid"] - CH.NOM) < 0.05                      # Kette auf dem feinen Gitter
        assert abs(c["sim_arl0"] - CH.NOM) <= 4 * c["sim_se"] and c["sim_runs"] == 100000 and c["sim_censored"] <= 1
    assert abs(REF.calibration(0.3)["h_fine"] - 6.114) < 0.01 and abs(REF.calibration(0.6)["h_fine"] - 9.971) < 0.01
    c9 = REF.calibration(0.9)
    assert abs(c9["h_coarse"] - 22.277) < 0.01 and c9["h_fine"] < 22.15                # die grobe Kette reproduziert 22,277; fein: 22,10
    assert c9["arl0_at_h_coarse_fine_grid"] > 373                                      # mit dem groben h liegt die ARL₀ auf dem feinen Gitter um mehr als 0,7 % zu hoch


def test_nacheichung_cusum_schwelle_phi_03_simulation_im_band():
    phi = 0.3
    h = REF.calibration(phi)["h_fine"]
    runs = 8000
    _, u = SC.ar1_series(__import__("abw_normal").Stream(444), runs, 2500, phi)
    rl = SC.first_alarm(SC.cusum_paths(u) >= h)
    se = rl.std(ddof=1) / math.sqrt(runs)
    assert abs(rl.mean() - CH.NOM) <= 4 * se


def test_messversatz_delta_2_phi_09_residuum_schneller_prozessmittel_langsamer():
    off = CH.residual_shewhart_arl(0.9, 2.0, "offset")
    cal = CH.shewhart_arl(0.9, -REF.calibration(0.9)["limit"], REF.calibration(0.9)["limit"], 480, zshift=2.0)
    assert off < cal and abs(off - 10.68) < 0.01 and abs(cal - 17.86) < 0.05
    pro = CH.residual_shewhart_arl(0.9, 2.0, "process")
    cal_p = CH.shewhart_arl(0.9, -REF.calibration(0.9)["limit"], REF.calibration(0.9)["limit"], 480, drift=0.2, stat_mean=0.2)
    assert pro > 5 * cal_p and abs(pro - 172.55) < 0.05
