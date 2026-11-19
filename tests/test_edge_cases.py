"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): φ am Rand, δ = 0, R = 1, tmax-Zensur, ρ nahe 1, sehr große Schwelle, Zufalls-Seed 0, Schwelle 1."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import abw_chain as CH
import abw_cusum as CU
import abw_queue as Q
import abw_scenario as SC
import abw_segment as SG
import abw_skew as SK
import abw_toy as T

H = 4.0965


@pytest.mark.parametrize("phi", [-0.98, -0.6, 0.0, 0.95, 0.98])
def test_phi_am_rand_endliche_zahlen_beider_karten(phi):
    s = CH.shewhart_arl_two_sided(phi, n=240)
    assert math.isfinite(s) and s >= 1.0
    c = CU.ar1_arl(phi, H, ms=(8, 16), n=64)
    assert math.isfinite(c) and c >= 1.0


def test_delta_null_arl1_gleich_arl0():
    for phi in (0.3, 0.9):
        a = CH.shewhart_arl_two_sided(phi, n=240)
        assert abs(CH.shewhart_arl(phi, -3.0, 3.0, 240, zshift=0.0) - a) < 1e-9
        assert abs(CH.shewhart_arl(phi, -3.0, 3.0, 240, drift=0.0, stat_mean=0.0) - a) < 1e-9
        c = CU.ar1_arl(phi, H, ms=(8, 16), n=64)
        assert abs(CU.ar1_arl(phi, H, kind="offset", delta=0.0, ms=(8, 16), n=64) - c) < 1e-9
        assert abs(CU.ar1_arl(phi, H, kind="process", delta=0.0, ms=(8, 16), n=64) - c) < 1e-9


def test_sehr_grosse_schwelle_und_sehr_kleine_geben_keinen_absturz():
    grob = CU.ar1_arl(0.6, 40.0, ms=(8, 16), n=64)                     # ARL jenseits von 10⁸: die Lösung verliert auf dem groben Gitter ihre Genauigkeit
    assert math.isnan(grob) or grob >= 1.0                              # nie eine unmögliche Zahl unter 1 (früher kam −1·10⁸ heraus)
    assert CU.ar1_arl(0.6, 12.0) > 500
    assert 5.0 < CU.ar1_arl(0.6, 0.2, ms=(8, 16), n=64) < 8.0          # sehr kleine Schwelle: kurze Läufe, Gitter fein genug (Zelle 0 trägt fast die ganze Masse)


def test_ein_lauf_und_seed_null():
    r = SC.ar1_runs(0.6, 1, 0, 2000)
    assert all(v.size == 1 for v in r.lengths.values()) and np.isnan(r.summary["cusum"]["se"])
    assert SC.cluster_series(0.5, 10, 0).length == 10


def test_rho_nahe_eins_geschlossene_loesung_endlich_und_lag_nahe_eins():
    cl = Q.shewhart_closed(0.95)
    assert math.isfinite(cl["arl"]) and cl["arl"] > Q.shewhart_closed(0.9)["arl"]
    assert 0.99 < Q.lag1_exact(0.95) < 1.0


def test_atom_bei_null_und_ohne_atom():
    q, pin = Q.shewhart_chain(0.5, n=100)
    assert abs(pin.sum() - (1 - 0.5 * math.exp(-0.5 * Q.shewhart_threshold(0.5)))) < 1e-12 and pin[0] > 0.5       # erste Zelle enthält das Atom 1 − ρ
    d = SK.DISTS["normal"]
    assert abs(float(d.cdf_z(-1e9))) < 1e-12                                                                      # Normalverteilung: kein Atom


def test_vorzeichen_cusum_und_diskrete_wartezeit_mit_schwelle_eins():
    assert SK.sign_cusum_arl(1) == 2
    assert T.discrete_wait(1)["arl"] == Fr(5, 2) and T.discrete_wait(1)["iid_arl"] == 2


def test_zwei_zustaende_kette_fuer_alle_regler_werte_und_summenvarianz_m_eins():
    for phi in (0.0, 0.5, 0.8):
        r = T.two_state(Fr(phi).limit_denominator(100))
        assert r["stationary"] == Fr(1, 10) and r["arl_a"] > 0 and r["arl_b"] > 0
    assert T.sum_variance(1, Fr(9, 10)) == 1


def test_segmentierung_weniger_werte_als_mindestlaenge_und_genau_mindestlaenge():
    cuts, _ = SG.optimal_partition(np.array([[1.0, 2.0, 3.0]]), [1.0], 5)
    assert cuts[0][0] == []
    cuts, _ = SG.optimal_partition(np.array([[0.0] * 5]), [1.0], 5)
    assert cuts[0][0] == []


def test_keine_zensur_bei_ausreichendem_tmax_und_zensur_bei_zu_kleinem():
    gross = SC.ar1_runs(0.9, 200, 4, 10000).summary["shewhart2"]["censored"]
    klein = SC.ar1_runs(0.9, 200, 4, 100).summary["shewhart2"]["censored"]
    assert gross <= 2 and klein > 150


def test_gate_angebot_klein_und_gross_randverteilung():
    low, high = Q.gate_exact(0.2), Q.gate_exact(6.0)
    assert low["loss"] < 1e-4 and low["p0"] > 0.97 and high["loss"] > 0.6
