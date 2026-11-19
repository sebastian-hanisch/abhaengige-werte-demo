"""Orakel: Segmentierung. Prüfling: optimale Zerlegung durch dynamische Programmierung (`abw_segment`); zweiter Weg: Aufzählung aller Zerlegungen mit Mindestlänge (Brute Force)."""

import math

import numpy as np
import pytest

import abw_reference as REF
import abw_segment as SG


@pytest.mark.parametrize("n,minlen,beta", [(10, 2, 1.0), (12, 2, 3.0), (11, 3, 2.0), (9, 1, 0.5)])
def test_dp_gleich_aufzaehlung_zufallsinstanzen(n, minlen, beta):
    rng = np.random.default_rng(n * 100 + minlen)
    for _ in range(25):
        x = rng.normal(size=n) + np.where(np.arange(n) >= n // 2, rng.uniform(-1.5, 1.5), 0.0)
        cuts, f = SG.optimal_partition(x[None, :], [beta], minlen)
        best, _ = SG.brute_force(x, beta, minlen)
        assert abs(float(f[0, 0]) - best) < 1e-9
        b = [0] + cuts[0][0] + [n]
        assert all(c - a >= minlen for a, c in zip(b[:-1], b[1:]))


def test_dp_findet_den_offensichtlichen_bruch_und_keinen_bei_konstanter_reihe():
    x = np.concatenate([np.zeros(20), 5 * np.ones(20)])
    cuts, _ = SG.optimal_partition(x[None, :], [2 * math.log(40)], 5)
    assert cuts[0][0] == [20]
    cuts, _ = SG.optimal_partition(np.zeros((1, 30)), [2 * math.log(30)], 5)
    assert cuts[0][0] == []


def test_reihe_kuerzer_als_mindestlaenge_hat_keinen_bruch():
    cuts, _ = SG.optimal_partition(np.arange(4.0)[None, :], [1.0], 5)
    assert cuts[0][0] == []


def test_fehlbrueche_neu_gerechnet_gegen_die_gefrorene_tabelle_im_band():
    """φ = 0,6, Rohdaten und Residuen, c = 2 (150 Reihen, anderer Zufalls-Seed): Anteil gegen die gespeicherte Zeile, Band 4 Standardfehler beider Seiten."""
    for resid in (False, True):
        neu = SG.false_breaks(0.6, 150, 4242, resid)
        alt = REF.segmentation(0.6, resid)
        p, q = neu["share"][0] / 100, alt["share"][0] / 100
        se = math.sqrt(max(p * (1 - p), 0.0005) / 150 + max(q * (1 - q), 0.0005) / alt["runs"])
        assert abs(p - q) <= 4 * se


def test_fehlbrueche_wachsen_mit_phi_und_residuen_bleiben_klein():
    roh = [REF.segmentation(p, False)["share"][0] for p in (0.0, 0.3, 0.6, 0.9)]
    res = [REF.segmentation(p, True)["share"][0] for p in (0.0, 0.3, 0.6, 0.9)]
    assert roh[0] < roh[1] < roh[2] < roh[3] and max(res) <= 8.0 and roh[0] <= 8.0
