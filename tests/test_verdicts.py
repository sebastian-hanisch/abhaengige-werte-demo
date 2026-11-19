"""Verdikte als reine Funktionen (Art, Text): Schwellen, jede Zahl im Text an handgesetzten Zeilen, Formate (Dezimalkomma, Tausenderleerzeichen)."""

from fractions import Fraction as Fr

import pytest

import abw_evaluation as EV


def test_zahlenformat_dezimalkomma_tausender_vorzeichen():
    assert EV.de(1147.88, 1) == "1 147,9" and EV.de(284222.327, 0) == "284 222" and EV.de(0.2302, 4) == "0,2302" and EV.de(-0.3, 1) == "-0,3"
    assert EV.de_s(0.04, 1) == "+0,0" and EV.de_s(-308.2, 1) == "-308,2"
    assert EV.pct(0.2369) == "23,7 %" and EV.factor(62.2, 370.398) == "0,17fach"
    assert EV.frac(Fr(409, 11)) == "409/11" and EV.frac(Fr(110)) == "110" and EV.phi_text(-0.3) == "−0,3" and EV.phi_text(0.05) == "0,05" and EV.phi_text(0.6) == "0,6"
    assert EV.phi_text(0.0) == "0" and EV.phi_text(0.98) == "0,98" and EV.phi_text(-0.6) == "−0,6" and EV.phi_text(0.5) == "0,5"


def test_exp1_gegenlaeufig_bei_phi_06_anker_aus_dem_plan():
    kind, t = EV.verdict_exp1(0.6, 419.377, 436.597, 62.171, 0.2126, 0.8074)
    assert kind == "warning"
    for teil in ("419,4 (1,13fach)", "0,213 (exakt)", "62,2 (0,17fach)", "Wahrscheinlichkeit 0,807", "Dieselbe Zahl 370,4 gilt bei φ = 0", "einseitig, k = 0,5, h = 4,0965", "zweiseitig, ±3σ"):
        assert teil in t, teil


def test_exp1_phi_null_ist_info_und_nennt_die_gueltigkeit():
    kind, t = EV.verdict_exp1(0.0, 370.398, 370.398, 370.431, 0.2369, 0.2302)
    assert kind == "info" and "hält die Karte, was die Zahl verspricht" in t


def test_exp1_extreme_phi_nennen_minimum_bzw_vorzeichen():
    k, t = EV.verdict_exp1(0.98, 2834.8, 3279.2, 137.7, 0.0387, 0.5586)
    assert k == "warning" and "fällt bis φ ≈ 0,8 (55,0)" in t and "steigt danach wieder (137,7" in t
    k, t = EV.verdict_exp1(-0.3, 376.4, 370.5, 3798.2, 0.2336, 0.0252)
    assert "10,3fach" in t and "symmetrisch in φ" in t and "3 798,2" in t


def test_exp1_negative_zahl_hat_ein_minus_zeichen_im_phi():
    assert "φ = −0,3:" in EV.verdict_exp1(-0.3, 376.4, 370.5, 3798.2, 0.23, 0.03)[1]


def test_sim_caption_z_wert_und_zensur():
    s = {k: dict(mean=m, se=se, censored=c) for k, (m, se, c) in {"shewhart2": (376.6, 27.7, 0), "shewhart1": (435.9, 29.9, 1), "cusum": (67.0, 4.8, 0)}.items()}
    t = EV.sim_caption(200, 10000, s, {"shewhart2": 419.4, "shewhart1": 436.6, "cusum": 62.2})
    assert "200 Läufe" in t and "z = -1,5" in t and "zensiert 1" in t and "z = +1,0" in t


def test_exp2_klumpen_mit_teilerklaerung_nur_bei_grossem_anstieg():
    k, t = EV.verdict_exp2(0.9, 0.4522, 1.8254, 676.14, 831.78, 0.6627)
    assert k == "warning" and "0,452" in t and "1,83" in t and "676,1" in t and "831,8" in t and "0,66 des Anstiegs" in t and "nicht isoliert" in t
    k, t = EV.verdict_exp2(0.3, 0.0177, 1.018, 377.07, 376.38, 1.114)
    assert k == "info" and "klein" in t and "wird bei so kleinem Anstieg nicht angegeben" in t and "1,11" not in t       # kein Anteil über 1 melden
    k, t = EV.verdict_exp2(0.0, 0.0027, 1.003, 371.4, 370.4, float("nan"))
    assert k == "info" and "einzeln" in t


def test_toy_verdikt_bruecke_und_phi_null():
    from abw_toy import two_state
    r0, r = two_state(Fr(0)), two_state(Fr(4, 5))
    k, t = EV.verdict_toy(Fr(4, 5), r, r0)
    assert k == "warning" and "ARL 46 (bei φ = 0: 10)" in t and "2386/41 = 58,20 (bei φ = 0: 110)" in t and "Regel A wartet länger, Regel B kürzer" in t
    k, t = EV.verdict_toy(Fr(0), r0, r0)
    assert k == "info" and "1/p = 10" in t and "110" in t


@pytest.mark.parametrize("res,cal,expected_kind", [(10.68, 17.86, "info"), (292.6, 100.3, "warning")])
def test_exp3_schneller_info_langsamer_warnung(res, cal, expected_kind):
    shew = dict(residual=res, calibrated=cal, nominal=165.99, arl0_nominal=831.8)
    cus = dict(residual=res * 0.8, calibrated=cal, nominal=21.49, arl0_nominal=63.4)
    k, t = EV.verdict_exp3(0.9, 2.0, "Messversatz", shew, cus)
    assert k == expected_kind and "ARL₀ 370,4 für jedes φ (φ bekannt)" in t and "ν = 0" in t and "Bestfall" in t


def test_exp3_gemischt_nennt_beide_richtungen():
    shew = dict(residual=223.3, calibrated=88.6, nominal=153.0, arl0_nominal=831.8)
    cus = dict(residual=50.0, calibrated=50.5, nominal=14.7, arl0_nominal=63.4)
    k, t = EV.verdict_exp3(0.9, 1.0, "Messversatz", shew, cus)
    assert k == "warning" and "mit Shewhart langsamer, mit CUSUM gleich schnell" in t and "223,3 gegen 88,6" in t


def test_exp4_anker_rho_08_und_09():
    shew = dict(iid=52.59, real=1147.88, p100_iid=0.853, p100_real=0.106)
    cus = dict(iid=101.43, real=150.14, p100_iid=0.629, p100_real=0.507)
    k, t = EV.verdict_exp4(0.8, 0.963, shew, cus)
    assert k == "warning" and "1 147,9 statt 52,6" in t and "150,1 statt 101,4" in t and "Lag 1 = 0,963" in t and "addieren sich nicht" in t
    k, t = EV.verdict_exp4(0.9, 0.9904, dict(iid=54.07, real=4669.44, p100_iid=0.845, p100_real=0.047), dict(iid=104.02, real=359.89, p100_iid=0.619, p100_real=0.334))
    assert "nahe an 370,4 durch Zufall" in t and "1,45mal" in t


def test_gate_verdikt_nennt_runs_und_beide_zahlen():
    k, t = EV.verdict_gate(dict(runs="2000 + 2000", shewhart=131.2, shewhart_se=2.1, shewhart_iid=60.1, cusum2=42.5, cusum2_se=0.6, cusum2_iid=175.5))
    assert k == "info" and "131,2 ± 2,1 statt 60,1" in t and "42,5 ± 0,6 statt 175,5" in t and "nicht M/M/1" in t


def test_exp5_normal_info_andere_warnung_mit_artefakt():
    k, t = EV.verdict_exp5("Normal", 370.4, 370.4, 370.4, Fr(442), 9, False)
    assert k == "info" and "442" in t and "wie in Stück 1" in t
    k, t = EV.verdict_exp5("Exponential", 54.598, 43.91, 104.95, Fr(442), 9, True)
    assert k == "warning" and "54,60 (0,15fach von 370,4)" in t and "104,95" in t and "nach genau 9 Beobachtungen" in t and "keine Stärke der Karte" in t
    k, t = EV.verdict_exp5("t(5)", 85.29, 127.56, 275.98, Fr(14), 3, False)
    assert "nach genau" not in t and "(H = 3) 14 " in t


def test_seg_caption_zaehlt_aus_den_gegebenen_zeilen():
    t = EV.seg_caption(dict(roh0=5.0, roh3=33.8, roh6=86.5, roh9=98.2, res_min=2.2, res_max=4.8))
    assert "5,0 / 33,8 / 86,5 / 98,2 %" in t and "2,2 bis 4,8 %" in t
