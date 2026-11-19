"""Gefrorene Tabellen (abw_tables.json): Aufbau, Herkunft, Beziehungen untereinander, Neurechnung im Band (`tools/abw_make_tables.py --check`) und Abgleich mit den Vormessungen V2 und V14.

Neu gerechnet wird als Band (|z| <= 4, Standardfehler beider Seiten) bzw. mit relativer Toleranz, nie bitgleich: die Normalzahlen kommen aus der Umkehrfunktion der C-Bibliothek, Linux und Windows können in der
letzten Binärstelle abweichen (DEMO-PLAYBOOK §5). Die Vormessungswerte (Plan 05) sind Erwartungen, keine Ziele: die Simulationszellen sind mit SplitMix64 und anderen Zufalls-Seeds neu gemessen; die Abweichungen stehen in der README."""

import importlib.util
import math
from pathlib import Path

import abw_chain as CH
import abw_constants as C
import abw_queue as Q
import abw_reference as REF

ROOT = Path(__file__).resolve().parent.parent
T_ = REF.tables()
H = CH.cusum_h_for_arl(0.5)


def _tool():
    spec = importlib.util.spec_from_file_location("abw_make_tables", ROOT / "tools" / "abw_make_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kopf_nennt_quelle_stand_zufall_und_gitter():
    m = REF.meta()
    assert m["stand"] == "2026-11-19" and "SplitMix64" in m["zufall"] and "Simulation" in m["hinweis"] and "m = 16 und 32" in m["grid_coarse"] and "n = 192" in m["grid_fine"]
    assert abs(m["h_one"] - H) < 1e-9


def test_tabellen_enthalten_genau_die_gemessenen_zellen():
    assert set(T_["ar1"]) == {REF.pk(p) for p in C.PHI_OPTIONS}
    assert set(T_["zweiseitig"]) == {REF.pk(p) for p in C.PHI_FOCUS}
    assert set(T_["nacheichung"]) == {REF.pk(p) for p in C.PHI3_OPTIONS}
    assert set(T_["residuen_cusum"]) == {f"{a}|{REF.pk(p)}|{d:.1f}" for a in ("offset", "process") for p in C.PHI3_OPTIONS for d in C.DELTA_OPTIONS}
    assert set(T_["mm1"]) == {f"{r:.1f}" for r in C.RHO_OPTIONS} and set(T_["gate"]) == {"seed1", "seed2"}
    assert set(T_["schiefe"]) == set(C.DIST_KEYS) - {"normal"}
    assert set(T_["segmentierung"]) == {f"{REF.pk(p)}|{k}" for p in (0.0, 0.3, 0.6, 0.9) for k in ("roh", "resid")}


def test_simulationszellen_tragen_seed_und_r_und_alle_seeds_sind_verschieden():
    seeds = []
    for k, c in T_["zweiseitig"].items():
        assert c["runs"] == 20000 and isinstance(c["seed"], int) and c["censored"] == 0
        seeds.append(c["seed"])
    for k, c in T_["nacheichung"].items():
        assert c["sim_runs"] == 100000 and isinstance(c["sim_seed"], int)
        seeds.append(c["sim_seed"])
    for k, c in T_["mm1"].items():
        assert c["sim_shewhart_runs"] > 0 and c["sim_cusum_runs"] == 5000
        seeds.append(c["seed"])
    for k, c in T_["gate"].items():
        assert c["runs"] == 2000 and c["lag_runs"] == 400 and c["iid_runs"] == 5000
        seeds.append(c["seed"])
    for k, c in T_["segmentierung"].items():
        assert c["runs"] == 400
        seeds.append(c["seed"])
    assert len(seeds) == len(set(seeds)) and all(51000 <= s < 59000 for s in seeds)


def test_ar1_zeilen_sind_innerlich_stimmig():
    for k, r in T_["ar1"].items():
        assert r["shewhart2"] >= 1 and 0 <= r["p100"] <= 1 and 0 <= r["cusum_p100"] <= 1
        if r["median"] is not None:
            assert r["q05"] <= r["median"] <= r["q95"]
        assert 0 < r["q"] < 1 and r["cluster_length"] >= 1
    assert abs(T_["ar1"]["0.00"]["shewhart2"] - CH.NOM) < 1e-6 and abs(T_["ar1"]["0.00"]["shewhart1"] - CH.NOM) < 1e-6
    assert abs(T_["ar1"]["0.00"]["cusum"] - 370.43) < 0.01
    # zweiseitig symmetrisch in φ: die Zeilen −0,6 und 0,6 sowie −0,3 und 0,3 stimmen überein
    for p in ("0.60", "0.30"):
        assert abs(T_["ar1"]["-" + p]["shewhart2"] - T_["ar1"][p]["shewhart2"]) < 1e-9
    assert T_["ar1"]["-0.30"]["shewhart1"] < T_["ar1"]["0.30"]["shewhart1"]                  # einseitig nicht
    assert T_["ar1"]["0.95"]["cusum"] == T_["ar1"]["0.95"]["cusum_fine"] and T_["ar1"]["0.60"]["cusum"] == T_["ar1"]["0.60"]["cusum_coarse"]
    for p in ("0.80", "0.90", "0.95", "0.98"):
        assert abs(T_["ar1"][p]["cusum_fine"] / T_["ar1"][p]["cusum_coarse"] - 1) < 0.01      # feines gegen grobes Gitter: unter 1 %


def test_shewhart_zeilen_gegen_live_rechnung():
    for p in (0.0, 0.6, 0.9, -0.3):
        assert abs(CH.shewhart_arl_two_sided(p) - REF.ar1(p)["shewhart2"]) < 1e-6
        assert abs(CH.shewhart_arl_one_sided(p) - REF.ar1(p)["shewhart1"]) < 1e-6


def test_tool_check_rechnet_proben_neu_und_findet_keine_abweichung():
    assert _tool().check() == []


def test_tool_check_meldet_abweichungen(tmp_path):
    import json
    tab = json.loads((ROOT / "abw_tables.json").read_text(encoding="utf-8"))
    tab["ar1"]["0.60"]["shewhart2"] *= 1.2
    p = tmp_path / "x.json"
    p.write_text(json.dumps(tab), encoding="utf-8")
    bad = _tool().check(p)
    assert any("shewhart2 φ=0.6" in b for b in bad)


# ----------------------------------------------------------------------------------------------------------------------------------- Vormessung V14
V14_CUSUM = {-0.3: 3798.216, 0.0: 370.431, 0.05: 288.002, 0.1: 229.798, 0.2: 156.248, 0.3: 114.225, 0.4: 88.630, 0.5: 72.428, 0.6: 62.171, 0.7: 56.328, 0.8: 55.043, 0.9: 63.399}


def test_cusum_tafel_gleich_v14_auf_0_1_prozent_und_fein_gitter_werte_ab_095():
    for p, soll in V14_CUSUM.items():
        assert abs(REF.ar1(p)["cusum"] - soll) <= 1e-3 * soll, p
    assert abs(REF.ar1(0.95)["cusum"] - 83.8) < 0.1 and abs(REF.ar1(0.98)["cusum"] - 137.7) < 0.1
    assert abs(REF.ar1(-0.6)["cusum"] - 284222) < 5 and abs(REF.ar1(-0.3)["cusum"] / CH.NOM - 10.254) < 0.01


def test_cusum_verteilung_tafel_phi_0_ist_die_eindimensionale_kette():
    r = REF.ar1(0.0)
    assert abs(r["cusum_p100"] - 0.2302) < 3e-4 and r["cusum_median"] == 257
    assert abs(REF.ar1(0.6)["cusum_p100"] - 0.8074) < 2e-3 and abs(REF.ar1(0.3)["cusum_p100"] - 0.5837) < 2e-3


def test_klumpen_tafel_v14():
    for p, (q, inv) in {0.3: (0.0177, 377.1), 0.6: (0.1035, 413.1), 0.9: (0.4522, 676.1)}.items():
        assert abs(REF.ar1(p)["q"] - q) < 5e-5 and abs(REF.ar1(p)["start_rate_inv"] - inv) < 0.06


def test_zweiseitig_naeherung_gegen_simulation_v14_und_neu_gemessen():
    erwartet = {0.0: (737.2, 368.6), 0.3: (170.9, 85.4), 0.6: (78.6, 39.3), 0.9: (70.7, 35.4)}
    for p, (up, ap) in erwartet.items():
        c = REF.twosided(p)
        assert abs(c["one_sided"] - up) < 0.06 and abs(c["approx"] - ap) < 0.06
    v14_sim = {0.0: (368.3, 2.6), 0.3: (84.1, 0.6), 0.6: (37.6, 0.2), 0.9: (27.2, 0.2)}
    for p, (m, se) in v14_sim.items():
        c = REF.twosided(p)
        assert abs(c["sim"] - m) <= 4 * math.hypot(se, c["se"]), p                           # neu gemessen gegen V14 im Band
    abw = {p: (REF.twosided(p)["approx"] - REF.twosided(p)["sim"]) / REF.twosided(p)["sim"] for p in (0.3, 0.6, 0.9)}
    assert abs(abw[0.3]) < 0.02 and 0.03 < abw[0.6] < 0.07 and 0.25 < abw[0.9] < 0.35          # 1,6 %, 4,6 %, 30 %: Näherung gilt nur bis φ = 0,3


def test_residuen_cusum_zellen_gegen_v14():
    v14 = {("offset", 0.6, 1.0): (9.84, 21.28), ("offset", 0.9, 1.0): (14.73, 50.69), ("offset", 0.9, 2.0): (4.93, 18.53),
           ("process", 0.6, 1.0): (11.23, 22.71), ("process", 0.9, 1.0): (21.49, 58.81), ("process", 0.9, 2.0): (11.99, 27.81),
           ("offset", 0.3, 0.5): (22.67, 39.23), ("process", 0.3, 2.0): (4.07, 5.43)}
    for (art, p, d), (nom, cal) in v14.items():
        c = REF.resid_cell(art, p, d)
        assert abs(c["nominal"] - nom) <= 2e-3 * nom, (art, p, d)
        assert abs(c["calibrated"] - cal) <= 3e-3 * cal if p < 0.9 else abs(c["calibrated"] - cal) <= 0.01 * cal, (art, p, d)      # φ = 0,9: neue Schwelle auf feinem Gitter (Abweichung ≤ 1 %)
    for k, c in T_["residuen_cusum"].items():
        assert c["calibrated"] > c["nominal"] > 1                                          # die nachgeeichte Karte (ARL₀ 370 statt 63 bis 370) meldet später als die nominelle


def test_nominelle_rohkarte_meldet_oft_vor_der_aenderung_arl0_unter_370():
    for p in (0.3, 0.6, 0.9):
        assert REF.ar1(p)["cusum"] < CH.NOM and REF.ar1(p)["shewhart2"] > CH.NOM            # Shewhart: ARL₀ ÜBER 370, die nominelle Karte ist langsam; CUSUM: ARL₀ weit darunter


def test_mm1_tafel_v14_und_simulation_im_band():
    erwartet = {0.5: (70.22, 87.31, 0.760, 0.685, 48, 61), 0.8: (150.14, 101.43, 0.507, 0.628, 98, 71), 0.9: (359.89, 104.02, 0.334, 0.619, 222, 73)}
    for rho, (real, iid, p_re, p_iid, med_re, med_iid) in erwartet.items():
        c = T_["mm1"][f"{rho:.1f}"]
        assert abs(c["cusum_real"] - real) <= 3e-3 * real and abs(c["cusum_iid"] - iid) <= 2e-3 * iid
        assert abs(c["p100_real"] - p_re) < 2e-3 and abs(c["p100_iid"] - p_iid) < 2e-3 and (c["median_real"], c["median_iid"]) == (med_re, med_iid)
        assert abs(c["sim_cusum"] - c["cusum_real"]) <= 4 * c["sim_cusum_se"]
        assert abs(c["sim_shewhart"] - Q.shewhart_closed(rho)["arl"]) <= 4 * c["sim_shewhart_se"]


def test_gate_tafel_neu_gemessen_gegen_v2_und_plan_08_im_band():
    for seed in ("seed1", "seed2"):
        g = T_["gate"][seed]
        assert abs(g["shewhart"]["mean"] - 128.6) <= 4 * math.hypot(g["shewhart"]["se"], 2.4)            # V2 6.1
        assert abs(g["cusum2"]["mean"] - 42.28) <= 4 * math.hypot(g["cusum2"]["se"], 0.38)             # Plan 08 / V17
        assert abs(g["iid_shewhart_sim"]["mean"] - Q.gate_shewhart_iid_arl(Q.gate_exact(C.GATE_LAMBDA))) <= 4 * g["iid_shewhart_sim"]["se"]
        for got, soll in zip(g["lag"], (0.781, 0.586, 0.412, 0.255, 0.178)):
            assert abs(got[0] - soll) < 0.003
        assert abs(g["iid_cusum2"]["mean"] - 177.7) <= 4 * math.hypot(g["iid_cusum2"]["se"], 3.0)
    a, b = T_["gate"]["seed1"], T_["gate"]["seed2"]
    assert abs(a["shewhart"]["mean"] - b["shewhart"]["mean"]) <= 4 * math.hypot(a["shewhart"]["se"], b["shewhart"]["se"])      # die zwei Zufalls-Seeds stimmen im Band überein
    assert abs(a["cusum2"]["mean"] - b["cusum2"]["mean"]) <= 4 * math.hypot(a["cusum2"]["se"], b["cusum2"]["se"])


def test_nacheichung_schiefe_tafel_ist_selbstkonsistent():
    from abw_skew import DISTS, shewhart_two_sided
    for k, rec in T_["schiefe"].items():
        assert abs(shewhart_two_sided(DISTS[k], rec["shewhart_limit"]) - CH.NOM) < 1e-4 and rec["cusum_h"] > 4.0
    assert T_["schiefe"]["expon"]["shewhart_limit"] > T_["schiefe"]["t5"]["shewhart_limit"] > 3.0


def test_segmentierung_gegen_v14_im_band():
    v14 = {(0.0, "roh"): 3.0, (0.3, "roh"): 36.5, (0.6, "roh"): 87.0, (0.9, "roh"): 99.5, (0.0, "resid"): 3.0, (0.3, "resid"): 3.8, (0.6, "resid"): 3.2, (0.9, "resid"): 4.5}
    for (p, kind), soll in v14.items():
        c = T_["segmentierung"][f"{REF.pk(p)}|{kind}"]
        se = math.hypot(math.sqrt(max(soll * (100 - soll), 50.0) / 400), math.sqrt(max(c["share"][0] * (100 - c["share"][0]), 50.0) / 400))      # Prozentpunkte, beide Seiten 400 Reihen
        assert abs(c["share"][0] - soll) <= 4 * se, (p, kind)
    for p in (0.0, 0.3, 0.6, 0.9):
        assert all(a >= b for a, b in zip(T_["segmentierung"][f"{REF.pk(p)}|roh"]["share"], T_["segmentierung"][f"{REF.pk(p)}|roh"]["share"][1:]))   # strengere Strafe, weniger Brüche
