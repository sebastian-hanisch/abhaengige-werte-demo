"""Claims: jede Zahl der README wird hier aus der Rechnung bzw. der gefrorenen Tabelle neu erzeugt und muss wörtlich in der README stehen.

`readme_facts.py` erzeugt die Zeilen der Tabelle „Befunde“ und der Tabelle „Befunde und Korrekturen“; dieser Test prüft, dass sie vollständig in der README stehen, dass jeder genannte Test existiert, dass die Zählaussagen
(Gelesen als Zählung) stimmen und dass der Text „Wo die Annahmen enden“ in App und README wortgleich ist. Zufallsabhängige Zahlen stehen in `abw_tables.json` und werden in `tests/test_tables.py` im Band neu gerechnet;
exakte Zahlen werden live gerechnet. Zahlen der Vormessung (Plan) stehen als Erwartung in der README; sie sind unten als Literale festgehalten."""

import math
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import readme_facts as RF

import abw_chain as CH
import abw_constants as C
import abw_reference as REF
import abw_scenario as SC
import abw_texts as TX
from abw_evaluation import de

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
H = CH.cusum_h_for_arl(0.5)


def _im_readme(text):
    assert text in README, f"nicht in der README: {text!r}"


# ----------------------------------------------------------------------------------------------------------------------------------- Tabellen
def test_jede_zeile_der_befunde_tabelle_steht_in_der_readme():
    for frage, befund, test in RF.rows():
        _im_readme(f"| {frage} | {befund} | {test} |")


def test_jede_zeile_der_korrekturen_tabelle_steht_in_der_readme():
    for gr, plan, hier, ein in RF.corrections():
        _im_readme(f"| {gr} | {plan} | {hier} | {ein} |")


def test_jeder_genannte_test_existiert():
    namen = set(re.findall(r"`(test_[a-z0-9_]+\.py)(?:::([a-z0-9_]+))?`", README))
    assert namen
    for datei, funktion in namen:
        pfad = ROOT / "tests" / datei
        assert pfad.exists(), datei
        if funktion:
            assert f"def {funktion}(" in pfad.read_text(encoding="utf-8"), (datei, funktion)


def test_annahmen_text_ist_in_app_und_readme_wortgleich():
    _im_readme(TX.ASSUMPTIONS_MARKDOWN.strip())
    assert "🚧 Wo die Annahmen enden" in (ROOT / "app.py").read_text(encoding="utf-8") and "TX.ASSUMPTIONS_MARKDOWN" in (ROOT / "app.py").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------------------------- einzelne Zahlen
def test_shewhart_zahlen():
    soll = {0.0: 370.398, 0.3: 376.383, 0.6: 419.377, 0.9: 831.783}
    for phi, s in soll.items():
        assert abs(CH.shewhart_arl_two_sided(phi) - s) < 5e-4
    assert abs(CH.shewhart_arl_two_sided(-0.9) - 831.783) < 5e-4 and abs(CH.shewhart_arl_two_sided(-0.6) - 419.377) < 5e-4
    one = {0.0: 370.398, 0.3: 380.701, 0.6: 436.597, 0.9: 924.029, -0.3: 370.458, -0.6: 383.158, -0.9: 603.037}
    for phi, s in one.items():
        assert abs(CH.shewhart_arl_one_sided(phi) - s) < 5e-4, phi
    d = {p: CH.shewhart_distribution(p) for p in (0.0, 0.3, 0.6, 0.9)}
    assert [x["median"] for x in d.values()] == [257, 261, 291, 576] and [x["q05"] for x in d.values()] == [19, 20, 22, 41] and [x["q95"] for x in d.values()] == [1109, 1127, 1256, 2494]
    assert [round(x["p100"], 4) for x in d.values()] == [0.2369, 0.2336, 0.2126, 0.1151]


def test_klumpen():
    for phi, q, inv in ((0.0, 0.0027, 371.4), (0.3, 0.0177, 377.1), (0.6, 0.1035, 413.1), (0.9, 0.4522, 676.1)):
        c = CH.clusters(phi)
        assert round(c["q"], 4) == q and round(c["start_rate_inv"], 1) == inv
    c = CH.clusters(0.9)
    assert round(c["cluster_length"], 2) == 1.83 and round(c["share"], 2) == 0.66


def test_cusum_zahlen():
    from abw_cusum import ar1_arl
    soll = {-0.3: 3798.2, 0.0: 370.4, 0.1: 229.8, 0.3: 114.2, 0.6: 62.2, 0.8: 55.0, 0.9: 63.4}
    for phi, s in soll.items():
        assert abs(ar1_arl(phi, H) - s) <= 0.0015 * s + 0.06, phi
    # φ = 0,95 und 0,98: feines Gitter (n = 224, m = 16 / 24, Richardson mit Exponent 2)
    for phi, s in ((0.95, 83.8), (0.98, 137.7)):
        a16 = ar1_arl(phi, H, ms=(16,), n=224, lr=4.5, richardson=False)
        a24 = ar1_arl(phi, H, ms=(24,), n=224, lr=4.5, richardson=False)
        assert abs((2.25 * a24 - a16) / 1.25 - s) < 0.06
    assert round(1 - REF.ar1(0.1)["cusum"] / CH.NOM, 2) == 0.38                  # „schon φ = 0,1 senkt sie um 38 %“
    assert round(REF.ar1(-0.3)["cusum"] / CH.NOM, 1) == 10.3


def test_residuenkarte_zellen():
    """Zählaussagen der README über die 18 Zellen je Karte: Residuenkarte gegen nachgeeichte Rohkarte."""
    sh_slow = cu_slow = 0
    faster_cu, small = [], []
    for art in ("offset", "process"):
        for phi in C.PHI3_OPTIONS:
            cal = REF.calibration(phi)
            for d in C.DELTA_OPTIONS:
                kw = dict(zshift=d) if art == "offset" else dict(drift=(1 - phi) * d, stat_mean=(1 - phi) * d)
                sc = CH.shewhart_arl(phi, -cal["limit"], cal["limit"], 480, **kw)
                sr = CH.residual_shewhart_arl(phi, d, art)
                m1, m2 = CH.residual_innovation_means(phi, d, art)
                cr = CH.residual_cusum_arl(H, 0.5, m1, m2)
                cc = REF.resid_cell(art, phi, d)["calibrated"]
                sh_slow += sr > sc
                cu_slow += cr > cc
                if cr < cc:
                    faster_cu.append((art, phi, d))
                if d <= 1.0:
                    small.append((sr / sc, cr / cc))
    assert (sh_slow, cu_slow) == (17, 13)
    assert sorted(faster_cu) == [("offset", 0.3, 2.0), ("offset", 0.6, 2.0), ("offset", 0.9, 2.0), ("process", 0.3, 2.0), ("process", 0.6, 2.0)]
    assert len(small) == 12 and all(a > 1 and b > 1 for a, b in small)
    assert (round(min(a for a, _ in small), 1), round(max(a for a, _ in small), 1)) == (1.4, 2.9) and (round(min(b for _, b in small), 1), round(max(b for _, b in small), 1)) == (1.1, 1.6)


V14_CELLS = {("offset", 0.3, 0.5): (22.67, 39.23), ("offset", 0.3, 1.0): (8.93, 12.91), ("offset", 0.3, 2.0): (3.57, 4.92), ("offset", 0.6, 0.5): (21.03, 58.93),
             ("offset", 0.6, 1.0): (9.84, 21.28), ("offset", 0.6, 2.0): (3.88, 7.90), ("process", 0.3, 0.5): (22.96, 39.52), ("process", 0.3, 1.0): (9.36, 13.34),
             ("process", 0.3, 2.0): (4.07, 5.43), ("process", 0.6, 0.5): (21.99, 59.91), ("process", 0.6, 1.0): (11.23, 22.71), ("process", 0.6, 2.0): (5.44, 9.59)}
V14_PHI09 = {("offset", 0.5): 114.29, ("offset", 1.0): 50.69, ("offset", 2.0): 18.53, ("process", 0.5): 119.91, ("process", 1.0): 58.81, ("process", 2.0): 27.81}


def test_korrekturen_gegen_die_plan_zahlen():
    for (art, phi, d), (nom, cal) in V14_CELLS.items():
        c = REF.resid_cell(art, phi, d)
        assert abs(c["nominal"] - nom) <= max(0.001 * nom, 0.011) and abs(c["calibrated"] - cal) <= max(0.001 * cal, 0.011), (art, phi, d)      # „stimmen auf 0,1 % überein“
    rel = [abs(REF.resid_cell(a, 0.9, d)["calibrated"] / v - 1) for (a, d), v in V14_PHI09.items()]
    assert max(rel) < 0.007 and all(REF.resid_cell(a, 0.9, d)["calibrated"] < v for (a, d), v in V14_PHI09.items())                              # „um höchstens 0,7 % kleiner“
    for phi in (0.3, 0.6):
        c = REF.calibration(phi)
        assert abs(c["arl0_at_h_fine_coarse_grid"] / CH.NOM - 1) < 0.001 and abs(c["arl0_at_h_coarse_fine_grid"] / CH.NOM - 1) < 0.001              # „innerhalb von 0,1 %“
    c9 = REF.calibration(0.9)
    assert round(100 * (c9["arl0_at_h_coarse_fine_grid"] / CH.NOM - 1), 1) == 1.3 and abs(c9["h_coarse"] - 22.277) < 0.001
    assert abs(CH.calibrated_limit(0.9) - 2.7113) < 2e-4                                                                                          # Plan 2,7113: Abweichung höchstens 1·10⁻⁴ bis 2·10⁻⁴


def test_mm1_zusatz():
    c = REF.mm1(0.9)
    assert round(c["p100_real"] / 0.2302, 2) == 1.45
    iid = [REF.mm1(r)["cusum_iid"] for r in C.RHO_OPTIONS]
    assert round(min(iid)) == 87 and round(max(iid)) == 104
    assert round(REF.mm1(0.8)["cusum_real"], 1) == 150.1 and round(REF.mm1(0.9)["cusum_real"], 1) == 359.9 and round(REF.mm1(0.5)["cusum_real"], 1) == 70.2


def test_gate():
    from abw_queue import gate_exact, gate_shewhart_iid_arl
    ex = gate_exact(C.GATE_LAMBDA)
    assert (round(ex["loss"], 4), round(ex["mean"], 3), round(ex["sd"], 3), round(ex["skew"], 3)) == (0.0760, 0.659, 0.898, 1.738) and round(gate_shewhart_iid_arl(ex), 1) == 60.1
    gp = RF.gate_pooled()
    assert 128 <= gp["shewhart"][0] <= 140 and 41 <= gp["cusum2"][0] <= 44.5 and 170 <= gp["iid_cusum2"][0] <= 182
    assert all(abs(a - b) < 0.0025 for a, b in zip(gp["lag"], (0.781, 0.586, 0.412, 0.255, 0.178)))
    from abw_queue import mm1_moments
    assert round(1147.88 / 52.59, 1) == 21.8 and mm1_moments(0.8)[0] == pytest.approx(4.0)
    assert round(CH.shewhart_arl_two_sided(0.781), 1) == 530.9                                              # AR(1) mit dem Lag 1 des Gates (App-Text)


def test_schiefe():
    from abw_skew import DISTS, cusum_arl, shewhart_two_sided, sign_cusum_arl
    assert [round(shewhart_two_sided(DISTS[k]), 2) for k in ("expon", "gamma4", "lognorm", "t5")] == [54.60, 96.75, 65.01, 85.29]
    assert [round(cusum_arl(DISTS[k]), 2) for k in ("expon", "gamma4", "t5")] == [104.95, 159.01, 275.98]
    assert abs(cusum_arl(DISTS["lognorm"]) - 120.08) < 0.01               # Plan: 120,08 (Kette n = 400 bis 1 600); hier n = 400 / 800 mit Richardson: 120,09
    assert [int(sign_cusum_arl(h)) for h in (3, 8, 9)] == [14, 266, 442]
    assert round(math.exp(4) / 1, 2) == 54.60 and round(370.398 / math.exp(4), 2) == 6.78 and round(DISTS["expon"].mean, 6) == 1.0


def test_segmentierung():
    for p in (0.0, 0.3, 0.6, 0.9):
        assert REF.segmentation(p, False)["runs"] == 400
    assert max(REF.segmentation(p, True)["share"][0] for p in (0.0, 0.3, 0.6, 0.9)) <= 8.0           # „Residuen ... klein“
    assert REF.segmentation(0.9, False)["share"][0] > 95 and REF.segmentation(0.0, False)["share"][0] < 8


def test_zeichen_der_intro_und_der_warum_zahlen():
    _im_readme("bei φ = 0,6 meldet die Shewhart-Karte (zweiseitig) im Mittel erst nach 419,4 Beobachtungen, der einseitige CUSUM schon nach 62,2")
    assert round(CH.shewhart_arl_two_sided(0.6), 1) == 419.4 and round(REF.ar1(0.6)["cusum"], 1) == 62.2
    assert de(REF.ar1(0.6)["shewhart2"], 1) in TX.intro() and de(REF.ar1(0.6)["cusum"], 1) in TX.intro()


def test_lag_1_der_warum_zeile():
    from abw_queue import lag1_exact
    assert (round(lag1_exact(0.5), 2), round(lag1_exact(0.9), 2)) == (0.78, 0.99)
    _im_readme("Lag 1 = 0,78 bis 0,99 bei Auslastung 0,5 bis 0,9 (M/M/1, exakt)")


def test_nominalwerte_in_den_verdikten():
    assert (de(REF.ar1(0.0)["p100"], 3), de(REF.ar1(0.0)["cusum_p100"], 3), REF.ar1(0.0)["median"], REF.ar1(0.0)["cusum_median"]) == ("0,237", "0,230", 257, 257)
    _im_readme("P(T ≤ 100) = 0,237 (Shewhart) und 0,230 (CUSUM); Median 257 (Shewhart und CUSUM)")


def test_ziel_und_schwellen():
    from abw_cusum import ar1_arl
    assert abs(CH.NOM - 370.398) < 5e-4 and abs(H - 4.0965) < 5e-5 and abs(CH.one_sided_limit() - 2.7822) < 5e-5
    assert round(ar1_arl(0.0, C.H_TWO) / 2.0, 1) == 368.6                      # zweiseitiger CUSUM mit h = 4,77: einseitig exakt 737,2, halbiert 368,6 (Näherung; Simulation 368,3 ± 2,6)
    assert abs(REF.twosided(0.0)["sim"] - 368.6) <= 4 * REF.twosided(0.0)["se"]
    _im_readme(f"Simulation {de(REF.twosided(0.0)['sim'], 1)} ± {de(REF.twosided(0.0)['se'], 1)}")


# ----------------------------------------------------------------------------------------------------------------------------------- Wortlisten
VERBOTEN = ("TODO", "@@", "geplant", "Saatwert", "Schwanz", "trägt", "tragen", "Kipppunkt")


def test_keine_platzhalter_und_verbotenen_woerter():
    for quelle in (README, (ROOT / "app.py").read_text(encoding="utf-8")) + tuple((ROOT / n).read_text(encoding="utf-8") for n in ("abw_texts.py", "abw_evaluation.py", "abw_constants.py")):
        for wort in VERBOTEN:
            assert wort not in quelle, wort


def test_readme_hat_die_pflichtabschnitte_in_der_reihenfolge():
    kopf = [m.group(0) for m in re.finditer(r"^## .+$", README, flags=re.M)]
    erwartet = ["## Warum dieses Problem", "## Befunde und Korrekturen gegenüber dem Plan", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "## Ehrliche Grenzen und 🚧 Nicht gemessen",
                "## Tests", "## Dateistruktur", "## Bewusst nicht umgesetzt", "## Lokal ausführen", "## Tests ausführen"]
    assert kopf == erwartet


# ----------------------------------------------------------------------------------------------------------------------------------- Tests und Laufzeit
def test_testzahl_der_readme_stimmt_mit_der_suite_ueberein():
    m = re.search(r"\*\*(\d+) Tests\*\*", README)
    assert m, "die README nennt keine Testzahl"
    out = subprocess.run([sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    n = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert int(m.group(1)) == n, f"README nennt {m.group(1)}, die Suite hat {n}"


def test_worst_case_laufzeit_der_live_rechnungen():
    """Obergrenzen (kein Befund; die gemessenen Zeiten stehen in der README): Shewhart-Kette, Simulation am Limit R · tmax = 5 Mio., CUSUM-Kette."""
    t = time.perf_counter()
    CH.shewhart_arl_two_sided(0.98)
    assert time.perf_counter() - t < 20
    t = time.perf_counter()
    SC.ar1_runs(0.98, 500, 1, 10_000)
    assert time.perf_counter() - t < 60
    from abw_cusum import ar1_arl
    t = time.perf_counter()
    ar1_arl(0.9, H)
    assert time.perf_counter() - t < 30
