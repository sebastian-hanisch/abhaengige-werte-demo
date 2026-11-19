"""Die Zahlen der README als Texte: jede Zeile der Tabelle „Befunde“ und der Abschnitt „Befunde und Korrekturen“ werden hier aus der Rechnung bzw. der gefrorenen Tabelle erzeugt.

`tests/test_claims.py` prüft, dass jeder dieser Texte wörtlich in der README steht (so kann keine Zahl in der README stehen, die nicht aus der Rechnung kommt, und keine Rechnung ohne Zeile).
Hilfsmodul, kein Test. Formatierung wie in der App (`abw_evaluation.de`: Dezimalkomma, Tausenderleerzeichen)."""

from __future__ import annotations

import math
from fractions import Fraction as Fr

import abw_chain as CH
import abw_constants as C
import abw_queue as Q
import abw_reference as REF
import abw_skew as SK
import abw_toy as T
from abw_evaluation import de, frac

TB = REF.tables()
H = CH.cusum_h_for_arl(0.5)


def sl(values, f=lambda v: de(v, 1), sep=" / "):
    return sep.join(f(v) for v in values)


def ar1(p):
    return REF.ar1(p)


def gate_pooled():
    g1, g2 = REF.gate()["seed1"], REF.gate()["seed2"]
    out = {}
    for key in ("shewhart", "cusum2"):
        out[key] = ((g1[key]["mean"] + g2[key]["mean"]) / 2, math.hypot(g1[key]["se"], g2[key]["se"]) / 2)
    out["iid_cusum2"] = ((g1["iid_cusum2"]["mean"] + g2["iid_cusum2"]["mean"]) / 2, math.hypot(g1["iid_cusum2"]["se"], g2["iid_cusum2"]["se"]) / 2)
    out["lag"] = [(a[0] + b[0]) / 2 for a, b in zip(g1["lag"], g2["lag"])]
    return out


def rows() -> list[tuple[str, str, str]]:
    """(Frage, Befund, Test) je Zeile der Tabelle „Befunde“."""
    r = []
    p4 = (0.0, 0.3, 0.6, 0.9)
    r.append(("Shewhart zweiseitig auf AR(1), ARL₀ bei φ = 0 / 0,3 / 0,6 / 0,9 (Beobachtungen, exakt, zwei Rechenwege)", sl([ar1(p)["shewhart2"] for p in p4], lambda v: de(v, 3)),
              "`test_oracle_shewhart.py::test_zellenkette_gleich_integralgleichung_auf_5e_4`, `test_claims.py::test_shewhart_zahlen`"))
    r.append(("Symmetrie: φ = −0,6 und −0,9 (zweiseitig); einseitig (Alarm bei x > 2,7822) bei φ = 0 / 0,3 / 0,6 / 0,9 und bei −0,3 / −0,6 / −0,9",
              f"{de(ar1(-0.6)['shewhart2'], 3)} und {de(CH.shewhart_arl_two_sided(-0.9), 3)}; einseitig {sl([ar1(p)['shewhart1'] for p in p4], lambda v: de(v, 3))} und "
              f"{sl([CH.shewhart_arl_one_sided(p) for p in (-0.3, -0.6, -0.9)], lambda v: de(v, 3))}",
              "`test_oracle_shewhart.py::test_symmetrie_in_phi_gleichheit_auf_1e_9_nur_zweiseitig`, `test_claims.py::test_shewhart_zahlen`"))
    r.append(("Lauflängenverteilung der zweiseitigen Shewhart-Karte bei φ = 0 / 0,3 / 0,6 / 0,9: P(T ≤ 100); Median; 5-%-Quantil; 95-%-Quantil (exakt, Kette n = 960)",
              f"{sl([ar1(p)['p100'] for p in p4], lambda v: de(v, 4))}; {sl([ar1(p)['median'] for p in p4], lambda v: str(v))}; {sl([ar1(p)['q05'] for p in p4], lambda v: str(v))}; "
              f"{sl([ar1(p)['q95'] for p in p4], lambda v: de(v, 0))}",
              "`test_oracle_shewhart.py::test_verteilung_tafelwerte_phi_09`, `test_claims.py::test_shewhart_zahlen`"))
    r.append(("Alarm-Klumpen bei φ = 0 / 0,3 / 0,6 / 0,9: q = P(zweiter Alarm in Folge ǀ Alarm); 1/(p(1 − q)) (Kehrwert der Rate der Klumpenanfänge); bei φ = 0,9: mittlere Klumpenlänge und Anteil des Anstiegs",
              f"q {sl([ar1(p)['q'] for p in p4], lambda v: de(v, 4))}; {sl([ar1(p)['start_rate_inv'] for p in p4])}; bei φ = 0,9: {de(ar1(0.9)['cluster_length'], 2)} und {de(ar1(0.9)['share'], 2)}",
              "`test_oracle_shewhart.py::test_alarmpaar_quadratur_gleich_einschluss_ausschluss_der_kette`, `test_claims.py::test_klumpen`"))
    r.append(("CUSUM einseitig (k = 0,5, h = 4,0965) auf AR(1), ARL₀ bei φ = −0,3 / 0 / 0,1 / 0,3 / 0,6 / 0,8 / 0,9 / 0,95 / 0,98 (zweidimensionale Kette); φ = −0,6",
              f"{sl([ar1(p)['cusum'] for p in (-0.3, 0.0, 0.1, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98)], lambda v: de(v, 0 if v >= 1000 else 1))}; φ = −0,6: {de(ar1(-0.6)['cusum'], 0)} (nur Kette)",
              "`test_oracle_cusum.py`, `test_claims.py::test_cusum_zahlen`"))
    r.append(("CUSUM, Verhältnis zu 370,4 bei φ = 0,1 / 0,3 / 0,6 / 0,8 / 0,9 / 0,98 und −0,3; P(T ≤ 100) bei φ = 0 / 0,3 / 0,6 / 0,9",
              f"{sl([ar1(p)['cusum'] / CH.NOM for p in (0.1, 0.3, 0.6, 0.8, 0.9, 0.98)], lambda v: de(v, 2))}; −0,3: {de(ar1(-0.3)['cusum'] / CH.NOM, 1)}; "
              f"P(T ≤ 100) {sl([ar1(p)['cusum_p100'] for p in p4], lambda v: de(v, 3))}",
              "`test_claims.py::test_cusum_zahlen`, `test_oracle_cusum.py::test_verteilung_der_kette_gegen_simulation_p100`"))
    r.append(("Der CUSUM-Wert ist nicht monoton in φ", f"fällt bis φ = 0,8 ({de(ar1(0.8)['cusum'], 1)}), danach steigend ({de(ar1(0.9)['cusum'], 1)} / {de(ar1(0.95)['cusum'], 1)} / {de(ar1(0.98)['cusum'], 1)}); "
              f"schon φ = 0,1 senkt ihn um {de(100 * (1 - ar1(0.1)['cusum'] / CH.NOM), 0)} %",
              "`test_oracle_cusum.py::test_nicht_monoton_minimum_bei_0_8_im_gitter_und_anstieg_danach`, `test_claims.py::test_cusum_zahlen`"))
    t2 = [REF.twosided(p) for p in p4]
    r.append(("Zweiseitiger CUSUM, h = 4,77 (ARL₀ 368,6 bei φ = 0): einseitig exakt / Näherung „einseitig/2“ / Simulation (20 000 Läufe) bei φ = 0 / 0,3 / 0,6 / 0,9",
              f"{sl([c['one_sided'] for c in t2])} / {sl([c['approx'] for c in t2])} / " + " / ".join(f"{de(c['sim'], 1)} ± {de(c['se'], 1)}" for c in t2)
              + "; Abweichung der Näherung " + sl([100 * (c['approx'] - c['sim']) / c['sim'] for c in t2], lambda v: de_s(v)) + " %",
              "`test_tables.py::test_zweiseitig_naeherung_gegen_simulation_v14_und_neu_gemessen`, `test_oracle_cusum.py::test_zweiseitiger_cusum_ist_simulation_naeherung_gilt_nur_bis_phi_03`"))
    na = [REF.calibration(p) for p in (0.3, 0.6, 0.9)]
    r.append(("Nacheichung bei φ = 0,3 / 0,6 / 0,9: Grenze L_φ (Shewhart, zweiseitig); Schwelle h_φ (CUSUM, feines Gitter); ARL₀ der Kette auf dem feinen Gitter; Simulation (100 000 Läufe)",
              f"L_φ {sl([c['limit'] for c in na], lambda v: de(v, 4))}; h_φ {sl([c['h_fine'] for c in na], lambda v: de(v, 3))}; ARL₀ {sl([c['arl0_at_h_fine_fine_grid'] for c in na], lambda v: de(v, 2))}; "
              + " / ".join(f"{de(c['sim_arl0'], 1)} ± {de(c['sim_se'], 1)}" for c in na) + f" (zensiert: {sl([c['sim_censored'] for c in na], str)} Läufe)",
              "`test_oracle_residual.py::test_nacheichung_grenze_gegen_integralgleichung_und_tafel`, `test_oracle_residual.py::test_nacheichung_schwelle_gefroren_gegen_kette_und_simulation_aus_der_tabelle`"))
    r.append(("Residuenkarte: ARL₀ für jedes φ (bekanntes φ; zwei Wege: Zellenkette und Simulation der Reihe); H3: ARL₁ bei φ = 0,6, Messversatz δ = 1 mit x₀ / ohne x₀",
              f"{de(CH.NOM, 3)} für φ = 0,3 / 0,6 / 0,9; ARL₁ {de(CH.residual_shewhart_arl(0.6, 1.0, 'offset'), 2)} / {de(1 + (1 - CH.alarm_prob(1.0)) / CH.alarm_prob(CH.residual_innovation_means(0.6, 1.0, 'offset')[1]), 2)}",
              "`test_oracle_residual.py::test_residuenkarte_arl0_ist_370_398_fuer_jedes_phi_zellenkette`, `test_handbeispiele.py::test_h3_residuenkarte_phi_06_messversatz_1`"))
    r.append(("Residuenkarte gegen nachgeeichte Rohkarte, ARL₁ (ν = 0): φ = 0,9, Messversatz δ = 1: Shewhart / CUSUM, je nominell / nachgeeicht / Residuum", _res_text("offset", 0.9, 1.0),
              "`test_claims.py::test_residuenkarte_zellen`, `test_oracle_residual.py::test_arl1_der_residuenkarte_formel_gegen_simulation`"))
    r.append(("dasselbe bei Verschiebung des Prozessmittels, φ = 0,9, δ = 1", _res_text("process", 0.9, 1.0), "`test_claims.py::test_residuenkarte_zellen`"))
    r.append(("Messversatz δ = 2, φ = 0,9: die Residuenkarte ist schneller", _res_text("offset", 0.9, 2.0), "`test_oracle_residual.py::test_messversatz_delta_2_phi_09_residuum_schneller_prozessmittel_langsamer`, `test_claims.py::test_residuenkarte_zellen`"))
    r.append(("Verschiebung des Prozessmittels δ = 2, φ = 0,9: die Residuenkarte bleibt langsam (Shewhart)", _res_text("process", 0.9, 2.0, shewhart_only=True), "`test_claims.py::test_residuenkarte_zellen`"))
    r.append(("φ = 0,6, Messversatz δ = 1: nominelle Rohkarte / nachgeeicht / Residuum", _res_text("offset", 0.6, 1.0), "`test_claims.py::test_residuenkarte_zellen`"))
    r.append(("M/M/1, Lag 1 bei ρ = 0,5 / 0,8 / 0,9 (exakt, Brüche); Lag 1 bis 5 bei ρ = 0,8 (Kette)",
              f"{sl([Q.lag1_exact(Fr(a, b)) for a, b in ((1, 2), (4, 5), (9, 10))], frac)}; ρ = 0,8: {sl(Q.lag_chain(0.8), lambda v: de(v, 4))}",
              "`test_handbeispiele.py::test_h6_mm1_lag1_als_bruch_und_durch_quadratur`, `test_oracle_queue.py::test_lag_1_bis_5_aus_der_kette_gleich_lag1_geschlossen_und_v2_simulation`"))
    cl = [Q.shewhart_closed(x)["arl"] for x in C.RHO_OPTIONS]
    r.append(("M/M/1, Shewhart nach oben bei μ + 3σ, ρ = 0,5 / 0,8 / 0,9: iid-Sicht / wirkliche Reihe (geschlossene Lösung); P(T ≤ 100) iid-Sicht / wirklich",
              f"{sl([Q.shewhart_iid_arl(x) for x in C.RHO_OPTIONS], lambda v: de(v, 1))} / {sl(cl, lambda v: de(v, 1 if v < 1000 else 1))}; "
              f"{sl([1 - (1 - x * math.exp(-(1 - x) * Q.shewhart_threshold(x))) ** 100 for x in C.RHO_OPTIONS], lambda v: de(v, 3))} / {sl([Q.shewhart_real_p100(x) for x in C.RHO_OPTIONS], lambda v: de(v, 3))}",
              "`test_oracle_queue.py::test_shewhart_geschlossene_loesung_gleich_kette_und_tafel`, `test_oracle_queue.py::test_shewhart_p100_gegen_simulation_im_band`"))
    mm = [REF.mm1(x) for x in C.RHO_OPTIONS]
    r.append(("M/M/1, CUSUM einseitig auf (W − μ)/σ, ρ = 0,5 / 0,8 / 0,9: iid-Sicht / wirkliche Reihe (zweidimensionale Kette); P(T ≤ 100) iid / wirklich; Median iid / wirklich",
              f"{sl([c['cusum_iid'] for c in mm])} / {sl([c['cusum_real'] for c in mm])}; {sl([c['p100_iid'] for c in mm], lambda v: de(v, 3))} / {sl([c['p100_real'] for c in mm], lambda v: de(v, 3))}; "
              f"{sl([c['median_iid'] for c in mm], str)} / {sl([c['median_real'] for c in mm], str)}",
              "`test_oracle_queue.py::test_cusum_iid_sicht_und_wirkliche_reihe_gegen_tafel`, `test_oracle_queue.py::test_cusum_wirkliche_reihe_gegen_simulation_im_band`"))
    r.append(("Schiefe und Abhängigkeit addieren sich nicht (CUSUM, M/M/1)", f"Schiefe allein: 370,4 → {de(min(c['cusum_iid'] for c in mm), 0)} bis {de(max(c['cusum_iid'] for c in mm), 0)}; mit Abhängigkeit {de(mm[0]['cusum_real'], 1)} (ρ = 0,5, tiefer), "
              f"{de(mm[1]['cusum_real'], 1)} und {de(mm[2]['cusum_real'], 1)} (höher); bei ρ = 0,9 ist P(T ≤ 100) {de(mm[2]['p100_real'], 3)}, das {de(mm[2]['p100_real'] / 0.2302, 2)}fache des Nominalwerts 0,230",
              "`test_oracle_queue.py::test_schiefe_allein_senkt_abhaengigkeit_hebt_und_senkt_keine_feste_richtung`, `test_claims.py::test_mm1_zusatz`"))
    ex = Q.gate_exact(C.GATE_LAMBDA)
    gp = gate_pooled()
    r.append(("Gate M/M/2/6 (Angebot 1,6; Zeiteinheit = mittlere Abfertigungsdauer, angenommene Lkw), Randverteilung exakt: Verlust / Mittel / Standardabweichung / Schiefe der Wartezeit; Shewhart 3σ iid-Sicht",
              f"{de(ex['loss'], 4)} / {de(ex['mean'], 3)} / {de(ex['sd'], 3)} / {de(ex['skew'], 3)}; {de(Q.gate_shewhart_iid_arl(ex), 1)}",
              "`test_oracle_queue.py::test_gate_randverteilung_gegen_geburts_und_sterbeprozess_und_little`, `test_claims.py::test_gate`"))
    r.append(("Gate, Simulation (2 × 2000 Läufe, zwei Zufalls-Seeds): Shewhart 3σ; CUSUM zweiseitig (k = 0,5, h = 4,77); dessen iid-Sicht; Lag 1 bis 5",
              f"{de(gp['shewhart'][0], 1)} ± {de(gp['shewhart'][1], 1)}; {de(gp['cusum2'][0], 1)} ± {de(gp['cusum2'][1], 1)}; {de(gp['iid_cusum2'][0], 1)} ± {de(gp['iid_cusum2'][1], 1)}; {sl(gp['lag'], lambda v: de(v, 3))}",
              "`test_tables.py::test_gate_tafel_neu_gemessen_gegen_v2_und_plan_08_im_band`, `test_claims.py::test_gate`"))
    d = SK.DISTS
    r.append(("Schiefe, Shewhart 3σ zweiseitig, ARL₀: Exponential (e⁴) / Gamma(4) (e¹⁰·3/683) / Lognormal(0; 0,5) / t(5)", sl([SK.shewhart_two_sided(d[k]) for k in ("expon", "gamma4", "lognorm", "t5")], lambda v: de(v, 2)),
              "`test_oracle_skew.py::test_exponential_e_hoch_4_und_gamma4_geschlossen_gegen_quadratur`, `test_claims.py::test_schiefe`"))
    r.append(("Schiefe, CUSUM einseitig, ARL₀ (dieselben vier Verteilungen)", sl([SK.cusum_arl(d[k]) for k in ("expon", "gamma4", "lognorm", "t5")], lambda v: de(v, 2)),
              "`test_oracle_skew.py::test_cusum_kette_gleich_simulation_im_band`, `test_claims.py::test_schiefe`"))
    r.append(("Vorzeichen-CUSUM, ARL₀ exakt als Bruch bei H = 3 / 8 / 9 (verteilungsfrei bei stetiger Verteilung)", sl([SK.sign_cusum_arl(h) for h in (3, 8, 9)], frac),
              "`test_handbeispiele.py::test_h4_schiefe_geschlossen_und_vorzeichen_cusum_von_hand`, `test_oracle_skew.py::test_vorzeichen_cusum_gleitkomma_gleich_bruch_und_gegen_zufallsbewegung`"))
    nrm = d["normal"]
    r.append(("Normale Werte, Lageverschiebung 1 σ (ν = 0): Vorzeichen-CUSUM (H = 9) / CUSUM; Exponential, Lageverschiebung 2 σ: Vorzeichen-CUSUM",
              f"{de(SK.sign_cusum_shift_arl(nrm, 1.0, 9), 2)} / {de(SK.cusum_arl(nrm, shift=1.0), 2)}; Exponential: {de(SK.sign_cusum_shift_arl(d['expon'], 2.0, 9), 0)} Beobachtungen (untere Stützstelle, Artefakt)",
              "`test_oracle_skew.py::test_artefakt_exponential_nach_lageverschiebung_meldet_nach_genau_h`"))
    seg = lambda p, k: REF.segmentation(p, k)  # noqa: E731
    r.append(("Rückblick: Reihen ohne jeden Bruch (n = 200, Mindestlänge 5, 400 Reihen), Anteil mit mindestens einem gefundenen Bruch bei c = 2, φ = 0 / 0,3 / 0,6 / 0,9: Rohdaten; Residuen (φ bekannt)",
              f"{sl([seg(p, False)['share'][0] for p in p4], lambda v: de(v, 1))} %; {sl([seg(p, True)['share'][0] for p in p4], lambda v: de(v, 1))} %",
              "`test_oracle_segment.py`, `test_tables.py::test_segmentierung_gegen_v14_im_band`, `test_claims.py::test_segmentierung`"))
    r.append(("Rückblick, φ = 0,9, Rohdaten, Anteil bei c = 2 / 4 / 6 / 10 (c = 6 und 10: etwa 62 bis 65 % bzw. etwa 31 bis 36 %, eine Nachrechnung mit 2000 Reihen liegt am unteren Ende)", f"{sl(seg(0.9, False)['share'], lambda v: de(v, 1))} %", "`test_claims.py::test_segmentierung`"))
    t1 = [T.two_state(Fr(x)) for x in (Fr(0), Fr(1, 2), Fr(4, 5))]
    r.append(("Spielzeug Zwei-Zustände-Kette (P(Spitze) = 1/10), φ = 0 / 1/2 / 4/5: Regel A (erste Spitze) / Regel B (zwei Spitzen in Folge)",
              f"{sl([x['arl_a'] for x in t1], frac)} / {sl([x['arl_b'] for x in t1], frac)}", "`test_handbeispiele.py::test_h1_zwei_zustaende_kette_von_hand`"))
    dw = T.discrete_wait()
    r.append(("Spielzeug diskrete Wartezeit: Lag 1 / Lag 2; weitere Beobachtungen aus den Zuständen 0 / 1 / 2; ARL bei Alarm ab W ≥ 3 (iid-Sicht 8)",
              f"{frac(dw['lag1'])} / {frac(dw['lag2'])}; {sl(dw['further'], frac)}; {frac(dw['arl'])} = {de(float(dw['arl']), 3)}, das {de(float(dw['ratio']), 2)}fache",
              "`test_handbeispiele.py::test_h2_diskrete_wartezeit_von_hand`"))
    r.append(("Varianz einer Summe von m AR(1)-Werten: m = 2, φ = 1/2; Grenzwert je Wert bei φ = 4/5", f"{frac(T.sum_variance(2, Fr(1, 2)))} statt 2; {frac(T.sum_variance_limit(Fr(4, 5)))}", "`test_handbeispiele.py::test_h5_beispiele_und_grenzwert`"))
    r.append(("ARL₀-Ziel; einseitige Grenze; Schwelle des einseitigen CUSUM", f"1/(2Φ(−3)) = {de(CH.NOM, 3)}; c = {de(CH.one_sided_limit(), 4)}; h = {de(H, 4)}", "`test_oracle_cusum.py::test_schwelle_h_fuer_arl_370_398`"))
    return r


def de_s(x, nd=1):
    return ("+" if x >= 0 else "−") + de(abs(x), nd)


def _res_text(art, phi, delta, shewhart_only=False):
    cell = REF.resid_cell(art, phi, delta)
    cal = REF.calibration(phi)
    kw = dict(zshift=delta) if art == "offset" else dict(drift=(1 - phi) * delta, stat_mean=(1 - phi) * delta)
    s_nom = CH.shewhart_arl(phi, -3.0, 3.0, 480, **kw)
    s_cal = CH.shewhart_arl(phi, -cal["limit"], cal["limit"], 480, **kw)
    s_res = CH.residual_shewhart_arl(phi, delta, art)
    m1, m2 = CH.residual_innovation_means(phi, delta, art)
    c_res = CH.residual_cusum_arl(H, 0.5, m1, m2)
    s = f"Shewhart {de(s_nom, 2)} / {de(s_cal, 2)} / {de(s_res, 2)}"
    if shewhart_only:
        return s
    return s + f"; CUSUM {de(cell['nominal'], 2)} / {de(cell['calibrated'], 2)} / {de(c_res, 2)}"


def corrections() -> list[tuple[str, str, str, str]]:
    """(Größe, Plan, hier gemessen, Einordnung) für den Abschnitt „Befunde und Korrekturen“."""
    na9 = REF.calibration(0.9)
    t2 = {p: REF.twosided(p) for p in (0.3, 0.6, 0.9)}
    g = gate_pooled()
    out = []
    out.append(("Nachgeeichte CUSUM-Schwelle h_φ bei φ = 0,3 / 0,6 / 0,9", "6,114 / 9,971 / 22,277 (grobe Kette, V14)", sl([REF.calibration(p)["h_fine"] for p in (0.3, 0.6, 0.9)], lambda v: de(v, 3)),
                f"**verschoben bei φ = 0,9:** die grobe Kette gibt bei h = 22,277 die ARL₀ 370,40, das feine Gitter {de(na9['arl0_at_h_coarse_fine_grid'], 2)} ({de(100 * (na9['arl0_at_h_coarse_fine_grid'] / CH.NOM - 1), 1)} % zu hoch); "
                f"die Simulation (100 000 Läufe) mit der neuen Schwelle gibt {de(na9['sim_arl0'], 1)} ± {de(na9['sim_se'], 1)}. Die grobe Kette reproduziert den Plan-Wert (h_grob = {de(na9['h_coarse'], 3)})"))
    out.append(("Nachgeeichte Shewhart-Grenze L_φ bei φ = 0,3 / 0,6 / 0,9", "2,9951 / 2,9605 / 2,7113", sl([REF.calibration(p)["limit"] for p in (0.3, 0.6, 0.9)], lambda v: de(v, 4)),
                "mit Richardson statt der einfachen Kette (n = 480); Abweichung höchstens 1·10⁻⁴"))
    cells = [REF.resid_cell("offset", 0.9, d) for d in (0.5, 1.0, 2.0)] + [REF.resid_cell("process", 0.9, d) for d in (0.5, 1.0, 2.0)]
    out.append(("CUSUM „Roh nachgeeicht“, φ = 0,9: Messversatz δ = 0,5 / 1 / 2; Prozessmittel δ = 0,5 / 1 / 2", "114,29 / 50,69 / 18,53; 119,91 / 58,81 / 27,81", sl([c["calibrated"] for c in cells[:3]], lambda v: de(v, 2)) + "; " + sl([c["calibrated"] for c in cells[3:]], lambda v: de(v, 2)),
                "Folge der neuen Schwelle h_φ bei φ = 0,9: um höchstens 0,7 % kleiner; die Zellen bei φ = 0,3 und 0,6 stimmen mit dem Plan auf 0,1 % überein"))
    out.append(("Zweiseitiger CUSUM, Simulation bei φ = 0 / 0,3 / 0,6 / 0,9 (h = 4,77)", "368,3 ± 2,6 / 84,1 ± 0,6 / 37,6 ± 0,2 / 27,2 ± 0,2 (V14, numpy-Zufall)",
                " / ".join(f"{de(REF.twosided(p)['sim'], 1)} ± {de(REF.twosided(p)['se'], 1)}" for p in (0.0, 0.3, 0.6, 0.9)),
                f"im Band (4 Standardfehler beider Seiten); die Näherung „einseitig/2“ liegt bei φ = 0,3 nur {de(100 * (t2[0.3]['approx'] - t2[0.3]['sim']) / t2[0.3]['sim'], 1)} % über der Simulation (Plan: 1,6 %), bei φ = 0,6 {de(100 * (t2[0.6]['approx'] - t2[0.6]['sim']) / t2[0.6]['sim'], 1)} % (4,6 %), bei φ = 0,9 {de(100 * (t2[0.9]['approx'] - t2[0.9]['sim']) / t2[0.9]['sim'], 1)} % (30 %)"))
    out.append(("Gate, Shewhart 3σ (Simulation)", "128,6 ± 2,4 (V2); 131,5 ± 0,8 (Prüfung, 30 000 Läufe)", f"{de(g['shewhart'][0], 1)} ± {de(g['shewhart'][1], 1)} (2 × 2000 Läufe)",
                "höher als die V2-Zahl, aber im Band von 4 Standardfehlern beider Seiten; die Prüfung mit 30 000 Läufen bleibt die genaueste Einzelmessung"))
    out.append(("Gate, CUSUM zweiseitig (h = 4,77) und dessen iid-Sicht", "42,28 ± 0,38 (Plan 08, 10 000 Läufe); 177,7 (V2)", f"{de(g['cusum2'][0], 1)} ± {de(g['cusum2'][1], 1)}; {de(g['iid_cusum2'][0], 1)} ± {de(g['iid_cusum2'][1], 1)}", "im Band; die Zahl führt Plan 08 (Stück 8), hier nur zum Vergleich"))
    out.append(("Fehlbrüche (Anteil der Reihen, c = 2), Rohdaten bei φ = 0 / 0,3 / 0,6 / 0,9", "3,0 / 36,5 / 87,0 / 99,5 %", sl([REF.segmentation(p, False)["share"][0] for p in (0.0, 0.3, 0.6, 0.9)], lambda v: de(v, 1)) + " %",
                "im Band (Standardfehler bis 2,5 Prozentpunkte, zwei Stichproben zu je 400 Reihen); Residuen: " + sl([REF.segmentation(p, True)["share"][0] for p in (0.0, 0.3, 0.6, 0.9)], lambda v: de(v, 1)) + " % (Plan 3,0 / 3,8 / 3,2 / 4,5)"))
    return out
