"""Wenn die Werte nicht unabhängig sind: dieselbe Zahl, das Gegenteil - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 5 der Reihe „Änderungserkennung“: die Zahl ARL₀ = 370,4 gilt für unabhängige Werte. Hier hängen aufeinanderfolgende Werte zusammen (AR(1), Wartezeit einer Warteschlange);
alle Parameter sind bekannt, nur die Abhängigkeit wirkt. Gezählt werden Beobachtungen bis zum Alarm (zero-state).

Lauffähig mit: streamlit run app.py
"""

import time
from fractions import Fraction

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import ndtr

import abw_chain as CH
import abw_constants as C
import abw_cusum as CU
import abw_evaluation as EV
import abw_queue as Q
import abw_reference as REF
import abw_scenario as SC
import abw_skew as SK
import abw_texts as TX
import abw_toy as T
import abw_visualization as V
from abw_layout import dunkles_design, kopf_css
from abw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Wenn die Werte nicht unabhängig sind – Sebastian Hanisch", layout="wide")

H = CH.cusum_h_for_arl(0.5)           # 4,0965 (Integralgleichung), im Text gerundet
de, pct = EV.de, EV.pct


# ------------------------------------------------------------------------------------------------------------------ Rechnungen (zwischengespeichert)
@st.cache_data(show_spinner=False)
def _shewhart_live(phi):
    """Shewhart-Karten live: ARL₀ (Zellenkette n = 480 / 960, Richardson), Lauflängenverteilung der zweiseitigen Karte (n = 480)."""
    t0 = time.perf_counter()
    a2, a1 = CH.shewhart_arl_two_sided(phi), CH.shewhart_arl_one_sided(phi)
    q, pin = CH.shewhart_chain(phi, -3.0, 3.0, 480)
    surv = CH.chain_survival(q, pin, 9000)
    cdf = 1.0 - surv
    return dict(arl2=a2, arl1=a1, cdf=cdf, p100=float(cdf[100]), seconds=time.perf_counter() - t0)


@st.cache_data(show_spinner=False)
def _cusum_dist(phi):
    d = CU.ar1_distribution(phi, H, nmax=3000)
    return dict(cdf=d["cdf"])


@st.cache_data(show_spinner=False)
def _nominal_cusum_cdf():
    surv = CH.cusum_chain_survival(lambda t: ndtr(np.asarray(t, float)), 0.5, H, 400, 3000)
    return 1.0 - surv


@st.cache_data(show_spinner=False)
def _ar1_sim(phi, runs, seed):
    return SC.ar1_runs(phi, runs, seed)


@st.cache_data(show_spinner=False)
def _series(phi, length, seed):
    return SC.cluster_series(phi, length, seed)


@st.cache_data(show_spinner=False)
def _cluster_counts(phi, runs, length, seed):
    return SC.cluster_counts(phi, runs, length, seed)


@st.cache_data(show_spinner=False)
def _exp3(phi, delta, art):
    """Experiment 3: ARL₁ der drei Schemata; Shewhart live (Zellenkette), CUSUM nominell und nachgeeicht aus der gefrorenen Tabelle, CUSUM-Residuenkarte live (Integralgleichung)."""
    cal = REF.calibration(phi)
    ar1 = REF.ar1(phi)
    kw = dict(zshift=delta) if art == "offset" else dict(drift=(1 - phi) * delta, stat_mean=(1 - phi) * delta)
    shew = dict(nominal=CH.shewhart_arl(phi, -3.0, 3.0, 480, **kw), calibrated=CH.shewhart_arl(phi, -cal["limit"], cal["limit"], 480, **kw),
                residual=CH.residual_shewhart_arl(phi, delta, art), arl0_nominal=ar1["shewhart2"])
    cell = REF.resid_cell(art, phi, delta)
    m1, m2 = CH.residual_innovation_means(phi, delta, art)
    cus = dict(nominal=cell["nominal"], calibrated=cell["calibrated"], residual=CH.residual_cusum_arl(H, 0.5, m1, m2), arl0_nominal=ar1["cusum"])
    return shew, cus


@st.cache_data(show_spinner=False)
def _exp3_sim(phi, delta, art, runs, seed):
    cal = REF.calibration(phi)
    return SC.change_all(phi, delta, art, runs, seed, 4000, cal["limit"], cal["h_fine"])


@st.cache_data(show_spinner=False)
def _exp4(rho):
    lags = Q.lag_chain(rho)
    c = Q.shewhart_threshold(rho)
    mean, var = Q.mm1_moments(rho)
    cl = Q.shewhart_closed(rho)
    tab = REF.mm1(rho)
    shew = dict(iid=Q.shewhart_iid_arl(rho), real=cl["arl"], p100_iid=1.0 - (1.0 - rho * np.exp(-(1 - rho) * c)) ** 100, p100_real=Q.shewhart_real_p100(rho))
    cus = dict(iid=tab["cusum_iid"], real=tab["cusum_real"], p100_iid=tab["p100_iid"], p100_real=tab["p100_real"])
    return dict(lags=lags, lag1=Q.lag1_exact(rho), shew=shew, cus=cus, threshold=c, mean=mean, sd=float(np.sqrt(var)), residual=Q.shewhart_closed_residual(rho))


@st.cache_data(show_spinner=False)
def _exp4_sim(rho, runs, seed):
    return SC.mm1_runs(rho, runs, seed, 3000)


@st.cache_data(show_spinner=False)
def _skew_rows():
    rows = {}
    for key in C.DIST_KEYS:
        d = SK.DISTS[key]
        rows[key] = dict(label=C.DIST_LABELS[key], shewhart2=SK.shewhart_two_sided(d), shewhart1=SK.shewhart_one_sided(d), cusum=SK.cusum_arl(d))
    return rows


@st.cache_data(show_spinner=False)
def _skew_shift(key):
    d = SK.DISTS[key]
    return {s: dict(cusum=SK.cusum_arl(d, shift=s), sign={h: SK.sign_cusum_shift_arl(d, s, h) for h in C.SIGN_H_OPTIONS}, up=SK.sign_up_probability(d, s)) for s in (1.0, 2.0)}


@st.cache_data(show_spinner=False)
def _toy(phi):
    return T.two_state(phi), T.two_state(Fraction(0))


# ------------------------------------------------------------------------------------------------------------------ Kopf, Einleitung
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔀 Wenn die Werte nicht unabhängig sind: dieselbe Zahl, das Gegenteil")
st.markdown(TX.intro())

with st.expander("So funktioniert die Demo: Modell, Karten und das Maß", expanded=True):
    st.markdown(TX.SO_FUNKTIONIERT)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Abhängigkeit (Experimente 1 und 2)")
    phi = st.select_slider("AR-Koeffizient φ", options=C.PHI_OPTIONS, key="phi_select", format_func=EV.phi_text,
                           help="Korrelation zwischen Wert t und Wert t + k ist φ hoch k. Die Werte des Gitters sind gerechnet; die Randverteilung bleibt für jedes φ N(0, 1).")
    r_runs = st.slider("Live-Läufe R", *bounds("r_slider"), step=C.R_STEP, key="r_slider", help="Zahl der simulierten Läufe (je Lauf eine Reihe bis zum Alarm; tmax = 10 000, R · tmax höchstens 5 Mio.).")
    length = st.slider("Länge der gezeigten Reihe", *bounds("len_slider"), step=C.LEN_STEP, key="len_slider", help="Beobachtungen der Reihe ohne Anhalten in Experiment 2.")
    toy = st.select_slider("φ der Zwei-Zustände-Kette", options=C.TOY_OPTIONS, key="toy_select", format_func=lambda v: {0.0: "0", 0.5: "1/2", 0.8: "4/5"}[v],
                           help="Spielzeug zum Nachrechnen von Hand (Brüche): P(Spitze) = 1/10, zweiter Eigenwert der Kette = φ.")
    st.header("⚙️ Residuenkarte (Experiment 3)")
    phi3 = st.select_slider("AR-Koeffizient φ (Residuenkarte)", options=C.PHI3_OPTIONS, key="phi3_select", format_func=EV.phi_text, help="Gemessen sind φ = 0,3 / 0,6 / 0,9; die Residuenkarte kennt φ.")
    delta = st.select_slider("Größe der Änderung δ (marginale σ)", options=C.DELTA_OPTIONS, key="delta_select", format_func=lambda v: de(v, 1))
    art = st.radio("Art der Änderung", C.ARTEN, key="art_radio",
                   help="Messversatz: der Sensor liefert ab t = 1 u_t + δ. Verschiebung des Prozessmittels: die Reihe läuft allmählich auf das neue Mittel.")
    st.header("⚙️ Wartezeit am Gate (Experiment 4)")
    rho = st.select_slider("Auslastung ρ (M/M/1)", options=C.RHO_OPTIONS, key="rho_select", format_func=lambda v: de(v, 1),
                           help="Auslastung der Warteschlange, nie die Autokorrelation. Abfertigung 1, Ankünfte λ = ρ.")
    gate_r = st.select_slider("Gate-Läufe R", options=C.GATE_R_OPTIONS, key="gate_r_select", help="Zahl der Läufe der Gate-Simulation (Knopf unter Experiment 4).")
    st.header("⚙️ Schiefe (Tab)")
    dist = st.selectbox("Verteilung", C.DIST_KEYS, key="dist_select", format_func=lambda k: C.DIST_LABELS[k], help="Standardisierte Werte: Mittel 0, Streuung 1, bekannt.")
    sign_h = st.select_slider("Vorzeichen-CUSUM H", options=C.SIGN_H_OPTIONS, key="sign_h_select", help="Alarm bei C ≥ H.")
    st.header("⚙️ Zufall")
    seed = st.number_input("Zufalls-Seed der Live-Läufe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Live-Läufe", width="stretch", on_click=randomize_seed)

sync_query_params({"phi_select": float(phi), "r_slider": int(r_runs), "len_slider": int(length), "toy_select": float(toy), "phi3_select": float(phi3), "delta_select": float(delta),
                   "art_radio": art, "rho_select": float(rho), "gate_r_select": int(gate_r), "dist_select": dist, "sign_h_select": int(sign_h), "seed_input": int(seed)})
S = EV.Settings(round(float(phi), 4), int(r_runs), int(length), float(toy), float(phi3), float(delta), art, float(rho), int(gate_r), dist, int(sign_h), int(seed))
art_key = C.ART_KEY[S.art]

# ====================================================================================================================================================
# Experiment 1
# ====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 1: Dieselbe Karte, andere Abhängigkeit")
st.markdown(TX.exp1_frage())
row = REF.ar1(S.phi)
live = _shewhart_live(S.phi)
cus_cell = _cusum_dist(S.phi)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Shewhart zweiseitig, ARL₀", de(live["arl2"], 1), delta=f"{EV.de_s(live['arl2'] - C.GAMMA0, 1)} gegen 370,4", delta_color="off")
c2.metric("Shewhart einseitig, ARL₀", de(live["arl1"], 1), delta=f"{EV.de_s(live['arl1'] - C.GAMMA0, 1)} gegen 370,4", delta_color="off")
c3.metric("CUSUM einseitig, ARL₀", de(row["cusum"], 1), delta=f"{EV.de_s(row['cusum'] - C.GAMMA0, 1)} gegen 370,4", delta_color="off")
c4.metric("P(T ≤ 100): Shewhart / CUSUM", f"{de(live['p100'], 3)} / {de(row['cusum_p100'], 3)}", delta="nominell 0,237 / 0,230", delta_color="off")
kind, text = EV.verdict_exp1(S.phi, live["arl2"], live["arl1"], row["cusum"], live["p100"], row["cusum_p100"], REF.ar1(0.8)["cusum"])
getattr(st, kind)(text)
st.caption(f"Shewhart live gerechnet in {de(live['seconds'], 2)} s (Zellenkette, Richardson; Rechenzeit dieser Maschine, kein Befund). CUSUM: zweidimensionale Kette, Gitter {row['cusum_grid']}, "
           f"Stand {REF.meta()['stand']} (gefrorene Tabelle). Varianz einer Summe von 10 Werten je Wert: {de(float(T.sum_variance(10, Fraction(S.phi).limit_denominator(100))) / 10, 2)} "
           f"(Grenzwert (1 + φ)/(1 − φ) = {de(float(T.sum_variance_limit(Fraction(S.phi).limit_denominator(100))), 2)}).")

left, right = st.columns(2)
with left:
    st.plotly_chart(V.build_arl_over_phi({p: REF.ar1(p) for p in C.PHI_OPTIONS}, S.phi), width="stretch", key="arl_phi_chart")
with right:
    t_grid = np.unique(np.round(np.logspace(0, np.log10(8000), 160)).astype(int))
    nominal_shew = 1.0 - (1.0 - 2 * float(ndtr(-3.0))) ** t_grid
    cdf_n = _nominal_cusum_cdf()
    exact = {"Shewhart zweiseitig": (t_grid, live["cdf"][t_grid])}
    cc = cus_cell["cdf"]
    ok_t = t_grid[t_grid < len(cc)]
    ok_t = ok_t[~np.isnan(cc[ok_t])]
    exact["CUSUM einseitig"] = (ok_t, cc[ok_t])
    nom_t = t_grid[t_grid < len(cdf_n)]
    nominal = {"Shewhart zweiseitig": (t_grid, nominal_shew), "CUSUM einseitig": (nom_t, cdf_n[nom_t])}
    with st.spinner(f"Simuliere {S.r} Läufe …"):
        sim1 = _ar1_sim(S.phi, S.r, S.seed)
    st.plotly_chart(V.build_run_length(exact, nominal, {"Shewhart zweiseitig": sim1.lengths["shewhart2"], "CUSUM einseitig": sim1.lengths["cusum"]}), width="stretch", key="runlength_chart")
st.caption(EV.sim_caption(S.r, SC.TMAX, sim1.summary, {"shewhart2": live["arl2"], "shewhart1": live["arl1"], "cusum": row["cusum"]})
           + f" Simuliert in {de(sim1.seconds, 2)} s (SplitMix64, Zufalls-Seed {S.seed}); Rechenzeit kein Befund.")

st.markdown("**Exakte Werte für alle gerechneten φ** (Beobachtungen bis zum Alarm, bekannte Parameter, zero-state; Shewhart: Zellenkette und Integralgleichung, CUSUM: zweidimensionale Kette):")
rows = []
for p in C.PHI_OPTIONS:
    r_ = REF.ar1(p)
    rows.append({"φ": de(p, 2), "Shewhart zweiseitig": de(r_["shewhart2"], 1), "Shewhart einseitig": de(r_["shewhart1"], 1), "CUSUM einseitig": de(r_["cusum"], 0 if r_["cusum"] >= 1000 else 1) + (" (nur Kette)" if p == -0.6 else ""),
                 "CUSUM / 370,4": de(r_["cusum"] / C.GAMMA0, 3), "P(T ≤ 100) Shewhart": de(r_["p100"], 3), "P(T ≤ 100) CUSUM": de(r_["cusum_p100"], 3),
                 "Median Shewhart / CUSUM": f"{r_['median']} / {r_['cusum_median'] if r_['cusum_median'] is not None else '> 3000'}", "5 %-Quantil Shewhart": r_["q05"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.markdown(TX.exp1_negativ())
st.warning(TX.exp1_falle())

# ====================================================================================================================================================
# Experiment 2
# ====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 2: Alarme kommen in Klumpen")
st.markdown(TX.exp2_frage())
kind, text = EV.verdict_exp2(S.phi, row["q"], row["cluster_length"], row["start_rate_inv"], row["shewhart2"], row["share"])
getattr(st, kind)(text)
series = _series(S.phi, S.length, S.seed)
counts = _cluster_counts(S.phi, max(1, 600_000 // S.length), S.length, S.seed)
c1, c2, c3, c4 = st.columns(4)
c1.metric("q = P(zweiter Alarm in Folge | Alarm)", de(row["q"], 4), delta="φ = 0: 0,0027", delta_color="off")
c2.metric("Mittlere Klumpenlänge 1/(1 − q)", de(row["cluster_length"], 2))
c3.metric("1/(p(1 − q)) gegen ARL₀", f"{de(row['start_rate_inv'], 1)} / {de(row['shewhart2'], 1)}")
c4.metric("Gezeigte Reihe: Alarme / Klumpen", f"{len(series.alarms)} / {len(series.clusters)}", delta=f"längster Klumpen: {max(series.clusters) if series.clusters else 0}", delta_color="off")
st.plotly_chart(V.build_series(series.values, series.alarms), width="stretch", key="series_chart")
obs_text = f"{counts['observations']:,}".replace(",", " ")
st.caption(f"Eine Reihe ohne Anhalten ({S.length} Beobachtungen, Simulation, Zufalls-Seed {S.seed}; {de(series.seconds, 3)} s). Gegenprobe über {obs_text} Beobachtungen in mehreren Reihen: "
           f"{counts['alarms']} Alarmbeobachtungen (Rate {de(counts['p'], 5)}, exakt 1/370,4 = {de(1 / C.GAMMA0, 5)}), davon {counts['pairs']} Paare in Folge (q geschätzt {de(counts['q'], 3)}, exakt {de(row['q'], 3)}).")
st.markdown(TX.exp2_toy_intro())
res, res0 = _toy(Fraction(S.toy).limit_denominator(100))
kind, text = EV.verdict_toy(Fraction(S.toy).limit_denominator(100), res, res0)
getattr(st, kind)(text)
st.markdown(TX.exp2_negativ())

# ====================================================================================================================================================
# Experiment 3
# ====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 3: Residuenkarte, Schutz und Preis")
st.markdown(TX.exp3_frage())
shew3, cus3 = _exp3(S.phi3, S.delta, art_key)
cal3 = REF.calibration(S.phi3)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nachgeeichte Grenze L_φ / Schwelle h_φ", f"{de(cal3['limit'], 4)} / {de(cal3['h_fine'], 3)}", delta="ARL₀ 370,398 (φ bekannt)", delta_color="off")
c2.metric("ARL₁ Shewhart: Residuum / nachgeeicht", f"{de(shew3['residual'], 1)} / {de(shew3['calibrated'], 1)}", delta=f"{EV.de_s(shew3['residual'] - shew3['calibrated'], 1)} (Residuum minus nachgeeicht)", delta_color="inverse")
c3.metric("ARL₁ CUSUM: Residuum / nachgeeicht", f"{de(cus3['residual'], 1)} / {de(cus3['calibrated'], 1)}", delta=f"{EV.de_s(cus3['residual'] - cus3['calibrated'], 1)} (Residuum minus nachgeeicht)", delta_color="inverse")
c4.metric("ARL₀ der Rohkarte, nominell", f"{de(shew3['arl0_nominal'], 1)} / {de(cus3['arl0_nominal'], 1)}", delta="Shewhart / CUSUM", delta_color="off")
kind, text = EV.verdict_exp3(S.phi3, S.delta, S.art, shew3, cus3)
getattr(st, kind)(text)
st.plotly_chart(V.build_scheme_bars(shew3, cus3), width="stretch", key="scheme_chart")
with st.spinner(f"Simuliere {S.r} Läufe je Karte und Schema …"):
    t0 = time.perf_counter()
    sim3 = _exp3_sim(S.phi3, S.delta, art_key, S.r, S.seed)
    sim3_s = time.perf_counter() - t0
parts = []
for kk, nm in (("shewhart", "Shewhart"), ("cusum", "CUSUM")):
    for sc, sn in (("nominal", "nominell"), ("calibrated", "nachgeeicht"), ("residual", "Residuum")):
        v = sim3[f"{kk}|{sc}"]
        ex = (shew3 if kk == "shewhart" else cus3)[sc]
        se = float(v.std(ddof=1) / np.sqrt(v.size))
        parts.append(f"{nm} {sn}: {de(float(v.mean()), 1)} ± {de(se, 1)} (exakt {de(ex, 1)}, zensiert {int((v > 4000).sum())})")
st.caption(f"Gegenprobe, {S.r} Läufe (ν = 0, tmax = 4000, Zufalls-Seed {S.seed}): " + "; ".join(parts) + ".")
if cal3["h_fine"] > 10:
    st.caption(f"Gitterhinweis: die Schwelle h = {de(cal3['h_fine'], 2)} ist groß, der Gitterfehler der CUSUM-Ketten ist dort größer (gefroren auf feinem Gitter, {REF.meta()['grid_fine']}).")
st.markdown(TX.exp3_negativ())
st.warning(TX.exp3_falle())

# ====================================================================================================================================================
# Experiment 4
# ====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 4: Wartezeit am Gate")
st.markdown(TX.exp4_frage())
e4 = _exp4(S.rho)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lag 1 der Wartezeit (exakt)", de(e4["lag1"], 4), delta=f"Lag 1 bis 5: {' / '.join(de(v, 3) for v in e4['lags'])}", delta_color="off")
c2.metric("Shewhart μ + 3σ: wirklich / iid-Sicht", f"{de(e4['shew']['real'], 1)} / {de(e4['shew']['iid'], 1)}", delta=f"Grenze {de(e4['threshold'], 2)}", delta_color="off")
c3.metric("CUSUM einseitig: wirklich / iid-Sicht", f"{de(e4['cus']['real'], 1)} / {de(e4['cus']['iid'], 1)}", delta="gefrorene Tabelle (Kette)", delta_color="off")
c4.metric("P(T ≤ 100) CUSUM: wirklich / iid", f"{de(e4['cus']['p100_real'], 3)} / {de(e4['cus']['p100_iid'], 3)}", delta="nominell 0,230", delta_color="off")
kind, text = EV.verdict_exp4(S.rho, e4["lag1"], e4["shew"], e4["cus"])
getattr(st, kind)(text)
left, right = st.columns(2)
with left:
    st.plotly_chart(V.build_lags(e4["lags"], S.rho), width="stretch", key="lags_chart")
with right:
    st.plotly_chart(V.build_exp4_bars(e4["shew"], e4["cus"]), width="stretch", key="exp4_chart")
with st.spinner(f"Simuliere {min(S.r, 500)} Läufe auf der Wartezeit …"):
    t0 = time.perf_counter()
    sim4 = _exp4_sim(S.rho, S.r, S.seed)
    sim4_s = time.perf_counter() - t0
sh, cu = sim4["shewhart"], sim4["cusum"]
p_sh, p_cu = float((sh <= 100).mean()), float((cu <= 100).mean())
cens = int((sh > 3000).sum())
sh_text = (f"Shewhart-Mittel {de(float(sh.mean()), 1)} ± {de(float(sh.std(ddof=1) / np.sqrt(sh.size)), 1)} (geschlossen {de(e4['shew']['real'], 1)})" if cens == 0 else
           f"das Mittel der Shewhart-Karte ist live nicht belegbar ({cens} von {sh.size} Läufen zensiert, also länger als tmax); dafür stehen die geschlossene Lösung (exakt) und die gefrorene Simulation (R und Zufalls-Seed in `abw_tables.json`)")
st.caption(f"Gegenprobe, {S.r} Läufe (stationärer Start, tmax = 3000, Zufalls-Seed {S.seed}; {de(sim4_s, 2)} s): innerhalb von 100 Beobachtungen Alarm: Shewhart {de(p_sh, 3)} (exakt {de(e4['shew']['p100_real'], 3)}), "
           f"CUSUM {de(p_cu, 3)} (exakt {de(e4['cus']['p100_real'], 3)}); CUSUM-Mittel {de(float(cu.mean()), 1)} ± {de(float(cu.std(ddof=1) / np.sqrt(cu.size)), 1)} (Kette {de(e4['cus']['real'], 1)}, zensiert {int((cu > 3000).sum())}); {sh_text}.")

st.markdown("**Das Gate selbst** (M/M/2/6: zwei Spuren, 6 Plätze insgesamt, Angebot 1,6, angenommene Lkw; Zeiteinheit = mittlere Abfertigungsdauer). Randverteilung exakt, Karten und Lag als Simulation (gefroren, zwei Zufalls-Seeds):")
ex_gate = Q.gate_exact(C.GATE_LAMBDA)
g1, g2 = REF.gate()["seed1"], REF.gate()["seed2"]
gate_sum = dict(runs=f"{g1['runs']} + {g2['runs']}", shewhart=(g1["shewhart"]["mean"] + g2["shewhart"]["mean"]) / 2, shewhart_se=float(np.hypot(g1["shewhart"]["se"], g2["shewhart"]["se"]) / 2),
                shewhart_iid=Q.gate_shewhart_iid_arl(ex_gate), cusum2=(g1["cusum2"]["mean"] + g2["cusum2"]["mean"]) / 2, cusum2_se=float(np.hypot(g1["cusum2"]["se"], g2["cusum2"]["se"]) / 2),
                cusum2_iid=(g1["iid_cusum2"]["mean"] + g2["iid_cusum2"]["mean"]) / 2)
kind, text = EV.verdict_gate(gate_sum)
getattr(st, kind)(text)
gate_rows = [
    {"Größe": "Verlust (Anteil abgewiesener Lkw), exakt", "Wert": de(ex_gate["loss"], 4)},
    {"Größe": "Wartezeit der Angenommenen: Mittel / Streuung / Schiefe, exakt", "Wert": f"{de(ex_gate['mean'], 3)} / {de(ex_gate['sd'], 3)} / {de(ex_gate['skew'], 3)}"},
    {"Größe": "Lag 1 bis 5 (Simulation, Mittel der zwei Zufalls-Seeds)", "Wert": " / ".join(de((a[0] + b[0]) / 2, 3) for a, b in zip(g1["lag"], g2["lag"]))},
    {"Größe": "Shewhart 3σ: wirkliche Reihe / iid-Sicht (exakt)", "Wert": f"{de(g1['shewhart']['mean'], 1)} ± {de(g1['shewhart']['se'], 1)} und {de(g2['shewhart']['mean'], 1)} ± {de(g2['shewhart']['se'], 1)} / {de(gate_sum['shewhart_iid'], 1)}"},
    {"Größe": "CUSUM zweiseitig (k = 0,5, h = 4,77): wirkliche Reihe / iid-Sicht (Simulation)",
     "Wert": f"{de(g1['cusum2']['mean'], 1)} ± {de(g1['cusum2']['se'], 1)} und {de(g2['cusum2']['mean'], 1)} ± {de(g2['cusum2']['se'], 1)} / {de(g1['iid_cusum2']['mean'], 1)} und {de(g2['iid_cusum2']['mean'], 1)}"},
]
st.dataframe(pd.DataFrame(gate_rows), hide_index=True, width="stretch")
st.caption(f"Gefroren: {g1['runs']} Läufe je Zufalls-Seed ({g1['seed']} und {g2['seed']}), je Lauf 1500 Lkw nach 300 verworfenen; Lag aus {g1['lag_runs']} Läufen mit {g1['lag_nobs']} Lkw. "
           "Karten auf (W − μ)/σ der Randverteilung; die zweiseitige CUSUM-Zahl führt Stück 8 der Reihe (Lastwechsel am Gate), hier nur zum Vergleich.")
if st.button(f"🚚 Gate simulieren ({S.gate_r} Läufe)", key="gate_button", help="Simuliert das Gate live (Zufalls-Seed der Seitenleiste); das Ergebnis erscheint darunter."):
    with st.spinner(f"Simuliere {S.gate_r} Läufe am Gate (je 1500 Lkw) …"):
        st.session_state["gate_live"] = ((S.gate_r, S.seed), SC.gate_run(C.GATE_LAMBDA, S.gate_r, S.seed))
gl = st.session_state.get("gate_live")
if gl is not None:
    (gr, gs), g = gl
    st.info(f"Live ({gr} Läufe, Zufalls-Seed {gs}, {de(g.seconds, 2)} s; Rechenzeit kein Befund): Shewhart 3σ {de(g.shewhart['mean'], 1)} ± {de(g.shewhart['se'], 1)} (zensiert {g.shewhart['censored']}), "
            f"CUSUM zweiseitig {de(g.cusum2['mean'], 1)} ± {de(g.cusum2['se'], 1)} (zensiert {g.cusum2['censored']}); Lag 1 bis 5: " + " / ".join(de(v[0], 3) for v in g.lag) + ".")
st.markdown(TX.exp4_negativ())
d_hand = T.discrete_wait()
st.markdown(TX.exp4_hand())
st.markdown(f"Weitere Beobachtungen bis zum Alarm aus den Zuständen 0 / 1 / 2: {' / '.join(EV.frac(v) for v in d_hand['further'])}; Lag 1 = {EV.frac(d_hand['lag1'])}, Lag 2 = {EV.frac(d_hand['lag2'])}; "
            f"**ARL = {EV.frac(d_hand['arl'])} = {de(float(d_hand['arl']), 3)}** gegen die iid-Sicht {EV.frac(d_hand['iid_arl'])}: das {de(float(d_hand['ratio']), 2)}fache.")

# ====================================================================================================================================================
# Tab Schiefe
# ====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Tab Schiefe: dieselben Karten auf schiefen Werten")
st.markdown(TX.exp5_frage())
rows5 = _skew_rows()
sel = rows5[S.dist]
shift = _skew_shift(S.dist)
sign = SK.sign_cusum_arl(S.sign_h)
kind, text = EV.verdict_exp5(sel["label"], sel["shewhart2"], sel["shewhart1"], sel["cusum"], sign, S.sign_h, SK.sign_cusum_note(SK.DISTS[S.dist], 2.0))
getattr(st, kind)(text)
st.plotly_chart(V.build_skew_bars(rows5, S.dist), width="stretch", key="skew_chart")
skew_rows = []
for key in C.DIST_KEYS:
    r_ = rows5[key]
    cal_ = REF.skew_calibration(key)
    skew_rows.append({"Verteilung": r_["label"], "Shewhart zweiseitig": de(r_["shewhart2"], 2 if r_["shewhart2"] < 100 else 1), "Shewhart einseitig": de(r_["shewhart1"], 1),
                      "CUSUM einseitig": de(r_["cusum"], 2 if r_["cusum"] < 150 else 1),
                      "nachgeeicht: Grenze ±L / Schwelle h": f"{de(cal_['shewhart_limit'], 2)} / {de(cal_['cusum_h'], 2)}" if cal_ else "3 / 4,0965"})
st.dataframe(pd.DataFrame(skew_rows), hide_index=True, width="stretch")
shift_rows = []
for s_, rec in shift.items():
    shift_rows.append({"Lageverschiebung": f"{de(s_, 0)} σ", "CUSUM einseitig, ARL₁": de(rec["cusum"], 2), f"Vorzeichen-CUSUM (H = {S.sign_h}), ARL₁": de(rec["sign"][S.sign_h], 2),
                       "P(Schritt +1 nach der Verschiebung)": de(rec["up"], 4)})
st.markdown(f"**Nach einer Lageverschiebung ({sel['label']}; ν = 0, zero-state):**")
st.dataframe(pd.DataFrame(shift_rows), hide_index=True, width="stretch")
st.caption(f"Vorzeichen-CUSUM, ARL₀ exakt als Bruch: H = 3: {EV.frac(SK.sign_cusum_arl(3))}; H = 8: {EV.frac(SK.sign_cusum_arl(8))}; H = 9: {EV.frac(SK.sign_cusum_arl(9))}. "
           "Shewhart: Exponential e⁴, Gamma(4) e¹⁰·3/683 geschlossen; CUSUM: Brook-Evans-Kette mit Richardson.")
st.markdown(TX.exp5_negativ())

# ====================================================================================================================================================
# Rückblick
# ====================================================================================================================================================
st.markdown("---")
with st.expander("🔙 Und rückblickend? Fehlbrüche der Segmentierung bei Abhängigkeit"):
    st.markdown(TX.RUECKBLICK)
    seg_rows = []
    for p in (0.0, 0.3, 0.6, 0.9):
        roh, res_ = REF.segmentation(p, False), REF.segmentation(p, True)
        seg_rows.append({"φ": de(p, 1), "Rohdaten c = 2": f"{de(roh['share'][0], 1)} % ({de(roh['mean_breaks'][0], 2)})", "c = 4": f"{de(roh['share'][1], 1)} %",
                         "c = 6": f"{de(roh['share'][2], 1)} %", "c = 10": f"{de(roh['share'][3], 1)} %", "Residuen (φ bekannt) c = 2": f"{de(res_['share'][0], 1)} %", "Residuen c = 4": f"{de(res_['share'][1], 1)} %"})
    st.dataframe(pd.DataFrame(seg_rows), hide_index=True, width="stretch")
    st.caption("Anteil der Reihen (in %) mit mindestens einem gefundenen Bruch, in Klammern die mittlere Bruchzahl; Reihen ohne jeden wahren Bruch, n = 200, Mindestlänge 5, Strafe c · ln n, 400 Reihen je Zeile "
               f"(Zufalls-Seed {REF.segmentation(0.0, False)['seed']} ff.), Standardfehler der Anteile höchstens 2,5 Prozentpunkte. Simulation, gefrorene Tabelle. Bei φ = 0,9 gelten die Anteile bei c = 6 und c = 10 als etwa 62 bis 65 % bzw. etwa 31 bis 36 % (eine Nachrechnung mit 2000 Reihen liegt am unteren Ende).")

# ====================================================================================================================================================
# Grenzen und Formeln
# ====================================================================================================================================================
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.ASSUMPTIONS_MARKDOWN)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.MATH_INTRO)
    st.markdown(TX.CONVENTIONS_MARKDOWN)
    st.markdown(TX.MATH_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html)."
)
