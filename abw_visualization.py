"""Plotly-Abbildungen (alle Achsen fest, fixedrange). Zahlen mit Dezimalkomma in Beschriftungen über `abw_evaluation.de`."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import abw_constants as C
from abw_evaluation import de

GRID = "rgba(128,128,128,0.25)"


def _layout(fig, title, xtitle, ytitle, height=380, logx=False, logy=False, xrange=None, yrange=None, legend_below=True):
    fig.update_layout(title=dict(text=title, font=dict(size=14), x=0.01, xanchor="left", y=0.97, yanchor="top"), height=height, margin=dict(l=60, r=15, t=70, b=50),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0) if legend_below else dict(), hovermode="closest")
    fig.update_xaxes(title=xtitle, fixedrange=True, type="log" if logx else "linear", range=xrange, gridcolor=GRID)
    fig.update_yaxes(title=ytitle, fixedrange=True, type="log" if logy else "linear", range=yrange, gridcolor=GRID)
    return fig


def build_arl_over_phi(rows: dict, phi_sel: float):
    """ARL₀ über φ (logarithmische Achse) für Shewhart zweiseitig, Shewhart einseitig und CUSUM einseitig; Linie bei 370,4; der gewählte Wert hervorgehoben.
    rows: {φ: dict(shewhart2, shewhart1, cusum)}."""
    phis = sorted(rows)
    fig = go.Figure()
    for key, name, col in (("shewhart2", "Shewhart zweiseitig (±3σ)", C.COLOR_SHEWHART), ("shewhart1", "Shewhart einseitig (x > 2,7822)", C.COLOR_SHEWHART1),
                           ("cusum", "CUSUM einseitig (k = 0,5, h = 4,0965)", C.COLOR_CUSUM)):
        fig.add_trace(go.Scatter(x=phis, y=[rows[p][key] for p in phis], mode="lines+markers", name=name, line=dict(color=col, width=2), marker=dict(size=5),
                                 hovertemplate="φ = %{x}<br>ARL₀ = %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[phis[0], phis[-1]], y=[C.GAMMA0, C.GAMMA0], mode="lines", name="370,4 (unabhängige Werte)", hoverinfo="skip", line=dict(color=C.COLOR_NOMINAL, dash="dash", width=1.5)))
    sel = rows[phi_sel]
    fig.add_trace(go.Scatter(x=[phi_sel] * 3, y=[sel["shewhart2"], sel["shewhart1"], sel["cusum"]], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color="#222", width=2))))
    return _layout(fig, "ARL₀ über dem AR-Koeffizienten φ (Beobachtungen bis zum Alarm, bekannte Parameter, zero-state)", "AR-Koeffizient φ", "ARL₀ (logarithmisch)",
                   height=420, logy=True, yrange=[math.log10(25), math.log10(6e5)])


def build_run_length(exact: dict, nominal: dict, sims: dict | None = None):
    """Verteilung der Lauflänge: P(T ≤ t) über t (logarithmisch). exact / nominal: Name -> (t, P); sims: Name -> Lauflängen je Lauf (Treppenkurve)."""
    fig = go.Figure()
    colors = {"Shewhart zweiseitig": C.COLOR_SHEWHART, "CUSUM einseitig": C.COLOR_CUSUM}
    for name, (t, p) in nominal.items():
        fig.add_trace(go.Scatter(x=t, y=p, mode="lines", name=f"{name}, φ = 0", line=dict(color=C.COLOR_NOMINAL, width=1.5, dash="dot")))
    for name, (t, p) in exact.items():
        fig.add_trace(go.Scatter(x=t, y=p, mode="lines", name=f"{name} (exakt)", line=dict(color=colors.get(name, "#333"), width=2.5)))
    for name, rl in (sims or {}).items():
        v = np.sort(np.asarray(rl, float))
        fig.add_trace(go.Scatter(x=v, y=(np.arange(v.size) + 1) / v.size, mode="lines", name=f"{name} (Läufe)", line=dict(color=colors.get(name, "#333"), width=1.2, shape="hv", dash="dash")))
    return _layout(fig, "Wahrscheinlichkeit eines Alarms bis zur Beobachtung t", "Beobachtung t (logarithmisch)", "P(T ≤ t)", height=420, logx=True, xrange=[0, math.log10(8000)], yrange=[0, 1.0])


def build_series(x, alarms, limit=3.0):
    """Eine Reihe ohne Anhalten: Werte, Grenzen ±3 und die Alarmbeobachtungen (|x| > 3)."""
    t = np.arange(1, len(x) + 1)
    x = np.asarray(x, float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name="Werte", line=dict(color=C.COLOR_LINE, width=1)))
    for y in (limit, -limit):
        fig.add_trace(go.Scatter(x=[1, len(x)], y=[y, y], mode="lines", showlegend=False, hoverinfo="skip", line=dict(color=C.COLOR_NOMINAL, dash="dash", width=1.5)))
    if alarms:
        a = np.asarray(alarms)
        fig.add_trace(go.Scatter(x=a, y=x[a - 1], mode="markers", name="Alarmbeobachtung (|x| > 3)", marker=dict(color=C.COLOR_SHEWHART, size=8, symbol="x")))
    lim = max(4.0, float(np.max(np.abs(x))) + 0.5)
    return _layout(fig, "Eine Reihe ohne Anhalten: Alarme kommen in Klumpen", "Beobachtung t", "Wert (marginale Standardabweichung 1)", height=340, yrange=[-lim, lim])


def build_scheme_bars(shew: dict, cus: dict):
    """ARL₁ der drei Schemata (Rohdaten nominell, Rohdaten nachgeeicht, Residuenkarte) für beide Karten, logarithmische Achse. shew / cus: dict(nominal, calibrated, residual)."""
    names = ("Rohdaten, nominell", "Rohdaten, nachgeeicht (ARL₀ 370,4)", "Residuenkarte (φ bekannt)")
    colors = (C.COLOR_NOMINAL, C.COLOR_SHEWHART, C.COLOR_RESID)
    fig = go.Figure()
    for i, (nm, key) in enumerate(zip(names, ("nominal", "calibrated", "residual"))):
        vals = [shew[key], cus[key]]
        fig.add_trace(go.Bar(x=["Shewhart (zweiseitig)", "CUSUM (einseitig)"], y=vals, name=nm, marker_color=colors[i], text=[de(v, 1) for v in vals], textposition="outside", cliponaxis=False))
    mx = max(shew["nominal"], shew["calibrated"], shew["residual"], cus["nominal"], cus["calibrated"], cus["residual"])
    fig = _layout(fig, "ARL₁ nach der Änderung (Beobachtungen bis zum Alarm, ν = 0, zero-state)", "", "ARL₁ (logarithmisch)", height=400, logy=True,
                  yrange=[0, math.log10(mx * 2.5)])
    fig.update_layout(barmode="group")
    return fig


def build_lags(exact_lags, rho: float):
    """Lag 1 bis 5 der M/M/1-Wartezeit (exakt) neben dem AR(1) mit demselben Lag 1 (φ^k)."""
    ks = list(range(1, len(exact_lags) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=list(exact_lags), mode="lines+markers", name="M/M/1 (exakt)", line=dict(color=C.COLOR_SHEWHART, width=2.5), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=ks, y=[exact_lags[0] ** k for k in ks], mode="lines+markers", name="AR(1) mit demselben Lag 1", line=dict(color=C.COLOR_NOMINAL, dash="dot", width=1.5), marker=dict(size=5)))
    return _layout(fig, f"Autokorrelation der Wartezeit bei ρ = {de(rho, 1)}", "Lag k", "Autokorrelation", height=300, yrange=[0, 1.02])


def build_exp4_bars(shew: dict, cus: dict):
    """ARL₀ iid-Sicht gegen wirkliche Reihe für beide Karten (M/M/1), logarithmische Achse, Linie bei 370,4."""
    cats = ["Shewhart (μ + 3σ)", "CUSUM (einseitig)"]
    fig = go.Figure()
    for nm, key, col in (("iid-Sicht (Randverteilung)", "iid", C.COLOR_NOMINAL), ("wirkliche Reihe", "real", C.COLOR_SHEWHART)):
        vals = [shew[key], cus[key]]
        fig.add_trace(go.Bar(x=cats, y=vals, name=nm, marker_color=col, text=[de(v, 1) for v in vals], textposition="outside", cliponaxis=False))
    fig.add_trace(go.Scatter(x=[cats[0], cats[1]], y=[C.GAMMA0, C.GAMMA0], mode="lines", name="370,4 (normale unabhängige Werte)", hoverinfo="skip", line=dict(color="#222", dash="dash", width=1.5)))
    mx = max(shew["iid"], shew["real"], cus["iid"], cus["real"])
    fig = _layout(fig, "ARL₀ der Wartezeit-Reihe: iid-Sicht gegen wirkliche Reihe", "", "ARL₀ (logarithmisch)", height=380, logy=True, yrange=[1, math.log10(mx * 3)])
    fig.update_layout(barmode="group")
    return fig


def build_skew_bars(rows: dict, selected: str):
    """ARL₀ je Verteilung für drei Karten (logarithmisch); die gewählte Verteilung ist kräftig, die anderen blass. rows: {Schlüssel: dict(label, shewhart2, shewhart1, cusum)}."""
    fig = go.Figure()
    keys = list(rows)
    labels = [rows[k]["label"] for k in keys]
    for nm, key, col in (("Shewhart zweiseitig", "shewhart2", C.COLOR_SHEWHART), ("Shewhart einseitig", "shewhart1", C.COLOR_SHEWHART1), ("CUSUM einseitig", "cusum", C.COLOR_CUSUM)):
        vals = [rows[k][key] for k in keys]
        fig.add_trace(go.Bar(x=labels, y=vals, name=nm, marker_color=col, marker_opacity=[1.0 if k == selected else 0.35 for k in keys],
                             text=[de(v, 1) for v in vals], textposition="outside", cliponaxis=False))
    fig.add_trace(go.Scatter(x=[labels[0], labels[-1]], y=[C.GAMMA0, C.GAMMA0], mode="lines", name="370,4", hoverinfo="skip", line=dict(color="#222", dash="dash", width=1.5)))
    mx = max(max(r["shewhart2"], r["shewhart1"], r["cusum"]) for r in rows.values())
    fig = _layout(fig, "ARL₀ bei schiefen Werten (unabhängig, bekannte Parameter, zero-state)", "", "ARL₀ (logarithmisch)", height=400, logy=True, yrange=[1, math.log10(mx * 3)])
    fig.update_layout(barmode="group")
    return fig
