"""Abbildungen: alle Achsen fest (fixedrange), logarithmische Achsen mit festem Bereich, Beschriftungen mit Dezimalkomma."""

import numpy as np

import abw_constants as C
import abw_reference as REF
import abw_visualization as V


def _fest(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def test_arl_ueber_phi_log_achse_drei_kurven_linie_und_markierung():
    rows = {p: REF.ar1(p) for p in C.PHI_OPTIONS}
    fig = V.build_arl_over_phi(rows, 0.6)
    assert _fest(fig) and fig.layout.yaxis.type == "log" and fig.layout.yaxis.range[0] < 2 and fig.layout.yaxis.range[1] > 5.5
    kurven = [t for t in fig.data if t.mode == "lines+markers"]
    assert len(kurven) == 3 and all(len(t.x) == len(C.PHI_OPTIONS) for t in kurven)
    assert any(t.name == "370,4 (unabhängige Werte)" and list(t.y) == [C.GAMMA0, C.GAMMA0] for t in fig.data)
    assert max(max(t.y) for t in kurven) < 10 ** fig.layout.yaxis.range[1]            # kein Wert außerhalb des Achsenbereichs


def test_lauflaenge_kurven_enden_bei_eins_und_steigen():
    t = np.array([1, 10, 100, 1000])
    fig = V.build_run_length({"Shewhart zweiseitig": (t, np.array([0.0027, 0.03, 0.24, 0.93]))}, {"Shewhart zweiseitig": (t, np.array([0.0027, 0.03, 0.24, 0.93]))},
                             {"Shewhart zweiseitig": np.array([5, 300, 900, 50])})
    assert _fest(fig) and fig.layout.xaxis.type == "log"
    sim = next(tr for tr in fig.data if "Läufe" in (tr.name or ""))
    assert sim.y[-1] == 1.0 and list(sim.x) == sorted(sim.x)


def test_reihe_markiert_nur_alarmbeobachtungen():
    x = [0.1, 3.4, -0.2, -3.6, 0.0]
    fig = V.build_series(x, [2, 4])
    assert _fest(fig) and fig.layout.yaxis.range[1] > 4
    marker = next(tr for tr in fig.data if tr.mode == "markers")
    assert list(marker.x) == [2, 4] and list(marker.y) == [3.4, -3.6]


def test_balken_mit_komma_beschriftung_und_log_achse():
    shew = dict(nominal=153.0, calibrated=88.6, residual=223.3)
    cus = dict(nominal=14.7, calibrated=50.6, residual=79.6)
    fig = V.build_scheme_bars(shew, cus)
    assert _fest(fig) and fig.layout.yaxis.type == "log" and len(fig.data) == 3
    assert fig.data[0].text[0] == "153,0" and fig.data[2].text[1] == "79,6"
    f4 = V.build_exp4_bars(dict(iid=44.3, real=159.8), dict(iid=87.3, real=70.2))
    assert _fest(f4) and f4.data[1].text[0] == "159,8"
    f5 = V.build_skew_bars({"normal": dict(label="Normal", shewhart2=370.4, shewhart1=370.4, cusum=370.4), "expon": dict(label="Exponential", shewhart2=54.6, shewhart1=43.9, cusum=104.95)}, "expon")
    assert _fest(f5) and f5.data[0].marker.opacity == (0.35, 1.0)                  # die gewählte Verteilung ist kräftig


def test_lag_abbildung_zwei_kurven():
    fig = V.build_lags([0.78, 0.62, 0.50, 0.40, 0.33], 0.5)
    assert _fest(fig) and len(fig.data) == 2 and list(fig.data[1].y)[1] == 0.78 ** 2
