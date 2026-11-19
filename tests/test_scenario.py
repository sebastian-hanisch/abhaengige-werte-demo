"""Simulationen (abw_scenario): Zensur, Obergrenzen, Reproduzierbarkeit, Klumpen, Spiegelungsformeln, Konsistenz der Läufe untereinander."""

import numpy as np
import pytest

import abw_normal as N
import abw_queue as Q
import abw_scenario as SC


def test_first_alarm_zaehlt_ab_eins_und_zensiert_mit_t_plus_1():
    m = np.array([[False, False, True, True], [False, False, False, False], [True, False, False, False]])
    assert list(SC.first_alarm(m)) == [3, 5, 1]


def test_obergrenze_der_live_laeufe():
    SC.check_limit(500, 10_000)                                    # genau 5 Mio.: erlaubt
    with pytest.raises(SC.LiveLimitError, match="Obergrenze"):
        SC.check_limit(501, 10_000)
    with pytest.raises(SC.LiveLimitError):
        SC.ar1_runs(0.5, 501, 1)
    with pytest.raises(SC.LiveLimitError):
        SC.mm1_runs(0.5, 4000, 1, 3000)                             # R · tmax · 2 = 24 Mio.
    SC.mm1_runs(0.5, 400, 1, 3000)                                  # 2,4 Mio.: erlaubt


def test_zensierte_laeufe_haben_tmax_plus_1_und_werden_gezaehlt():
    r = SC.ar1_runs(0.9, 100, 3, tmax=20)
    assert all((v == 21).sum() == r.summary[k]["censored"] for k, v in r.lengths.items())
    assert r.summary["shewhart2"]["censored"] > 90                    # bei ARL 832 meldet fast kein Lauf in 20 Beobachtungen


def test_lauflaengen_sind_ganze_zahlen_mindestens_eins_und_reproduzierbar():
    a = SC.ar1_runs(0.3, 100, 9, 500)
    assert all(v.dtype.kind == "i" and v.min() >= 1 for v in a.lengths.values())
    b = SC.ar1_runs(0.3, 100, 9, 500)
    assert all(np.array_equal(a.lengths[k], b.lengths[k]) for k in a.lengths)


def test_ein_einziger_lauf_r_gleich_1_hat_keinen_standardfehler():
    s = SC.summarize(np.array([7]), 100)
    assert np.isnan(s["se"]) and s["mean"] == 7 and s["censored"] == 0


def test_klumpen_laengen_summieren_sich_zur_zahl_der_alarme():
    for phi in (0.0, 0.9, -0.9):
        c = SC.cluster_series(phi, 3000, 17)
        assert sum(c.clusters) == len(c.alarms) and all(abs(c.values[i - 1]) > 3 for i in c.alarms) and len(c.values) == 3000


def test_klumpen_zaehlung_q_schaetzer_nahe_am_exakten_wert():
    c = SC.cluster_counts(0.9, 300, 3000, 23)
    assert c["observations"] == 900_000 and abs(c["q"] - 0.4522) < 0.05 and abs(c["p"] - 1 / 370.398) < 0.0015


def test_change_all_gleich_einzelnen_laeufen_gleiche_reihe():
    runs, seed, tmax = 200, 77, 1500
    allv = SC.change_all(0.9, 1.0, "process", runs, seed, tmax, 2.7112, 22.099)
    assert set(allv) == {f"{k}|{s}" for k in ("shewhart", "cusum") for s in ("nominal", "calibrated", "residual")}
    einzeln = {"shewhart|nominal": ("roh", "shewhart", {}), "shewhart|residual": ("residuum", "shewhart", {}),
               "cusum|nominal": ("roh", "cusum", {}), "cusum|residual": ("residuum", "cusum", {}),
               "cusum|calibrated": ("roh", "cusum", dict(h=22.099)), "shewhart|calibrated": ("roh", "shewhart", dict(limit=2.7112))}
    for key, (scheme, card, kw) in einzeln.items():
        assert np.array_equal(allv[key], SC.change_runs(0.9, 1.0, "process", scheme, card, runs, seed, tmax, **kw)), key


def test_prozessmittel_und_messversatz_beobachtungen_haben_die_erwarteten_mittel():
    _, off = SC.observed_series(N.Stream(5), 4000, 50, 0.9, 1.0, "offset")
    assert abs(off.mean() - 1.0) < 0.05
    _, pro = SC.observed_series(N.Stream(6), 4000, 1, 0.9, 1.0, "process")
    assert abs(pro[:, 0].mean() - 0.1) < 0.05                          # erste Beobachtung: Mittel (1 − φ)δ
    _, pro80 = SC.observed_series(N.Stream(7), 4000, 80, 0.9, 1.0, "process")
    assert abs(pro80[:, -1].mean() - 1.0) < 0.1                        # später: die Reihe hat das neue Mittel erreicht


def test_gate_lauf_reproduzierbar_und_zensur_gezaehlt():
    a = SC.gate_run(1.6, 30, 5, 400, 100)
    b = SC.gate_run(1.6, 30, 5, 400, 100)
    assert a.shewhart == b.shewhart and a.cusum2 == b.cusum2 and len(a.lag) == 5
    assert a.shewhart["censored"] >= 0 and a.runs == 30


def test_gate_waits_ohne_ankunftsmangel_fuer_hohe_last():
    w = Q.gate_waits(N.Stream(3), 20, 200, 6.0, 50)                    # Angebot 6: fast alles wird abgewiesen, trotzdem genug Ankünfte
    assert w.shape == (20, 200) and np.all(w >= 0)
