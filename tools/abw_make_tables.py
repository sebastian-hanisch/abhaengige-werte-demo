"""Erzeugt abw_tables.json: die schweren Rechnungen und alle Simulationen, die die App nicht live zeigt.

    python tools/abw_make_tables.py --lang     # rechnet alles (rund 10 bis 20 Minuten, mehrere Prozesse) und schreibt abw_tables.json
    python tools/abw_make_tables.py --check    # kleines Gitter, wenige Läufe: vergleicht mit der gespeicherten Datei (rund 2 s); Rückgabewert 1 bei Abweichung

Inhalt (je Zelle Zufalls-Seed und R, wo Zufall im Spiel ist):
* ar1: je φ des Gitters ARL₀ der Shewhart-Karten (zweiseitig, einseitig; Zellenkette, Richardson), des CUSUM einseitig (zweidimensionale Kette), Lauflängenverteilung, Alarm-Klumpen.
* zweiseitig: einseitiger CUSUM mit h = 4,77 (exakt) gegen die Näherung „einseitig/2“ und die Simulation des zweiseitigen CUSUM.
* nacheichung: Shewhart-Grenze L_φ, CUSUM-Schwelle h_φ (grobes und feines Gitter) und Simulation der ARL₀ mit h_φ.
* residuen_cusum: ARL₁ des CUSUM auf den Rohdaten (nominell, nachgeeicht) für Messversatz und Verschiebung des Prozessmittels.
* mm1: CUSUM auf der M/M/1-Wartezeit (iid-Sicht gegen wirkliche Reihe, Kette) und Simulationen der Karten.
* gate: M/M/2/6 (Simulation, zwei Zufalls-Seeds): Karten, Lag 1 bis 5, iid-Sicht.
* schiefe: nachgeeichte Grenzen und Schwellen je Verteilung.
* segmentierung: Anteil der Reihen ohne Bruch, in denen die Zerlegung einen findet.
Zufall: SplitMix64 (abw_normal.py). Zahlen der Vormessungen sind Erwartungen; Abweichungen stehen in der README."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

import abw_chain as CH  # noqa: E402
import abw_constants as C  # noqa: E402
import abw_cusum as CU  # noqa: E402
import abw_normal as N  # noqa: E402
import abw_queue as Q  # noqa: E402
import abw_scenario as SC  # noqa: E402
import abw_segment as SG  # noqa: E402
import abw_skew as SK  # noqa: E402

OUT = ROOT / "abw_tables.json"
STAND = "2026-11-19"
H = CH.cusum_h_for_arl(0.5)                  # 4,096494807: einseitiger CUSUM, ARL₀ = 370,398
SEED0 = 51000                                 # Zufalls-Seeds dieser Tabelle: je Abschnitt ein eigener Block, innerhalb des Blocks je Zelle ein Abstand, der die Teilläufe (Seed + i) nicht überlappen lässt
FINE = dict(ms=(32, 64), n=192)               # feines Gitter für große Schwellen
COARSE = dict(ms=CU.M_PAIR, n=CU.N_LAT)       # Gitter der Tabellen der Vormessung (m = 16 und 32, n = 128)


def pk(phi: float) -> str:
    return f"{phi:.2f}"


# ------------------------------------------------------------------------------------------------------------------------------------- Rechenstücke
def job_ar1(phi: float) -> tuple[str, dict]:
    """Eine Zeile der AR(1)-Tabelle."""
    t0 = time.time()
    cl = CH.clusters(phi)
    dist = CH.shewhart_distribution(phi)
    rec = dict(
        shewhart2=CH.shewhart_arl_two_sided(phi), shewhart1=CH.shewhart_arl_one_sided(phi),
        p100=dist["p100"], median=dist["median"], q05=dist["q05"], q95=dist["q95"],
        q=cl["q"], cluster_length=cl["cluster_length"], start_rate_inv=cl["start_rate_inv"], share=cl["share"],
    )
    # CUSUM: grobes Gitter (Tabelle der Vormessung); feines Gitter (n = 224, m = 16 / 24, Richardson mit Exponent 2) ab φ = 0,8
    cus = CU.ar1_arl(phi, H, **COARSE)
    rec["cusum_coarse"] = cus
    if phi >= 0.8:
        a16 = CU.ar1_arl(phi, H, ms=(16,), n=224, lr=4.5, richardson=False)
        a24 = CU.ar1_arl(phi, H, ms=(24,), n=224, lr=4.5, richardson=False)
        rec["cusum_fine"] = (2.25 * a24 - a16) / 1.25
    rec["cusum"] = rec.get("cusum_fine") if phi >= 0.95 else cus
    rec["cusum_grid"] = "n = 224, m = 16 / 24" if phi >= 0.95 else "n = 128, m = 16 / 32"
    if phi == 0.0:
        sf = CH.cusum_chain_survival(lambda t: ndtr(np.asarray(t, float)), 0.5, H, 800, 100)   # eindimensionale Kette n = 800 (zweiter Weg; kein Gitterfehler der 2D-Kette)
        d = CU.ar1_distribution(phi, H)
        rec.update(cusum_p100=float(1.0 - sf[100]), cusum_median=d["median"], cusum_q05=d["q05"])
    else:
        d = CU.ar1_distribution(phi, H)
        rec.update(cusum_p100=d["p100"], cusum_median=d["median"], cusum_q05=d["q05"])
    rec["seconds"] = round(time.time() - t0, 1)
    return pk(phi), rec


def sim_chunks(fn, runs: int, chunk: int, seed: int):
    """Setzt fn(chunk, seed + i) mehrfach ein und hängt die Lauflängen aneinander."""
    parts = [fn(chunk, seed + i) for i in range(runs // chunk)]
    return np.concatenate(parts)


def job_twosided(phi: float, idx: int) -> tuple[str, dict]:
    up = CU.ar1_arl(phi, C.H_TWO, **COARSE)
    seed = SEED0 + 100 * idx                      # 51000, 51100, 51200, 51300 (je 10 Teilläufe)
    rl = sim_chunks(lambda r, s: SC.cusum_two_sided_runs(phi, r, s, C.H_TWO, 4000), 20000, 2000, seed)
    mean, se = float(rl.mean()), float(rl.std(ddof=1) / math.sqrt(rl.size))
    return pk(phi), dict(one_sided=up, approx=up / 2.0, sim=mean, se=se, runs=int(rl.size), seed=seed, censored=int((rl > 4000).sum()), tmax=4000)


def cusum_sim(phi: float, h: float, runs: int, seed: int, tmax: int) -> np.ndarray:
    """Lauflängen des einseitigen CUSUM (k = 0,5, Schwelle h) auf AR(1), zero-state, stationärer Start."""
    _, u = SC.ar1_series(N.Stream(seed), runs, tmax, phi)
    return SC.first_alarm(SC.cusum_paths(u) >= h)


def calibrate_cusum(phi: float, **grid) -> float:
    from scipy.optimize import brentq
    return float(brentq(lambda h: CU.ar1_arl(phi, h, **grid) - CH.NOM, 3.0, 40.0, xtol=1e-4))


def job_nacheichung(phi: float, idx: int) -> tuple[str, dict]:
    t0 = time.time()
    lim = CH.calibrated_limit(phi, n=480)
    h_fine = calibrate_cusum(phi, **FINE)
    h_coarse = calibrate_cusum(phi, **COARSE)
    arl_fine_at_fine = CU.ar1_arl(phi, h_fine, **FINE)
    arl_coarse_at_fine = CU.ar1_arl(phi, h_fine, **COARSE)
    arl_fine_at_coarse = CU.ar1_arl(phi, h_coarse, **FINE)
    seed = SEED0 + 1000 + 1000 * idx              # 52000, 53000, 54000 (je 100 Teilläufe)
    rl = sim_chunks(lambda r, s: cusum_sim(phi, h_fine, r, s, 6000), 100000, 1000, seed)
    mean, se = float(rl.mean()), float(rl.std(ddof=1) / math.sqrt(rl.size))
    # Shewhart mit der nachgeeichten Grenze: ARL₀ zweiter Weg (Integralgleichung)
    shew_check = CH.shewhart_nystrom(phi, -lim, lim, n=220)
    return pk(phi), dict(limit=lim, shewhart_check_nystrom=shew_check, h_fine=h_fine, h_coarse=h_coarse, arl0_at_h_fine_fine_grid=arl_fine_at_fine,
                         arl0_at_h_fine_coarse_grid=arl_coarse_at_fine, arl0_at_h_coarse_fine_grid=arl_fine_at_coarse, sim_arl0=mean, sim_se=se, sim_runs=int(rl.size),
                         sim_seed=seed, sim_censored=int((rl > 6000).sum()), seconds=round(time.time() - t0, 1))


def job_resid_cell(art: str, phi: float, delta: float, h_fine: float) -> tuple[str, dict]:
    kind = art
    nominal = CU.ar1_arl(phi, H, kind=kind, delta=delta, **COARSE)
    cal = CU.ar1_arl(phi, h_fine, kind=kind, delta=delta, **FINE)
    return f"{art}|{pk(phi)}|{delta:.1f}", dict(nominal=nominal, calibrated=cal)


def job_mm1(rho: float, idx: int) -> tuple[str, dict]:
    t0 = time.time()
    real = Q.cusum_real_arl(rho, H)
    iid = Q.cusum_iid_arl(rho, H)
    dist_iid = Q.cusum_iid_distribution(rho, H)
    dist_real = Q.cusum_real_distribution(rho, H)
    tmax_s = {0.5: 6000, 0.8: 20000, 0.9: 40000}[rho]
    runs_s = {0.5: 2000, 0.8: 800, 0.9: 300}[rho]
    seed = SEED0 + 5000 + 100 * idx               # 56000, 56100, 56200
    chunk = max(1, 5_000_000 // (2 * tmax_s))
    parts = []
    for i in range((runs_s + chunk - 1) // chunk):
        parts.append(SC.mm1_runs(rho, min(chunk, runs_s - i * chunk), seed + i, tmax_s, True, False, check=False)["shewhart"])
    sh = np.concatenate(parts)
    cs = np.concatenate([SC.mm1_runs(rho, 1000, seed + 50 + i, 6000, False, True, H, check=False)["cusum"] for i in range(5)])
    return f"{rho:.1f}", dict(
        cusum_real=real, cusum_iid=iid, p100_iid=dist_iid["p100"], median_iid=dist_iid["median"], p100_real=dist_real["p100"], median_real=dist_real["median"],
        sim_shewhart=float(sh.mean()), sim_shewhart_se=float(sh.std(ddof=1) / math.sqrt(sh.size)), sim_shewhart_runs=int(sh.size), sim_shewhart_tmax=tmax_s,
        sim_shewhart_censored=int((sh > tmax_s).sum()),
        sim_cusum=float(cs.mean()), sim_cusum_se=float(cs.std(ddof=1) / math.sqrt(cs.size)), sim_cusum_runs=int(cs.size), sim_cusum_censored=int((cs > 6000).sum()),
        seed=seed, seconds=round(time.time() - t0, 1))


def job_gate(idx: int) -> tuple[str, dict]:
    """Gate M/M/2/6, Angebot 1,6: Karten (R = 2000, nobs = 1500), Lag 1 bis 5 (R = 400, nobs = 30000), iid-Sicht des zweiseitigen CUSUM (R = 5000)."""
    t0 = time.time()
    lam = C.GATE_LAMBDA
    ex = Q.gate_exact(lam)
    seed = SEED0 + 6000 + 100 * idx               # 57000, 57100 (+1, +2 für Lag und iid-Sicht)
    g = SC.gate_run(lam, 2000, seed, 1500, 300)
    wlag = Q.gate_waits(N.Stream(seed + 1), 400, 30000, lam, 500)
    lag = Q.acf_rows(wlag, 5)
    iid = SC.gate_iid_draws(N.Stream(seed + 2), 5000, 1500, ex)
    cards = SC.gate_cards(iid, ex, 1500)
    return f"seed{idx + 1}", dict(
        seed=seed, runs=2000, nobs=1500, shewhart=g.shewhart, cusum2=g.cusum2, lag=[list(x) for x in lag], lag_runs=400, lag_nobs=30000,
        iid_cusum2=SC.summarize(cards["cusum2"], 1500), iid_shewhart_sim=SC.summarize(cards["shewhart"], 1500), iid_runs=5000, seconds=round(time.time() - t0, 1))


def job_skew(key: str) -> tuple[str, dict]:
    d = SK.DISTS[key]
    return key, dict(shewhart_limit=SK.shewhart_limit_for(d), cusum_h=SK.cusum_h_for(d))


def job_seg(phi: float, residuals: bool, idx: int) -> tuple[str, dict]:
    seed = SEED0 + 7000 + idx                     # 58000 … 58007
    rec = SG.false_breaks(phi, 400, seed, residuals)
    return f"{pk(phi)}|{'resid' if residuals else 'roh'}", rec


def _run(args):
    fn, a = args
    return fn(*a)


# ------------------------------------------------------------------------------------------------------------------------------------- Aufbau
def build(workers: int) -> dict:
    t0 = time.time()
    out: dict = {"meta": dict(
        stand=STAND, zufall="SplitMix64 (abw_normal.py, Stapel-Fassung, bitgleich mit der Vorlage-Klasse)", python=sys.version.split()[0], numpy=np.__version__,
        hinweis=("Exakte Zahlen (Zellenketten mit Richardson, Integralgleichung, geschlossene Formeln) sind Rechnung; alle anderen Zellen sind Simulation mit Zufalls-Seed und R. "
                 "CUSUM-Ketten: Gitterfehler rund 0,2 %, bei Schwellen h über 10 größer (siehe nacheichung)."),
        h_one=H, grid_coarse="n = 128 latente Zellen, m = 16 und 32 CUSUM-Zellen (Richardson)", grid_fine="n = 192, m = 32 und 64 (Richardson)")}
    jobs: list = []
    jobs += [("ar1", (job_ar1, (p,))) for p in C.PHI_OPTIONS]
    jobs += [("zweiseitig", (job_twosided, (p, i))) for i, p in enumerate(C.PHI_FOCUS)]
    jobs += [("nacheichung", (job_nacheichung, (p, i))) for i, p in enumerate(C.PHI3_OPTIONS)]
    jobs += [("mm1", (job_mm1, (r, i))) for i, r in enumerate(C.RHO_OPTIONS)]
    jobs += [("gate", (job_gate, (i,))) for i in range(2)]
    jobs += [("schiefe", (job_skew, (k,))) for k in C.DIST_KEYS if k != "normal"]
    i = 0
    for phi in (0.0, 0.3, 0.6, 0.9):
        for resid in (False, True):
            jobs.append(("segmentierung", (job_seg, (phi, resid, i))))
            i += 1
    sections: dict = {k: {} for k in ("ar1", "zweiseitig", "nacheichung", "mm1", "gate", "schiefe", "segmentierung")}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [(sec, ex.submit(_run, job)) for sec, job in jobs]
        for sec, f in futs:
            key, val = f.result()
            sections[sec][key] = val
            print(f"  {sec} {key} fertig ({time.time() - t0:.0f} s)", flush=True)
    out.update(sections)
    # CUSUM-Zellen der Residuenkarten-Tabelle (brauchen h_fine aus der Nacheichung)
    cells: dict = {}
    jobs2 = [(job_resid_cell, (art, phi, delta, sections["nacheichung"][pk(phi)]["h_fine"])) for art in ("offset", "process") for phi in C.PHI3_OPTIONS for delta in C.DELTA_OPTIONS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, val in ex.map(_run, jobs2):
            cells[key] = val
            print(f"  residuen_cusum {key} fertig ({time.time() - t0:.0f} s)", flush=True)
    out["residuen_cusum"] = cells
    out["meta"]["rechenzeit_s"] = round(time.time() - t0)
    return out


# ------------------------------------------------------------------------------------------------------------------------------------- Prüfung
def check(path: Path = OUT) -> list[str]:
    """Rechnet kleine Proben mit gröberem Gitter bzw. wenigen Läufen neu und vergleicht mit der gespeicherten Datei; Rückgabe: Liste der Abweichungen."""
    if not path.exists():
        return [f"{path.name} fehlt"]
    tab = json.loads(path.read_text(encoding="utf-8"))
    bad: list[str] = []

    def near(name, a, b, rtol):
        if not abs(a - b) <= rtol * abs(b):
            bad.append(f"{name}: neu {a:.5g} gegen gespeichert {b:.5g} (Toleranz {rtol:g})")
    for phi in (0.0, 0.6, 0.9):
        rec = tab["ar1"][pk(phi)]
        near(f"shewhart2 φ={phi}", CH.shewhart_arl_two_sided(phi, n=240), rec["shewhart2"], 2e-3)
        near(f"shewhart1 φ={phi}", CH.shewhart_arl_one_sided(phi, n=240), rec["shewhart1"], 2e-3)
        near(f"q φ={phi}", CH.clusters(phi, n=240)["q"], rec["q"], 2e-3)
    for phi in (0.3, 0.6):
        rec = tab["ar1"][pk(phi)]
        near(f"cusum φ={phi} (kleines Gitter)", CU.ar1_arl(phi, H, ms=(8, 16), n=64), rec["cusum"], 0.03)
    nach = tab["nacheichung"]["0.30"]
    near("ARL₀ mit h_φ (φ = 0,3, kleines Gitter)", CU.ar1_arl(0.3, nach["h_fine"], ms=(8, 16), n=64), CH.NOM, 0.04)
    for phi, kind in ((0.6, "roh"),):
        seg = tab["segmentierung"][f"{pk(phi)}|{kind}"]
        new = SG.false_breaks(phi, 60, 777, False)
        sh = seg["share"][0] / 100
        se = math.sqrt(max(sh * (1 - sh), 0.005) / seg["runs"] + 0.25 / 60) * 100          # Standardfehler der Differenz (Prozentpunkte); 0,25/60: größte Streuung der kleinen Probe
        if abs(new["share"][0] - seg["share"][0]) > 4 * se:
            bad.append(f"Fehlbrüche φ = {phi}: neu {new['share'][0]:.1f} % gegen gespeichert {seg['share'][0]:.1f} %")
    near("M/M/1 CUSUM wirklich ρ = 0,8 (kleines Gitter)", Q.cusum_real_arl(0.8, H, ms=(8, 16), n=64), tab["mm1"]["0.8"]["cusum_real"], 0.03)
    g = tab["gate"]["seed1"]
    new = SC.gate_run(C.GATE_LAMBDA, 150, 31, 1500, 300)
    se = math.hypot(new.cusum2["se"], g["cusum2"]["se"])
    if abs(new.cusum2["mean"] - g["cusum2"]["mean"]) > 4 * se:
        bad.append(f"Gate CUSUM zweiseitig: neu {new.cusum2['mean']:.1f} ± {new.cusum2['se']:.1f} gegen {g['cusum2']['mean']:.1f} ± {g['cusum2']['se']:.1f}")
    se = math.hypot(new.shewhart["se"], g["shewhart"]["se"])
    if abs(new.shewhart["mean"] - g["shewhart"]["mean"]) > 4 * se:
        bad.append(f"Gate Shewhart: neu {new.shewhart['mean']:.1f} ± {new.shewhart['se']:.1f} gegen {g['shewhart']['mean']:.1f} ± {g['shewhart']['se']:.1f}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lang", action="store_true", help="alles rechnen und abw_tables.json schreiben")
    ap.add_argument("--check", action="store_true", help="kleine Proben gegen die gespeicherte Datei")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args(argv)
    if a.lang:
        tab = build(a.workers)
        OUT.write_text(json.dumps(tab, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{OUT.name} geschrieben ({tab['meta']['rechenzeit_s']} s)")
        return 0
    bad = check()
    for b in bad:
        print("ABWEICHUNG:", b)
    print("OK: alle Proben stimmen" if not bad else f"{len(bad)} Abweichung(en)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
