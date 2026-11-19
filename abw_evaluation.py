"""Einstellungen, Zahlenformat und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Maß in jedem Satz: Beobachtungen bis zum Alarm (Lauflänge T ≥ 1), zero-state (CUSUM-Wert 0, Rauschen stationär), bekannte Parameter, ν = 0 für die Verzögerung (Änderung vor der ersten
Beobachtung); „einseitig“ oder „zweiseitig“ wird genannt; ARL₀-Ziel 370,398 (nicht das gesetzte 370)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from scipy.special import ndtr

import abw_constants as C


@dataclass(frozen=True)
class Settings:
    phi: float = C.DEFAULT_PHI
    r: int = C.DEFAULT_R
    length: int = C.DEFAULT_LEN
    toy: float = C.DEFAULT_TOY
    phi3: float = C.DEFAULT_PHI3
    delta: float = C.DEFAULT_DELTA
    art: str = C.DEFAULT_ART
    rho: float = C.DEFAULT_RHO
    gate_r: int = C.DEFAULT_GATE_R
    dist: str = C.DEFAULT_DIST
    sign_h: int = C.DEFAULT_SIGN_H
    seed: int = C.DEFAULT_SEED


# ------------------------------------------------------------------------------------------------------------------ Zahlenformat (Dezimalkomma, Tausender mit Leerzeichen)
def de(x: float, nd: int = 1) -> str:
    s = f"{abs(x):,.{nd}f}".replace(",", " ").replace(".", ",")
    return ("-" if x < 0 and float(s.replace(" ", "").replace(",", ".")) != 0 else "") + s


def de_s(x: float, nd: int = 1) -> str:
    """Mit Vorzeichen (ASCII-Minus, damit st.metric die Farbe des Deltas erkennt): mein Wert minus Referenz."""
    return ("+" if x >= 0 else "-") + de(abs(x), nd)


def pct(share: float, nd: int = 1) -> str:
    return de(100.0 * share, nd) + " %"


def factor(x: float, ref: float, nd: int = 2) -> str:
    """„0,17fach“ (Verhältnis zu ref, ASCII-Komma)."""
    return de(x / ref, nd) + "fach"


def frac(x: Fraction) -> str:
    """Bruch als „a/b“, ganze Zahlen ohne Nenner."""
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def phi_text(phi: float) -> str:
    """φ mit Dezimalkomma und Minuszeichen, ohne überflüssige Nullen: 0 → „0“, 0,6 → „0,6“, 0,05 → „0,05“, −0,3 → „−0,3“."""
    s = f"{phi:.2f}".rstrip("0").rstrip(".")
    return (s or "0").replace(".", ",").replace("-", "−")


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 1
def verdict_exp1(phi: float, shewhart2: float, shewhart1: float, cusum: float, p100_shewhart: float, p100_cusum: float, cusum_min: float = 55.0) -> tuple[str, str]:
    """Dieselbe Zahl 370,4 bei φ = 0: was sie bei diesem φ für die Karten bedeutet (Beobachtungen bis zum Alarm, bekannte Parameter, zero-state)."""
    nom = C.GAMMA0
    head = f"φ = {phi_text(phi)}: "
    body = (f"Shewhart (zweiseitig, ±3σ) ARL₀ {de(shewhart2, 1)} ({factor(shewhart2, nom)}), innerhalb der ersten 100 Beobachtungen mit Wahrscheinlichkeit {de(p100_shewhart, 3)} (exakt); "
            f"CUSUM (einseitig, k = 0,5, h = 4,0965) {de(cusum, 1)} ({factor(cusum, nom)}), Wahrscheinlichkeit {de(p100_cusum, 3)}. "
            "Dieselbe Zahl 370,4 gilt bei φ = 0 (Nominalwerte: Wahrscheinlichkeit 0,237 bzw. 0,230).")
    if abs(phi) < 1e-12:
        return "info", head + body + " Die Werte sind unabhängig: hier hält die Karte, was die Zahl verspricht (Stück 1)."
    extra = ""
    if phi >= 0.95:
        extra = f" Der CUSUM-Wert fällt bis φ ≈ 0,8 ({de(cusum_min, 1)}) und steigt danach wieder ({de(cusum, 1)} bei φ = {phi_text(phi)})."
    elif phi < 0:
        extra = f" Bei negativem φ steigt der CUSUM-Wert ({factor(cusum, nom, 1)}); die zweiseitige Shewhart-Zahl ist symmetrisch in φ."
    tail = " Einseitig nach oben (Alarm bei x > 2,7822) hängt die Shewhart-Zahl vom Vorzeichen von φ ab: " + de(shewhart1, 1) + "."
    return "warning", head + body + extra + tail


def sim_caption(runs: int, tmax: int, summary: dict, exact: dict) -> str:
    """Gegenprobe: simulierte Lauflängen gegen die exakten Werte (Mittel ± Standardfehler; zensierte Läufe stehen daneben)."""
    names = {"shewhart2": "Shewhart zweiseitig", "shewhart1": "Shewhart einseitig", "cusum": "CUSUM einseitig"}
    parts = []
    for k, nm in names.items():
        s = summary[k]
        z = (s["mean"] - exact[k]) / s["se"] if s["se"] > 0 else 0.0
        parts.append(f"{nm}: {de(s['mean'], 1)} ± {de(s['se'], 1)} (exakt {de(exact[k], 1)}, z = {de_s(z, 1)}, zensiert {s['censored']})")
    return f"Gegenprobe, {runs} Läufe (je Lauf eine Reihe, tmax = {tmax}): " + "; ".join(parts) + "."


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 2
def verdict_exp2(phi: float, q: float, cluster_length: float, start_rate_inv: float, arl: float, share: float) -> tuple[str, str]:
    """Alarm-Klumpen der zweiseitigen Shewhart-Karte: P(zweiter Alarm in Folge | Alarm), Klumpenlänge, Anteil des Anstiegs, den die Klumpung erklärt."""
    p = 2.0 * float(ndtr(-3.0))
    if abs(phi) < 1e-12:
        return "info", (f"φ = {phi_text(phi)}: auf einen Alarm folgt mit Wahrscheinlichkeit {de(q, 4)} sofort ein zweiter (gleich der Alarmrate je Beobachtung {de(p, 4)}); "
                        "die Alarme kommen einzeln, die ARL₀ ist 370,4.")
    text = (f"φ = {phi_text(phi)}: auf einen Alarm folgt mit Wahrscheinlichkeit {de(q, 3)} sofort ein zweiter (φ = 0: 0,0027); die Alarmbeobachtungen je Beobachtung bleiben 1/370,4, "
            f"die mittlere Klumpenlänge ist {de(cluster_length, 2)}. Der Kehrwert der Rate der Klumpenanfänge ist {de(start_rate_inv, 1)} gegen die ARL₀ {de(arl, 1)}: ")
    if arl < 1.1 * C.GAMMA0:
        return "info", text + ("der Anstieg der ARL₀ über 370,4 ist hier klein (höchstens 10 %); ein Anteil, den die Klumpung erklärt, wird bei so kleinem Anstieg nicht angegeben. "
                               "Zweiseitig, Beobachtungen bis zum ersten Alarm, bekannte Parameter, zero-state.")
    text += f"das erklärt {de(share, 2)} des Anstiegs über 370,4; der Rest ist nicht isoliert. Zweiseitig, Beobachtungen bis zum ersten Alarm, bekannte Parameter, zero-state."
    return "warning", text


def verdict_toy(phi: Fraction, res: dict, res0: dict) -> tuple[str, str]:
    """Zwei-Zustände-Kette (Spielzeug, kein AR(1)): dieselbe Spitzenhäufigkeit p = 1/10, Regel A (erste Spitze) gegen Regel B (zwei Spitzen in Folge); exakte Brüche."""
    a, b = res["arl_a"], res["arl_b"]
    a0, b0 = res0["arl_a"], res0["arl_b"]
    text = (f"φ = {frac(phi)}, Spitzenhäufigkeit {frac(res['p'])}: Regel A (Alarm bei der ersten Spitze) ARL {frac(a)} (bei φ = 0: {frac(a0)}), "
            f"Regel B (zwei Spitzen in Folge) {frac(b)} = {de(float(b), 2)} (bei φ = 0: {frac(b0)}).")
    if phi == 0:
        return "info", text + " Ohne Gedächtnis der Kette ist 1/p = 10 und (1 + p)/p² = 110."
    return "warning", text + " Dieselbe Spitzenhäufigkeit: Regel A wartet länger, Regel B kürzer. Spielzeug, kein AR(1); Beobachtungen bis zum ersten Alarm, Start stationär."


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 3
def verdict_exp3(phi: float, delta: float, art: str, shew: dict, cus: dict) -> tuple[str, str]:
    """shew / cus: dict(nominal, calibrated, residual) = ARL₁ (Beobachtungen bis zum Alarm, ν = 0, zero-state) der drei Schemata; art = Messversatz / Verschiebung des Prozessmittels."""
    def cmp(d):
        r = d["residual"] / d["calibrated"]
        if r < 0.95:
            return "schneller"
        if r > 1.05:
            return "langsamer"
        return "gleich schnell"
    cs, cc = cmp(shew), cmp(cus)
    text = (f"φ = {phi_text(phi)}, {art} δ = {de(delta, 1)} (in marginalen σ): Residuenkarte gegen die auf ARL₀ 370,4 nachgeeichte Rohdatenkarte, ARL₁ in Beobachtungen (ν = 0, zero-state): "
            f"Shewhart (zweiseitig) {de(shew['residual'], 1)} gegen {de(shew['calibrated'], 1)}, CUSUM (einseitig) {de(cus['residual'], 1)} gegen {de(cus['calibrated'], 1)}; "
            f"die nominelle Rohdatenkarte (ARL₀ {de(shew['arl0_nominal'], 1)} bzw. {de(cus['arl0_nominal'], 1)}) meldet nach {de(shew['nominal'], 1)} bzw. {de(cus['nominal'], 1)} Beobachtungen"
            + (" (der CUSUM mit ARL₀ unter 370,4 teils schon vor der Änderung)" if cus["arl0_nominal"] < C.GAMMA0 else "") + ". Die Residuenkarte hat ARL₀ 370,4 für jedes φ (φ bekannt).")
    if cs == cc:
        rich = f" Bei dieser Wahl ist die Residuenkarte {cs} (beide Karten)."
    else:
        rich = f" Bei dieser Wahl ist die Residuenkarte mit Shewhart {cs}, mit CUSUM {cc}."
    kind = "info" if (cs == "schneller" and cc == "schneller") else "warning"
    return kind, text + rich + " Das gilt für bekanntes φ (Bestfall)."


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 4
def verdict_exp4(rho: float, lag1: float, shew: dict, cus: dict) -> tuple[str, str]:
    """M/M/1-Wartezeit: iid-Sicht gegen wirkliche Reihe (Beobachtungen bis zum Alarm, ARL₀, zero-state, bekannte Parameter). shew / cus: dict(iid, real, p100_iid, p100_real)."""
    nom_p = 0.2302
    text = (f"M/M/1, ρ = {de(rho, 1)}: Lag 1 = {de(lag1, 3)}. Shewhart bei μ + 3σ (nach oben) {de(shew['real'], 1)} statt {de(shew['iid'], 1)} (iid-Sicht), "
            f"innerhalb von 100 Beobachtungen {de(shew['p100_real'], 3)} statt {de(shew['p100_iid'], 3)}; CUSUM (einseitig) {de(cus['real'], 1)} statt {de(cus['iid'], 1)}, "
            f"{de(cus['p100_real'], 3)} statt {de(cus['p100_iid'], 3)}.")
    if cus["real"] > cus["iid"] and abs(cus["real"] - C.GAMMA0) < 0.1 * C.GAMMA0:
        text += (f" Der CUSUM-Wert liegt hier nahe an 370,4 durch Zufall der beiden Wirkungen (Schiefe senkt, Abhängigkeit hebt); P(T ≤ 100) ist mit {de(cus['p100_real'], 3)} dennoch "
                 f"{de(cus['p100_real'] / nom_p, 2)}mal der nominelle Wert {de(nom_p, 3)}.")
    elif cus["real"] < cus["iid"]:
        text += " Die Wirkungen von Schiefe und Abhängigkeit addieren sich nicht: hier senkt die Abhängigkeit den CUSUM-Wert noch einmal, bei ρ = 0,8 und 0,9 hebt sie ihn."
    else:
        text += " Die Wirkungen von Schiefe und Abhängigkeit addieren sich nicht und haben keine feste Richtung (bei ρ = 0,5 senkt die Abhängigkeit den CUSUM-Wert)."
    return "warning", text


def verdict_gate(gate: dict) -> tuple[str, str]:
    """Gate M/M/2/6 (Simulation, Zeiteinheit = mittlere Abfertigungsdauer, angenommene Lkw): Shewhart und zweiseitiger CUSUM, iid-Sicht gegen wirkliche Reihe."""
    return "info", (f"Gate (M/M/2/6, Angebot 1,6; Simulation, {gate['runs']} Läufe, zwei Zufalls-Seeds): Shewhart 3σ {de(gate['shewhart'], 1)} ± {de(gate['shewhart_se'], 1)} "
                    f"statt {de(gate['shewhart_iid'], 1)} (iid-Sicht, exakt); CUSUM zweiseitig (k = 0,5, h = 4,77) {de(gate['cusum2'], 1)} ± {de(gate['cusum2_se'], 1)} statt "
                    f"{de(gate['cusum2_iid'], 1)} (iid-Sicht, Simulation). Das Gate ist nicht M/M/1: dieselbe Richtung, andere Größen. "
                    "Die Shewhart-Zahl streut: die Prüfung mit 30 000 Läufen gab 131,5 ± 0,8, die genaueste Einzelmessung.")


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 5
def verdict_exp5(label: str, shewhart2: float, shewhart1: float, cusum: float, sign: Fraction, sign_h: int, artifact: bool) -> tuple[str, str]:
    nom = C.GAMMA0
    text = (f"{label}: Shewhart 3σ zweiseitig {de(shewhart2, 2 if shewhart2 < 100 else 1)} ({factor(shewhart2, nom)} von 370,4), einseitig {de(shewhart1, 1)}, CUSUM einseitig {de(cusum, 2 if cusum < 150 else 1)}; "
            f"Vorzeichen-CUSUM (H = {sign_h}) {frac(sign)} bei jeder stetigen Verteilung (verteilungsfrei). Unabhängige Werte, bekannte Parameter, zero-state.")
    if label == C.DIST_LABELS["normal"]:
        return "info", text + " Bei normalen Werten gilt die Zahl 370,4 (wie in Stück 1)."
    if artifact:
        text += (f" Nach einer Lageverschiebung einer nach unten begrenzten Verteilung, deren Verschiebung die untere Stützstelle über den alten Median hebt, meldet der Vorzeichen-CUSUM nach genau {sign_h} Beobachtungen: das ist die untere Stützstelle der "
                 "Verteilung, keine Stärke der Karte.")
    return "warning", text


# ------------------------------------------------------------------------------------------------------------------ Rückblick (Segmentierung)
def seg_caption(rows: dict) -> str:
    return (f"Reihen ohne jeden Bruch (n = 200, Mindestlänge 5, Strafe c = 2): Anteil der Reihen mit mindestens einem gefundenen Bruch {de(rows['roh0'], 1)} / {de(rows['roh3'], 1)} / "
            f"{de(rows['roh6'], 1)} / {de(rows['roh9'], 1)} % bei φ = 0 / 0,3 / 0,6 / 0,9 auf Rohdaten; auf den Residuen (φ bekannt) {de(rows['res_min'], 1)} bis {de(rows['res_max'], 1)} %.")

