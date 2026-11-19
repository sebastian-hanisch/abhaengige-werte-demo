"""Texte der App. Zahlen, die aus den gefrorenen Tabellen oder aus der Rechnung stammen, werden von dort eingesetzt (nie von Hand abgeschrieben);
`ASSUMPTIONS_MARKDOWN` steht wortgleich in der README (tests/test_claims.py vergleicht beide)."""

from __future__ import annotations

import abw_chain as CH
import abw_constants as _C
import abw_queue as Q
import abw_reference as REF
import abw_skew as SK
from abw_evaluation import de


def _row(phi: float) -> dict:
    return REF.ar1(phi)


def intro() -> str:
    r0, r6 = _row(0.0), _row(0.6)
    return f"""
Die Zahl **ARL₀ = 370,4** gehört zur Alarmkarte „Alarm bei mehr als 3 Streuungen“: im Mittel vergehen 370,4 Beobachtungen bis zum Fehlalarm. Sie gilt unter einer Voraussetzung, die selten dasteht: **aufeinanderfolgende Werte sind unabhängig**. Diese Demo hebt sie auf und misst, was die Zahl dann noch verspricht. Alle Parameter bleiben **bekannt** (Mittel, Streuung und der Grad der Abhängigkeit, wie in den Stücken 1 bis 3 der Reihe); nur die Abhängigkeit wirkt, nichts wird geschätzt.

**Die Antwort ist nicht „etwas schlechter“, sondern je nach Karte das Gegenteil.** Bei einer AR(1)-Reihe mit φ = 0,6 meldet die Shewhart-Karte (zweiseitig) im Mittel erst nach {de(r6['shewhart2'], 1)} statt {de(r0['shewhart2'], 1)} Beobachtungen, der einseitige CUSUM schon nach {de(r6['cusum'], 1)}; beide Zahlen sind exakt gerechnet (Beobachtungen bis zum Alarm, zero-state). Die **Falle** ist, eine ARL₀ für unabhängige Werte als Eigenschaft der Karte zu lesen.

**Die Reihen sind ausgedacht:** ein AR(1)-Prozess mit normaler Randverteilung und die Wartezeit eines Lkw-Tors aus der Kette der Demo „Warteschlangen als Markov-Ketten“ (`markov-queue-demo`). In den gelesenen Quellen gibt es keine reale Reihe eines Terminals oder Lagers; wie sich die Karten dort verhalten, wird hier nicht behauptet. Gezählt werden Beobachtungen bis zum Alarm bei bekannten Parametern; die Parameter sind in dieser Demo nicht unbekannt.

Weiter unten: vier Experimente, der Tab **Schiefe**, ein Rückblick auf die Segmentierung, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung** (Symbole, Konventionen, Formeln).
"""


SO_FUNKTIONIERT = """
1. **Die Reihe:** u_t = φ·u_(t−1) + s·ε_t mit s = √(1 − φ²) und unabhängigen normalen ε_t. Die Randverteilung ist für jedes φ N(0, 1) (Mittel 0, Streuung 1, **bekannt**), der Start vor der ersten Beobachtung ist stationär. φ heißt hier **AR-Koeffizient**; die Korrelation zwischen Wert t und Wert t + k ist φ hoch k.
2. **Die Karten:** Shewhart (Alarm bei |x| > 3, zweiseitig; einseitig bei x > 2,7822) und der einseitige CUSUM (S_t = max(0, S_(t−1) + x_t − k), k = 0,5, Alarm bei S_t ≥ 4,0965). Beide sind so eingestellt, dass die ARL₀ bei φ = 0 auf vier Stellen 370,4 ist (Shewhart ±3: 370,398; die CUSUM-Schwelle ist gerundet).
3. **Das Maß:** **Lauflänge** T = Zahl der Beobachtungen bis einschließlich der alarmierenden. ARL₀ = E[T] ohne Änderung, **zero-state** (CUSUM-Wert 0, Rauschen stationär). Verzögerungen (ARL₁) gelten für **ν = 0** (Änderung vor der ersten Beobachtung, Rauschen vorher stationär, CUSUM bei 0); das ist weder das Lorden- noch das Pollak- noch das Shiryaev-Maß.
4. **Zwei Rechenwege:** Die Shewhart-Zahlen sind exakt (Zellenkette mit Richardson-Extrapolation **und** Integralgleichung, beide stimmen auf 5 · 10⁻⁴ überein). Die CUSUM-Zahlen sind Gitterwerte einer zweidimensionalen Kette (Zustand = CUSUM-Zelle und letzter Wert; Richardson, Fehler rund 0,2 %). Simulationen mit Zufalls-Seed sind Gegenproben und stehen mit Standardfehler daneben.

*Zwei Arten von Zahlen:* **live** gerechnete (Shewhart-Ketten, Residuenkarten, Wartezeit, Schiefe, Simulationen mit Zeitangabe) und **gefrorene** Tabellen (`abw_tables.json`: CUSUM-Ketten auf feinem Gitter, Nacheichung, Gate, Segmentierung; mit Stand, Zufalls-Seed und R).
"""


# ------------------------------------------------------------------------------------------------------------------ Experiment 1
def exp1_frage() -> str:
    return (
        "**Kernfrage 1: Was verspricht die Zahl ARL₀, wenn die Werte abhängig sind, und warum steigt sie für die eine Karte und fällt für die andere?** "
        "Die Karten bleiben dieselben und die Parameter bekannt; nur φ ändert sich. Gezeigt wird die exakt gerechnete ARL₀ beider Karten mit der Lauflängenverteilung, "
        "daneben eine Simulation derselben Reihen (Gegenprobe, mit Zufalls-Seed). Für den CUSUM hilft ein Blick auf die Summen: die Varianz einer Summe von m AR(1)-Werten ist m + 2·Σ(m − j)·φ^j, "
        "also je Wert rund (1 + φ)/(1 − φ)-mal so groß wie bei unabhängigen Werten (Zeile unten); eine Deutung, kein Beweis der ARL-Richtung."
    )


def exp1_negativ() -> str:
    c = {p: _row(p)["cusum"] for p in (0.7, 0.8, 0.9, 0.98)}
    two = REF.twosided(0.9)
    return (
        "**Ehrlicher Negativbefund.** Die Richtung gilt nicht für alle φ und nicht für jede Karte: der CUSUM-Wert fällt im Gitter bis φ = 0,8 "
        f"({de(c[0.8], 1)}; {de(c[0.7], 1)} bei 0,7) und steigt danach wieder ({de(c[0.9], 1)} bei 0,9, {de(c[0.98], 1)} bei 0,98); der Ort des Minimums ist nicht genauer bestimmt. "
        "Bei **negativem** φ steigt er stark. Die zweiseitige Shewhart-Zahl ist symmetrisch in φ, die einseitige nicht. Der **zweiseitige CUSUM** ist nur als Simulation belegt: "
        f"die Näherung „einseitig/2“ liegt bei h = 4,77 und φ = 0,9 bei {de(two['approx'], 1)} gegen simuliert {de(two['sim'], 1)} ± {de(two['se'], 1)} ({de(100 * (two['approx'] - two['sim']) / two['sim'], 0)} % zu hoch)."
    )


def exp1_falle() -> str:
    r0, r9 = _row(0.0), _row(0.9)
    return (
        "**Falle:** „ARL₀ = 370,4“ als Eigenschaft der Karte zu lesen. Bei φ = 0,9 ist dieselbe Zahl für die Shewhart-Karte "
        f"{de(r9['shewhart2'], 1)} ({de(r9['shewhart2'] / r0['shewhart2'], 2)}fach) und für den CUSUM {de(r9['cusum'], 1)} ({de(r9['cusum'] / r0['cusum'], 2)}fach)."
    )


# ------------------------------------------------------------------------------------------------------------------ Experiment 2
def exp2_frage() -> str:
    return (
        "**Kernfrage 2: Warum steigt die ARL₀ der Shewhart-Karte, obwohl die Alarmrate je Beobachtung gleich bleibt?** Läuft die Karte ohne anzuhalten weiter, ist die Rate der Alarmbeobachtungen "
        "für jedes φ gleich p = 1/370,4 (Stationarität). Aber die Alarme kommen in **Klumpen**: auf einen Alarm folgt mit Wahrscheinlichkeit q sofort ein zweiter. "
        "Dann gibt es weniger Klumpenanfänge je Beobachtung, und die Zeit bis zum **ersten** Alarm wird länger (rechnerisch 1/(p(1 − q)); wie viel das vom Anstieg erklärt, steht im Negativbefund). Gemessen: q aus der Quadratur (zweiter Weg: Einschluss-Ausschluss der Kette)."
    )


def exp2_toy_intro() -> str:
    return (
        "**Zum Nachrechnen von Hand: ein Spielzeug (Zwei-Zustände-Kette, kein AR(1)).** Zustand 0 = ruhig, 1 = Spitze, P(Spitze) = p = 1/10; die Übergänge sind a = p(1 − φ) und b = (1 − p)(1 − φ). "
        "**Regel A** meldet bei der ersten Spitze (Karte ohne Gedächtnis), **Regel B** bei zwei Spitzen in Folge (Gedächtnis). Dieselbe Spitzenhäufigkeit, aber Regel A wartet bei Abhängigkeit länger und Regel B kürzer; "
        "Regel B ist nicht monoton in φ (wie der CUSUM, dessen Minimum im Gitter bei φ = 0,8 liegt; bei Regel B liegt es bei kleinerem φ)."
    )


def exp2_negativ() -> str:
    r9 = _row(0.9)
    return (
        "**Ehrlicher Negativbefund.** Die Klumpung erklärt den Anstieg nur teilweise: bei φ = 0,9 sind es "
        f"{de(r9['share'], 2)} des Anstiegs über 370,4 (Kehrwert der Rate der Klumpenanfänge {de(r9['start_rate_inv'], 1)} gegen ARL₀ {de(r9['shewhart2'], 1)}); der Rest ist nicht isoliert. "
        "„Alarme kommen in Klumpen“ ist gemessen, „das erklärt den Anstieg“ bei φ = 0,9 nur zu etwa zwei Dritteln (bei φ = 0,6: 0,87)."
    )


# ------------------------------------------------------------------------------------------------------------------ Experiment 3
def exp3_frage() -> str:
    return (
        "**Kernfrage 3: Was schützt die Residuenkarte, und was kostet sie?** Die Residuenkarte überwacht e_t = (x_t − φ·x_(t−1))/s statt x_t, mit **bekanntem** φ; die Innovationen sind unabhängig, "
        "ihre ARL₀ ist für jedes φ gleich 370,398. Die Rohdatenkarte mit Standardschwelle meldet beim CUSUM oft schon **vor** der Änderung (ARL₀ 63 statt 370 bei φ = 0,9): ihre ARL₁ sieht nur deshalb schnell aus; bei der Shewhart-Karte ist die ARL₀ dagegen größer als 370. Fair ist der Vergleich mit der "
        "**nachgeeichten** Rohkarte (Grenze L_φ bzw. Schwelle h_φ so, dass die ARL₀ wieder 370,398 ist). Zwei Arten der Änderung: **Messversatz δ** (der Sensor liefert ab t = 1 u_t + δ, die Dynamik bleibt) und "
        "**Verschiebung des Prozessmittels** um δ (die Reihe läuft allmählich auf das neue Mittel); δ in marginalen Standardabweichungen. ARL₁ gilt für ν = 0."
    )


def exp3_negativ() -> str:
    lim = REF.calibration(0.9)["limit"]
    res_off, res_pro = CH.residual_shewhart_arl(0.9, 2.0, "offset"), CH.residual_shewhart_arl(0.9, 2.0, "process")
    cal_off = CH.shewhart_arl(0.9, -lim, lim, 480, zshift=2.0)
    cal_pro = CH.shewhart_arl(0.9, -lim, lim, 480, drift=0.2, stat_mean=0.2)
    return (
        "**Ehrlicher Negativbefund.** „Die Residuenkarte erkennt langsamer“ gilt nicht überall: bei großem Messversatz (δ = 2, φ = 0,9) ist sie schneller "
        f"(Shewhart {de(res_off, 2)} gegen {de(cal_off, 2)} der nachgeeichten Rohkarte), weil die erste Beobachtung dann den vollen Sprung δ/s sieht (eigene Herleitung). Bei **Verschiebung des Prozessmittels** sieht "
        f"die Residuenkarte nur (1 − φ)·δ/s je Beobachtung und bleibt langsam (Shewhart, δ = 2, φ = 0,9: {de(res_pro, 1)} gegen {de(cal_pro, 1)}). Das passt zu der Aussage von Larsen u. a., Residuen absorbierten die Verschiebung "
        "(aus deren Einleitung zitiert); hier ist es für dieses Modell nachgerechnet."
    )


def exp3_falle() -> str:
    return (
        "**Falle zweiter Hälfte:** „bekanntes φ“ ist der **Bestfall**. In der Praxis wird φ geschätzt (Stück 4b der Reihe misst, wie die bedingte ARL₀ dann streut), und der AR(1)-Filter muss zur Reihe passen: "
        "eine frühere Messung am Gate (nicht neu gemessen) zeigt, dass er dort nicht passt."
    )


# ------------------------------------------------------------------------------------------------------------------ Experiment 4
def exp4_frage() -> str:
    return (
        "**Kernfrage 4: Was geschieht an einer echten Warteschlange?** Die Wartezeiten aufeinanderfolgender Lkw sind stark abhängig **und** schief (Atom bei 0). Gezeigt wird die M/M/1-Wartezeit (Lindley-Kette, Abfertigung 1, Auslastung ρ = λ, "
        "stationär) als exakter Fall: Lag 1 bis 5, Shewhart nach oben bei μ + 3σ (geschlossene Lösung), CUSUM einseitig auf (W − μ)/σ. **iid-Sicht** heißt: dieselbe Randverteilung, aber unabhängige Werte. "
        "Darunter das Gate selbst (M/M/2/6: zwei Spuren, 6 Plätze **insgesamt**, angenommene Lkw; Simulation)."
    )


def exp4_negativ() -> str:
    g = REF.gate()
    sh = (g["seed1"]["shewhart"]["mean"] + g["seed2"]["shewhart"]["mean"]) / 2
    ar1_sh = CH.shewhart_arl_two_sided(0.781)
    return (
        "**Ehrlicher Negativbefund.** Schiefe und Abhängigkeit **addieren sich nicht** und haben keine feste Richtung: die Schiefe allein senkt die CUSUM-ARL₀ von 370,4 auf 87 bis 104; die zusätzliche Abhängigkeit senkt sie weiter bei ρ = 0,5 "
        "und hebt sie bei ρ = 0,8 und 0,9. Ein Wert in der Nähe von 370,4 ist hier Zufall: bei ρ = 0,9 ist P(T ≤ 100) trotzdem noch das 1,45fache des Nominalwerts. **Das Gate ist nicht M/M/1** (die Blockierung begrenzt die Warteschlange): dieselbe Richtung, andere Größen "
        f"(Shewhart am Gate {de(sh, 1)} statt der {de(Q.gate_shewhart_iid_arl(Q.gate_exact(_C.GATE_LAMBDA)), 1)} der iid-Sicht, also das {de(sh / Q.gate_shewhart_iid_arl(Q.gate_exact(_C.GATE_LAMBDA)), 1)}fache; die genauere Messung mit 30 000 Läufen gab 131,5 ± 0,8, das 2,2fache; bei M/M/1 und ρ = 0,8 das 21,8fache). "
        f"Ein AR(1) mit demselben Lag 1 (φ = 0,781) gäbe für die zweiseitige Shewhart-Karte {de(ar1_sh, 1)}: die Wartezeit ist kein AR(1) (Atom bei 0, Schiefe)."
    )


def exp4_hand() -> str:
    return (
        "**Zum Nachrechnen von Hand: diskrete Wartezeit** W' = W + 1 mit Wahrscheinlichkeit 1/3, sonst max(0, W − 1). Stationär ist π_n = (1/2)^(n+1) (Mittel 1, Varianz 2). "
        "Alarm bei W ≥ 3: Randverteilung P(W ≥ 3) = 1/8, iid-Sicht 8. Derselbe Mechanismus wie bei M/M/1, mit drei Zuständen."
    )


# ------------------------------------------------------------------------------------------------------------------ Experiment 5
def exp5_frage() -> str:
    return (
        "**Nebenbefund: Schiefe allein (unabhängige Werte).** Dieselben Karten auf standardisierten Werten (Mittel 0, Streuung 1, bekannt): die Zahl 370,4 gilt nur für normale Werte. "
        "Der **Vorzeichen-CUSUM** (Schritt +1 bei einem Wert über dem Median, −2 sonst) ist verteilungsfrei bei stetiger Verteilung: seine ARL₀ ist ein exakter Bruch."
    )


def exp5_negativ() -> str:
    nrm = SK.DISTS["normal"]
    return (
        "**Ehrlicher Negativbefund und Artefakt.** Der Vorzeichen-CUSUM hat bei jeder stetigen Verteilung dieselbe ARL₀, ist aber bei normalen Werten nach einer Lageverschiebung langsamer als der CUSUM "
        f"(1 σ: {de(SK.sign_cusum_shift_arl(nrm, 1.0, 9), 2)} gegen {de(SK.cusum_arl(nrm, shift=1.0), 2)} Beobachtungen, ν = 0; H = 9). "
        "Seine „9 Beobachtungen“ nach einer Lageverschiebung einer nach unten begrenzten Verteilung, deren Verschiebung die untere Stützstelle über den alten Median hebt (Exponential, 2 σ), sind ein Artefakt: jeder neue Wert liegt über dem alten Median, der Schritt ist immer +1."
    )


# ------------------------------------------------------------------------------------------------------------------ Rückblick
RUECKBLICK = """
**Und rückblickend?** Wer eine Reihe **nachträglich** in Strecken zerlegt (optimale Zerlegung, Strafe c·ln n je Bruch, Gauß-Kosten mit Streuung 1), hat dieselbe Falle: bei autokorreliertem Rauschen findet die Zerlegung Brüche, die es nicht gibt.
Die Tabelle zählt **Reihen** (n = 200, Mindestlänge 5, 400 Reihen je Zeile, **ohne jeden wahren Bruch**), auf denen mindestens ein Bruch gefunden wird; die Residuen setzen wie die Karten **bekanntes φ** voraus. Strafwahl und Fenster stehen in der Demo
`segmentierung-offline-demo` (Stück 7 der Reihe). Das stützt die Beobachtung, dass Standardstrafen die Zahl der Brüche bei autokorreliertem Rauschen überschätzen, **nur für dieses Modell und diese Kosten**.
"""


# ------------------------------------------------------------------------------------------------------------------ Grenzen und Formeln
ASSUMPTIONS_MARKDOWN = """
* **Modell:** AR(1) mit normaler Randverteilung, bekannten Parametern (Mittel, Streuung und φ), zero-state, ν = 0; andere Abhängigkeit (AR(2), saisonal, langes Gedächtnis) und andere Änderungen (Streuung, Dynamik) sind nicht gemessen.
* **Maß:** Beobachtungen bis zum Alarm, Zähleinheit je Satz genannt. Das Lorden-, das Pollak- und das Shiryaev-Maß sind bei abhängigen Werten **nicht gemessen**; alle Verzögerungen gelten für ν = 0.
* **Zweiseitiger CUSUM auf AR(1)** ist nur als Simulation belegt (exakt nur einseitig); die Näherung „einseitig/2“ gilt nur bis φ = 0,3. φ = −0,6, 0,95 und 0,98 (CUSUM) sind nicht durch Simulation geprüft (0,95 und 0,98 nur auf dem feinen Gitter).
* **CUSUM-Ketten:** Gitterfehler rund 0,2 %, bei großen Schwellen (h über 10) größer; die nachgeeichte Schwelle bei φ = 0,9 (h = 22) ist auf etwa 1 % genau.
* **Residuenkarte** mit bekanntem φ ist der Bestfall; mit geschätztem φ̂ streut die bedingte ARL₀ (Stück 4b der Reihe). Messversatz und Verschiebung des Prozessmittels sind gewählte Änderungsarten.
* **M/M/1 ist nicht das Gate:** das Gate (M/M/2/6, Simulation) hat andere Größen; EWMA, Zusatzregeln, MEWMA und verteilungsfreie Karten bei abhängigen Werten sind nicht gemessen, ebenso schiefe abhängige Werte außer M/M/1.
* **Ausgedachte Reihen:** ein Nutzen einer Alarmregel in Logistik, Lager oder Fertigung ist nicht belegt.
"""

MATH_INTRO = """
**Gegeben:** die Reihe u_t (AR(1), bekannte Parameter) und eine Karte. **Gesucht:** die Verteilung der Lauflänge T (Beobachtungen bis zum Alarm), vor allem ARL₀ = E[T] und P(T ≤ 100).

| Symbol | Bedeutung |
|---|---|
| φ | AR-Koeffizient; die Korrelation zwischen Wert t und Wert t + k ist φ^k |
| s | √(1 − φ²), Streuung der Innovation |
| T | Lauflänge: Zahl der Beobachtungen bis einschließlich der alarmierenden |
| ARL₀, ARL₁ | E[T] ohne Änderung bzw. nach der Änderung (ν = 0) |
| k, h | Referenzwert (0,5) und Schwelle des CUSUM |
| δ | Größe der Änderung in marginalen Standardabweichungen |
| ρ | Auslastung der Warteschlange (nie die Autokorrelation) |
| p, q | Alarmrate je Beobachtung und P(zweiter Alarm in Folge ǀ Alarm) |
"""

CONVENTIONS_MARKDOWN = """
**Konventionen.** *Zähleinheiten:* **Beobachtungen** (Lauflänge, Verzögerung), **Läufe** (Simulation; ein Lauf = eine Reihe), **Reihen** (Segmentierung). *Zero-state:* CUSUM-Wert 0, Rauschen stationär; ein Anfangsvorsprung kommt nicht vor.
*Einseitig/zweiseitig* wird immer genannt: der einseitige CUSUM hat h = 4,0965 (ARL₀ auf vier Stellen 370,4 bei φ = 0), der zweiseitige h = 4,7749 für dasselbe Ziel; die Demo rechnet den zweiseitigen mit dem gerundeten h = 4,77 (ARL₀ 368,6), das ist **nicht** dieselbe Karte.
*Ziel:* ARL₀ = 1/(2Φ(−3)) = 370,398 (nicht das gesetzte 370). *Nachgeeicht:* die Schwelle so gesetzt, dass die ARL₀ der Rohdatenkarte 370,398 ist. *Residuenkarte:* dieselbe Karte auf e_t = (x_t − φ·x_(t−1))/s mit bekanntem φ; die Karte kennt den Wert x₀ vor dem ersten
Beobachtungsintervall (vom Messversatz nicht betroffen), e₁ hat bei Messversatz δ das Mittel δ/s, ab e₂ das Mittel (1 − φ)·δ/s; nähme sie stattdessen e₁ = x₁ ohne x₀, wäre die ARL₁ für φ = 0,6 und δ = 1 nicht 150,00, sondern 152,69.
*Gate* heißt Lkw-Tor; Hotelling-T² kommt hier nicht vor (Stück 6).
"""

MATH_MARKDOWN = r"""
**AR(1) und Randverteilung.**

$$u_t = \varphi\,u_{t-1} + s\,\varepsilon_t,\qquad s = \sqrt{1-\varphi^2},\quad \varepsilon_t \sim N(0,1)\ \text{unabhängig},\quad u_t \sim N(0,1).$$

**Shewhart-Karte auf AR(1) (Zellenkette).** Der Zustand ist der letzte Wert; Q_jl ist die Wahrscheinlichkeit, aus der Zelle j in die Zelle l zu gehen, ohne Alarm; π ist die Masse der ersten Beobachtung:

$$\mathrm{ARL}_0 = 1 + \pi^{\top}(I-Q)^{-1}\mathbf{1}.$$

Zweiter Weg: die Integralgleichung

$$g(x) = 1 + \int_{-3}^{3} f(y\mid x)\,g(y)\,dy,\qquad f(y\mid x) = \tfrac{1}{s}\,\phi\!\left(\tfrac{y-\varphi x}{s}\right),\qquad \mathrm{ARL}_0 = \int \phi(x)\,g(x)\,dx.$$

**Alarm-Klumpen.** Mit p = 2Φ(−3) und q = P(|x_(t+1)| > 3 | |x_t| > 3):

$$q = \frac{2}{p}\int_{3}^{\infty}\phi(x)\left[\Phi\!\left(\tfrac{-3-\varphi x}{s}\right) + 1-\Phi\!\left(\tfrac{3-\varphi x}{s}\right)\right]dx,\qquad \frac{1}{p\,(1-q)}\ \text{(Kehrwert der Rate der Klumpenanfänge)}.$$

**CUSUM einseitig** (Alarm bei S_t ≥ h):

$$S_t = \max(0,\; S_{t-1} + x_t - k).$$

Auf AR(1) ist (S, letzter Wert) die Markov-Kette; die CUSUM-Zellen nach Brook und Evans (Breite w = h/(m − ½)) und die latenten Zellen bilden die zweidimensionale Kette, Richardson (4A(32) − A(16))/3.

**Residuenkarte** (φ bekannt):

$$e_t = \frac{x_t - \varphi\,x_{t-1}}{s}\sim N(0,1)\ \text{unabhängig},\qquad \mathrm{ARL}_1 = 1 + \frac{1-p_1}{p_2},\quad p_i = 1-\Phi(3-\mu_i)+\Phi(-3-\mu_i).$$

Messversatz δ: μ₁ = δ/s, μ₂ = (1 − φ)·δ/s; Verschiebung des Prozessmittels: μ₁ = μ₂ = (1 − φ)·δ/s.

**M/M/1-Wartezeit** (μ = 1, λ = ρ; Lindley-Kette):

$$W_{n+1} = \max(0,\; W_n + S_n - A_n),\qquad \mathrm{E}[W]=\frac{\rho}{1-\rho},\quad \mathrm{Var}[W] = \frac{\rho(2-\rho)}{(1-\rho)^2}.$$

Lag-1-Autokorrelation (geschlossen):

$$\mathrm{E}[W_n W_{n+1}] = \frac{2\rho}{(1-\rho)^2} - 1 + \frac{1-\rho}{1+\rho},\qquad r_1 = \frac{\mathrm{E}[W_n W_{n+1}]-\mathrm{E}[W]^2}{\mathrm{Var}[W]}.$$

Shewhart nach oben bei c = μ + 3σ: die Integralgleichung für g(w) (weitere Beobachtungen bis zum Alarm aus der Wartezeit w ≤ c) geht durch zweimaliges Ableiten über in

$$g''(w) - (1-\rho)\,g'(w) = -\rho\quad\Longrightarrow\quad g(w) = C_0 + C_1\,e^{(1-\rho)w} + \frac{\rho\,w}{1-\rho},$$

die Konstanten folgen aus der Integralgleichung bei w = 0 und w = c. Die ARL bei stationärem Start ist

$$\mathrm{ARL} = 1 + (1-\rho)\,g(0) + \int_0^{c}\rho(1-\rho)\,e^{-(1-\rho)w}\,g(w)\,dw.$$

**Vorzeichen-CUSUM.** Schritt +1 bei einem Wert über dem Median (Wahrscheinlichkeit ½ bei stetiger Verteilung), −2 sonst, C' = max(0, C + Schritt), Alarm bei C ≥ H: die ARL₀ ist die Lösung eines linearen Gleichungssystems mit Brüchen (H = 3: 14).

**Varianz einer Summe** von m AR(1)-Werten:

$$\mathrm{Var}\Big(\sum_{t=1}^{m} u_t\Big) = m + 2\sum_{j=1}^{m-1}(m-j)\,\varphi^{j}\ \longrightarrow\ m\,\frac{1+\varphi}{1-\varphi}\quad\text{je Wert für } m\to\infty.$$

Alle Formeln gelten für bekannte Parameter, die genannten Karten und Modelle, zero-state. Dieser Block ist ein Modell, aber **keine** vollständige Beschreibung jeder Rechnung: die Randbehandlung der Zellen (Atom bei 0, offene Randzellen), die Richardson-Extrapolation und die Simulation stehen in README und Code.
"""
