"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Experiment 1 und 2: abhängige Werte (AR(1)) ---------------------------------------------------------------------------------------------
PHI_OPTIONS = (-0.6, -0.3, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)   # Gitter der gerechneten (gefrorenen) AR-Koeffizienten
DEFAULT_PHI = 0.6
PHI_FOCUS = (0.0, 0.3, 0.6, 0.9)                     # die Werte der Shewhart-Tabelle
R_MIN, R_MAX, R_STEP, DEFAULT_R = 100, 500, 100, 200  # Zahl der Live-Läufe (R · tmax <= 5 · 10^6)
LEN_MIN, LEN_MAX, LEN_STEP, DEFAULT_LEN = 500, 3000, 500, 3000   # Länge der gezeigten Reihe (Alarm-Klumpen)
TOY_OPTIONS, DEFAULT_TOY = (0.0, 0.5, 0.8), 0.5      # φ der Zwei-Zustände-Kette (Brüche 0, 1/2, 4/5)

# --- Experiment 3: Residuenkarte --------------------------------------------------------------------------------------------------------------
PHI3_OPTIONS, DEFAULT_PHI3 = (0.3, 0.6, 0.9), 0.9
DELTA_OPTIONS, DEFAULT_DELTA = (0.5, 1.0, 2.0), 1.0
ART_OFFSET, ART_PROCESS = "Messversatz", "Verschiebung des Prozessmittels"
ARTEN = (ART_OFFSET, ART_PROCESS)
ART_KEY = {ART_OFFSET: "offset", ART_PROCESS: "process"}
DEFAULT_ART = ART_OFFSET

# --- Experiment 4: Wartezeit am Gate ----------------------------------------------------------------------------------------------------------
RHO_OPTIONS, DEFAULT_RHO = (0.5, 0.8, 0.9), 0.5
GATE_R_OPTIONS, DEFAULT_GATE_R = (20, 50, 100), 50
GATE_LAMBDA = 1.6                                    # Angebot am Gate (zwei Spuren, 6 Plätze insgesamt, Abfertigung 1)

# --- Experiment 5: Schiefe ---------------------------------------------------------------------------------------------------------------------
DIST_KEYS = ("normal", "expon", "gamma4", "lognorm", "t5")
DIST_LABELS = {"normal": "Normal", "expon": "Exponential", "gamma4": "Gamma(4)", "lognorm": "Lognormal(0; 0,5)", "t5": "t(5)"}
DEFAULT_DIST = "normal"
SIGN_H_OPTIONS, DEFAULT_SIGN_H = (3, 8, 9), 9

SEED_MAX, DEFAULT_SEED = 999999, 5

GAMMA0 = 370.398                                     # 1 / (2 Φ(-3)): ARL₀-Ziel (nicht das gesetzte 370)
H_ONE, H_TWO, K_CUSUM = 4.0965, 4.77, 0.5            # einseitiger CUSUM (ARL₀ = 370,398 bei φ = 0), zweiseitig gerundet wie in V2 (ARL₀ = 368,6)
C_ONE = 2.7822                                       # einseitige Shewhart-Grenze (ARL₀ = 370,398 bei φ = 0)

COLOR_SHEWHART = "#c0392b"
COLOR_SHEWHART1 = "#e59866"
COLOR_CUSUM = "#2F6B65"
COLOR_RESID = "#4c78a8"
COLOR_NOMINAL = "rgba(90,90,90,0.9)"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Mittel (φ = 0,6)": {},
    "Unabhängig (φ = 0)": {"phi": 0.0},
    "Schwache Abhängigkeit (φ = 0,1)": {"phi": 0.1},
    "Stark (φ = 0,9)": {"phi": 0.9},
    "Abwechselnd (φ = −0,3)": {"phi": -0.3},
    "Zwei Spitzen (Spielzeug)": {"toy": 0.8},
    "Residuenkarte, Prozessmittel δ = 1 (φ = 0,9)": {"art": ART_PROCESS},
    "Gate-Wartezeit (M/M/1, ρ = 0,8)": {"rho": 0.8},
    "Schiefe: Exponential": {"dist": "expon"},
}
PRESET_HELP = {
    "Mittel (φ = 0,6)": "Voreinstellung: AR-Koeffizient φ = 0,6, Experiment 3 mit φ = 0,9 und Messversatz, Experiment 4 mit M/M/1 bei ρ = 0,5, Experiment 5 mit normalen Werten.",
    "Unabhängig (φ = 0)": "Dieselben Karten bei unabhängigen Werten: hier gilt die Zahl 370,4, wie in Stück 1.",
    "Schwache Abhängigkeit (φ = 0,1)": "Schon φ = 0,1 senkt die ARL₀ des CUSUM um 38 %, die der Shewhart-Karte steigt kaum.",
    "Stark (φ = 0,9)": "Starke Abhängigkeit: die Shewhart-Karte wartet mehr als doppelt so lange, der CUSUM meldet nach rund einem Sechstel der Zeit.",
    "Abwechselnd (φ = −0,3)": "Negative Abhängigkeit (Werte wechseln sich ab): der CUSUM braucht mehr als das Zehnfache.",
    "Zwei Spitzen (Spielzeug)": "Das Spielzeug aus Experiment 2 mit φ = 4/5: dieselbe Häufigkeit von Spitzen, Regel A wartet länger, Regel B kürzer.",
    "Residuenkarte, Prozessmittel δ = 1 (φ = 0,9)": "Experiment 3 mit Verschiebung des Prozessmittels um 1 σ bei φ = 0,9: die Residuenkarte hält ihre ARL₀, braucht aber deutlich länger.",
    "Gate-Wartezeit (M/M/1, ρ = 0,8)": "Experiment 4 mit M/M/1 bei Auslastung 0,8: Lag 1 = 0,963, die Shewhart-Karte braucht mehr als das Zwanzigfache der iid-Sicht.",
    "Schiefe: Exponential": "Experiment 5 mit Exponentialverteilung: die Shewhart-Karte meldet nach 54,6 statt 370,4 Beobachtungen.",
}
