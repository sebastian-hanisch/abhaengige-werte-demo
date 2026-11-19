"""Je Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a Punkt 7): ein Preset ohne Effekt darf keinen Effekt melden, und die Anker des Plans stehen im Text der App.

Die Zeilen stammen aus der laufenden App (AppTest), nicht aus den reinen Funktionen: so ist geprüft, dass die App die Zahlen wirklich einsetzt."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import abw_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))

ERWARTET = {
    "Mittel (φ = 0,6)": ["φ = 0,6: Shewhart (zweiseitig, ±3σ) ARL₀ 419,4 (1,13fach)", "CUSUM (einseitig, k = 0,5, h = 4,0965) 62,2 (0,17fach)", "φ = 0,9, Messversatz δ = 1,0",
                         "M/M/1, ρ = 0,5: Lag 1 = 0,778", "Normal: Shewhart 3σ zweiseitig 370,4"],
    "Unabhängig (φ = 0)": ["φ = 0: Shewhart (zweiseitig, ±3σ) ARL₀ 370,4 (1,00fach)", "hält die Karte, was die Zahl verspricht", "gleich der Alarmrate je Beobachtung"],
    "Schwache Abhängigkeit (φ = 0,1)": ["φ = 0,1: Shewhart (zweiseitig, ±3σ) ARL₀ 370,9 (1,00fach)", "CUSUM (einseitig, k = 0,5, h = 4,0965) 229,8 (0,62fach)", "ist hier klein"],
    "Stark (φ = 0,9)": ["φ = 0,9: Shewhart (zweiseitig, ±3σ) ARL₀ 831,8 (2,25fach)", "63,4 (0,17fach)", "das erklärt 0,66 des Anstiegs"],
    "Abwechselnd (φ = −0,3)": ["φ = −0,3: Shewhart (zweiseitig, ±3σ) ARL₀ 376,4 (1,02fach)", "3 798,2 (10,25fach)", "symmetrisch in φ"],
    "Zwei Spitzen (Spielzeug)": ["φ = 4/5, Spitzenhäufigkeit 1/10: Regel A (Alarm bei der ersten Spitze) ARL 46 (bei φ = 0: 10)", "2386/41 = 58,20 (bei φ = 0: 110)"],
    "Residuenkarte, Prozessmittel δ = 1 (φ = 0,9)": ["φ = 0,9, Verschiebung des Prozessmittels δ = 1,0", "Shewhart (zweiseitig) 292,6 gegen 100,3", "CUSUM (einseitig) 91,3 gegen 58,7", "ist die Residuenkarte langsamer (beide Karten)"],
    "Gate-Wartezeit (M/M/1, ρ = 0,8)": ["M/M/1, ρ = 0,8: Lag 1 = 0,963", "1 147,9 statt 52,6 (iid-Sicht)", "150,1 statt 101,4"],
    "Schiefe: Exponential": ["Exponential: Shewhart 3σ zweiseitig 54,60 (0,15fach von 370,4)", "CUSUM einseitig 104,95", "nach genau 9 Beobachtungen", "keine Stärke der Karte"],
}


def _texte(at) -> str:
    return " ".join([w.value for w in at.success] + [w.value for w in at.info] + [w.value for w in at.warning])


def test_jedes_preset_hat_erwartete_zeilen():
    assert set(ERWARTET) == set(C.PRESETS)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_zeigt_seine_verdikt_zeilen(name):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    text = _texte(at)
    for teil in ERWARTET[name]:
        assert teil in text, (name, teil)


def test_ein_preset_ohne_effekt_meldet_keine_abhaengigkeit():
    """φ = 0: die App spricht von unabhängigen Werten, nicht von einem Effekt (Warnung nur bei φ ≠ 0 in Experiment 1)."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.button(key="preset_Unabhängig (φ = 0)").click().run(timeout=TIMEOUT)
    assert not any(w.value.startswith("φ = 0: Shewhart") for w in at.warning)
    assert any(i.value.startswith("φ = 0: Shewhart") for i in at.info)
