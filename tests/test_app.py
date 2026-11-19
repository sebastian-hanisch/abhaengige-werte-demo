"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Erweiterungen dieser Demo (der Anfang der Datei bleibt die Vorlage aus tools/templates/test_app_smoke.py) -----------------------------------
import pytest  # noqa: E402

import abw_constants as C  # noqa: E402
import abw_presets as P  # noqa: E402


def _texte(at) -> str:
    return " ".join([w.value for w in at.success] + [w.value for w in at.info] + [w.value for w in at.warning])


def test_seite_hat_vier_experimente_den_tab_und_den_titel():
    at = _fresh_app()
    assert at.title
    namen = [s.value for s in at.subheader]
    assert [n for n in namen if n.startswith("🔬 Experiment")] == [n for n in namen if "Experiment" in n and n.startswith("🔬")] and len([n for n in namen if n.startswith("🔬")]) == 5
    assert "🚧 Wo die Annahmen enden" in namen


def test_standard_zeigt_alle_verdikte_und_die_diagramme():
    at = _fresh_app()
    text = _texte(at)
    for stichwort in ("Shewhart (zweiseitig, ±3σ) ARL₀ 419,4", "Klumpenlänge", "Residuenkarte gegen die auf ARL₀ 370,4 nachgeeichte", "M/M/1, ρ = 0,5", "Gate (M/M/2/6", "Normal: Shewhart"):
        assert stichwort in text, stichwort
    assert len(at.get("plotly_chart")) == 7         # ARL₀ über φ, Lauflänge, Reihe, Schemata, Lag, Wartezeit-Balken, Schiefe


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_und_setzt_alle_regler(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key, state_key in P.PRESET_KEYS.items():
        soll = C.PRESETS[name].get(key, P.SETTING_SPECS[state_key].default)
        ist = at.session_state[state_key]
        assert abs(ist - soll) < 1e-9 if isinstance(soll, float) else ist == soll, (name, state_key)


def test_zufalls_button_aendert_den_seed():
    at = _fresh_app()
    alt = at.session_state["seed_input"]
    at.sidebar.button[-1].click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["seed_input"] != alt


def test_jede_option_jeder_auswahl_laeuft():
    """Jede Option jeder Auswahl (select_slider, radio, selectbox) einzeln; die Werte stammen aus SETTING_SPECS."""
    gefunden = 0
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.options is None:
            continue
        for option in spec.options:
            at = _fresh_app()
            for art in ("select_slider", "radio", "selectbox"):
                try:
                    w = getattr(at.sidebar, art)(key=state_key)
                    break
                except KeyError:
                    continue
            w.set_value(option).run(timeout=TIMEOUT)
            gefunden += 1
            assert not at.exception, (state_key, option, [str(e) for e in at.exception])
            assert at.session_state[state_key] == option
    assert gefunden == len(C.PHI_OPTIONS) + len(C.TOY_OPTIONS) + len(C.PHI3_OPTIONS) + len(C.DELTA_OPTIONS) + len(C.ARTEN) + len(C.RHO_OPTIONS) + len(C.GATE_R_OPTIONS) + len(C.DIST_KEYS) + len(C.SIGN_H_OPTIONS)


def test_gate_knopf_zeigt_das_live_ergebnis():
    at = _fresh_app()
    assert not any("Live (" in i.value for i in at.info)
    at.button(key="gate_button").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(i.value.startswith("Live (50 Läufe") and "Shewhart 3σ" in i.value for i in at.info)


def test_permalink_rastet_ein_und_uebernimmt_werte():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["phi"] = "0.93"       # liegt zwischen 0,9 und 0,95: rastet auf 0,95 ein
    at.query_params["dist"] = "expon"
    at.query_params["rho"] = "0.85"       # rastet auf 0,8 oder 0,9
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["phi_select"] == 0.95 and at.session_state["dist_select"] == "expon" and at.session_state["rho_select"] in (0.8, 0.9)
