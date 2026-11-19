"""Presets: Vollständigkeit, gültige Werte, das Standard-Preset ist die Voreinstellung, jedes andere Preset unterscheidet sich von ihr und von jedem anderen; Permalink rastet ein."""

import pytest

import abw_constants as C
import abw_evaluation as ev
import abw_presets as P


def test_jedes_preset_hat_hilfetext_und_nur_bekannte_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name].strip()


def test_preset_werte_liegen_in_den_reglergrenzen_und_optionen():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_standard_preset_ist_die_voreinstellung_und_alle_anderen_unterscheiden_sich():
    erstes = next(iter(C.PRESETS))
    assert C.PRESETS[erstes] == {}
    def werte(p):
        return tuple(p.get(k, P.SETTING_SPECS[sk].default) for k, sk in P.PRESET_KEYS.items())
    alle = [werte(p) for p in C.PRESETS.values()]
    assert len(set(alle)) == len(alle)                               # keine zwei Presets gleich: die Hervorhebung ist eindeutig
    assert ev.Settings() == ev.Settings(**{k: v for k, v in C.PRESETS[erstes].items()})


def test_setting_specs_decken_jeden_preset_schluessel_und_den_zufalls_seed():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) and "seed_input" in P.SETTING_SPECS


def test_phi_gitter_enthaelt_die_gemessenen_werte_und_ist_sortiert():
    assert list(C.PHI_OPTIONS) == sorted(C.PHI_OPTIONS) and set(C.PHI_FOCUS) <= set(C.PHI_OPTIONS) and set(C.PHI3_OPTIONS) <= set(C.PHI_OPTIONS)
    assert C.DEFAULT_PHI in C.PHI_OPTIONS and C.DEFAULT_PHI3 in C.PHI3_OPTIONS and C.DEFAULT_RHO in C.RHO_OPTIONS and C.DEFAULT_SIGN_H in C.SIGN_H_OPTIONS


@pytest.mark.parametrize("wert,soll", [(0.93, 0.95), (0.62, 0.6), (-0.9, -0.6), (5.0, 0.98), (0.03, 0.05)])
def test_snap_rastet_auf_das_naechste_gitter_ein(wert, soll):
    spec = P.SETTING_SPECS["phi_select"]
    assert P._snap(wert, spec) == soll


def test_snap_text_und_zahlen_mit_raster():
    assert P._snap("expon", P.SETTING_SPECS["dist_select"]) == "expon" and P._snap("unbekannt", P.SETTING_SPECS["dist_select"]) is None
    assert P._snap(730, P.SETTING_SPECS["r_slider"]) == 500 and P._snap(20, P.SETTING_SPECS["r_slider"]) == 100
    assert P._snap(-3, P.SETTING_SPECS["seed_input"]) == 0
