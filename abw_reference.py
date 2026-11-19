"""Gefrorene Tabellen (abw_tables.json, erzeugt von tools/abw_make_tables.py): schwere Rechnungen (zweidimensionale Ketten auf feinem Gitter) und alle Simulationen,
die die App nicht live zeigt. Jede Simulationszelle trägt Zufalls-Seed und R; der Kopf (`meta`) nennt Quelle, Stand und Gitter."""

import json
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parent / "abw_tables.json"


@lru_cache(maxsize=1)
def tables() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def meta() -> dict:
    return tables()["meta"]


def pk(phi: float) -> str:
    return f"{phi:.2f}"


def ar1(phi: float) -> dict:
    """Zeile der AR(1)-Tabelle: ARL₀ der Karten (Shewhart zweiseitig und einseitig, CUSUM einseitig), Lauflängenverteilung, Alarm-Klumpen."""
    return tables()["ar1"][pk(phi)]


def twosided(phi: float):
    return tables()["zweiseitig"].get(pk(phi))


def calibration(phi: float):
    return tables()["nacheichung"].get(pk(phi))


def resid_cell(art_key: str, phi: float, delta: float):
    """CUSUM-ARL₁ auf den Rohdaten (nominell, nachgeeicht) für Messversatz (offset) bzw. Verschiebung des Prozessmittels (process)."""
    return tables()["residuen_cusum"][f"{art_key}|{pk(phi)}|{delta:.1f}"]


def mm1(rho: float) -> dict:
    return tables()["mm1"][f"{rho:.1f}"]


def gate() -> dict:
    return tables()["gate"]


def skew_calibration(key: str):
    return tables()["schiefe"].get(key)


def segmentation(phi: float, residuals: bool) -> dict:
    return tables()["segmentierung"][f"{pk(phi)}|{'resid' if residuals else 'roh'}"]
