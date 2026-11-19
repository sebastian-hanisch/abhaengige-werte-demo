# Wenn die Werte nicht unabhängig sind: dieselbe Zahl, das Gegenteil – Streamlit-Demo

<!-- Nach dem Deploy direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-abhaengige-werte-demo.streamlit.app/)** -->

Eine Demo-Seite der Reihe **Änderungserkennung** („Wie schnell merkt man, dass sich etwas geändert hat?“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Stück 5 der Reihe; Vorgänger ist Stück 4 (`anfangsparameter-demo`, Parameter schätzen), Nachfolger Stück 6 (viele Kanäle). Die Kante ist ein `fix`: die Stücke 1 bis 4 nehmen an, **aufeinanderfolgende Beobachtungen sind unabhängig** (und normal); dieses Stück hebt die Unabhängigkeit auf und misst, was die Zahl „ARL₀ = 370,4“ dann noch verspricht. Alle Parameter sind **bekannt** (wie in Stück 1 bis 3), damit nur die Abhängigkeit wirkt; die Frage, wie sich ein **geschätztes** φ̂ in Phase I auswirkt, beantwortet Stück 4b (`anfangsparameter-abhaengig-demo`), hier nur der Verweis.

Vehikel: **ausgedachte** Reihen. Ein AR(1)-Prozess mit normaler Randverteilung und die Wartezeit eines Lkw-Tors aus der Kette der Demo „Warteschlangen als Markov-Ketten“ (`markov-queue-demo`), dazu M/M/1 als exakter Fall. In den gelesenen Quellen gibt es keine reale Reihe eines Terminals oder Lagers; wie sich die Karten dort verhalten, wird hier nicht behauptet. Gezählt werden Beobachtungen bis zum Alarm bei bekannten Parametern; die Parameter sind in dieser Demo nicht unbekannt.

## Warum dieses Problem

Die Zahl ARL₀ = 370,4 („im Mittel 370,4 Beobachtungen bis zum Fehlalarm“) gehört zur Karte „Alarm bei mehr als 3 Streuungen“ **bei unabhängigen Werten**. Aufeinanderfolgende Wartezeiten an einer Warteschlange sind es nicht: Lag 1 = 0,78 bis 0,99 bei Auslastung 0,5 bis 0,9 (M/M/1, exakt). **Kernfrage:** Was verspricht die Zahl, wenn die Werte abhängig sind, warum steigt sie für die eine Karte und fällt für die andere, was schützt die Residuenkarte und was kostet sie, und was geschieht an einer echten Warteschlange? Die Antwort ist je nach Karte das **Gegenteil**: bei φ = 0,6 meldet die Shewhart-Karte (zweiseitig) im Mittel erst nach 419,4 Beobachtungen, der einseitige CUSUM schon nach 62,2. Die Falle ist, eine ARL₀ für unabhängige Werte als Eigenschaft der Karte zu lesen.

## Befunde und Korrekturen gegenüber dem Plan

Die Zahlen des Plans (Vormessung V14 mit numpy-Zufall, V2) sind **Erwartungen**. Hier ist alles Exakte mit anderem Code neu gerechnet (zweidimensionale Kette dünn besetzt gelöst, schneller als in der Vormessung) und jede Simulation mit SplitMix64 und eigenen Zufalls-Seeds (ab 51000, in `abw_tables.json` je Zelle genannt) **neu gemessen** (`tools/abw_make_tables.py --lang`). Abweichungen:

| Größe | Plan (Erwartung) | hier gemessen | Einordnung |
|---|---|---|---|
| Nachgeeichte CUSUM-Schwelle h_φ bei φ = 0,3 / 0,6 / 0,9 | 6,114 / 9,971 / 22,277 (grobe Kette, V14) | 6,114 / 9,969 / 22,099 | **verschoben bei φ = 0,9:** die grobe Kette gibt bei h = 22,277 die ARL₀ 370,40, das feine Gitter 375,15 (1,3 % zu hoch); die Simulation (100 000 Läufe) mit der neuen Schwelle gibt 370,8 ± 1,1. Die grobe Kette reproduziert den Plan-Wert (h_grob = 22,277) |
| Nachgeeichte Shewhart-Grenze L_φ bei φ = 0,3 / 0,6 / 0,9 | 2,9951 / 2,9605 / 2,7113 | 2,9951 / 2,9605 / 2,7112 | mit Richardson statt der einfachen Kette (n = 480); Abweichung höchstens 1·10⁻⁴ |
| CUSUM „Roh nachgeeicht“, φ = 0,9: Messversatz δ = 0,5 / 1 / 2; Prozessmittel δ = 0,5 / 1 / 2 | 114,29 / 50,69 / 18,53; 119,91 / 58,81 / 27,81 | 114,21 / 50,55 / 18,42; 119,86 / 58,71 / 27,72 | Folge der neuen Schwelle h_φ bei φ = 0,9: um höchstens 0,7 % kleiner; die Zellen bei φ = 0,3 und 0,6 stimmen mit dem Plan auf 0,1 % überein |
| Zweiseitiger CUSUM, Simulation bei φ = 0 / 0,3 / 0,6 / 0,9 (h = 4,77) | 368,3 ± 2,6 / 84,1 ± 0,6 / 37,6 ± 0,2 / 27,2 ± 0,2 (V14, numpy-Zufall) | 368,3 ± 2,6 / 85,4 ± 0,6 / 37,8 ± 0,2 / 27,5 ± 0,2 | im Band (4 Standardfehler beider Seiten); die Näherung „einseitig/2“ liegt bei φ = 0,3 nur 0,1 % über der Simulation (Plan: 1,6 %), bei φ = 0,6 4,0 % (4,6 %), bei φ = 0,9 28,6 % (30 %) |
| Gate, Shewhart 3σ (Simulation) | 128,6 ± 2,4 (V2); 131,5 ± 0,8 (Prüfung, 30 000 Läufe) | 135,4 ± 2,1 (2 × 2000 Läufe) | höher als die V2-Zahl, aber im Band von 4 Standardfehlern beider Seiten; die Prüfung mit 30 000 Läufen bleibt die genaueste Einzelmessung |
| Gate, CUSUM zweiseitig (h = 4,77) und dessen iid-Sicht | 42,28 ± 0,38 (Plan 08, 10 000 Läufe); 177,7 (V2) | 43,0 ± 0,6; 176,4 ± 1,7 | im Band; die Zahl führt Plan 08 (Stück 8), hier nur zum Vergleich |
| Fehlbrüche (Anteil der Reihen, c = 2), Rohdaten bei φ = 0 / 0,3 / 0,6 / 0,9 | 3,0 / 36,5 / 87,0 / 99,5 % | 4,2 / 34,0 / 88,2 / 98,8 % | im Band (Standardfehler bis 2,5 Prozentpunkte, zwei Stichproben zu je 400 Reihen); Residuen: 4,0 / 4,5 / 2,0 / 5,5 % (Plan 3,0 / 3,8 / 3,2 / 4,5) |

Weitere Abweichungen ohne Zahl in der Tabelle:

* **Nacheichung bei großer Schwelle:** der Plan (Z5 d) ließ offen, wie genau die nachgeeichten CUSUM-Werte bei h = 22 sind (rund 2 %). Gemessen: bei φ = 0,3 und 0,6 liegen grobes und feines Gitter innerhalb von 0,1 %; bei φ = 0,9 verfehlt das grobe Gitter die ARL₀ um etwa 1 % (siehe oben), das feine Gitter (n = 192, m = 32 und 64) ist noch nicht vollständig ausgerechnet (Messung beim Bau, nicht im Test: bei h = 22,277 gab m = 32 und 64 den Wert 375,15, das noch feinere m = 64 und 128 den Wert 375,41; Rest 0,1 bis 0,3 %).
* **Eigenschaft „ARL₀ fällt in k“ (Plan, Orakel und Eigenschaften):** gilt nicht; ein größerer Referenzwert k lässt die Summe stärker zurücklaufen, die ARL₀ **steigt** in k (getestet an zufälligen Instanzen).
* **Vorzeichen-CUSUM bei normalen Werten:** der Plan nannte 15,45 gegen 8,58 (δ = 1 σ) als „aus V2, nicht neu gemessen“; hier sind beide Zahlen exakt gerechnet (Kette bzw. Gleichungssystem) und stimmen mit V2 überein.
* **Rechenzeit:** die dünn besetzte Lösung macht die zweidimensionale Kette auf dem Gitter der Tabellen in 0,1 bis 0,9 s möglich (Plan: 0,3 bis 4 s dicht); trotzdem bleiben die CUSUM-Zellen von Experiment 3, die Nacheichung, die M/M/1-Ketten und alle Simulationen mit Zufallsanteil gefroren, damit die Seite schnell antwortet und die Herkunft (Zufalls-Seed, R, Stand) sichtbar bleibt.
* **Regler:** φ steht als Auswahl auf dem Gitter der gerechneten Werte (15 Werte), nicht als freier Regler; zum Plan (Z1) gehört der Port 10047 (Auftrag), nicht 10035.
* **Grenzfall:** bei riesigen ARL (h sehr groß) verliert die Richardson-Rechnung auf dem groben Gitter ihre Genauigkeit und gab zunächst eine unmögliche negative Zahl; `abw_cusum.ar1_arl` meldet das jetzt als „nicht gerechnet“ (NaN, `tests/test_edge_cases.py`).

## Modell

AR(1): u_t = φ·u_(t−1) + s·ε_t mit s = √(1 − φ²) und unabhängigen normalen ε_t; die Randverteilung ist für jedes φ N(0, 1) (Mittel 0, Streuung 1, **bekannt**), der Start vor der ersten Beobachtung ist stationär. φ heißt hier **AR-Koeffizient**; die Korrelation zwischen Wert t und Wert t + k ist φ^k. **ρ** ist in dieser Demo nur die Auslastung der Warteschlange, nie die Autokorrelation. **Lauflänge** T = Zahl der Beobachtungen bis einschließlich der alarmierenden (T ≥ 1); ein Simulationslauf heißt „Lauf“ und ist eine Reihe bis zum Alarm. Zähleinheiten getrennt: **Beobachtungen** (Lauflänge, Verzögerung), **Läufe** (Simulation), **Reihen** (Segmentierung).

**Konventionen.** ARL₀ = E[T] ohne Änderung, **zero-state** (CUSUM-Wert 0, Rauschen stationär); ein Anfangsvorsprung kommt nicht vor. **Verzögerung (ARL₁):** alle Zahlen gelten für **ν = 0** (Änderung vor der ersten Beobachtung, Rauschen vorher stationär, CUSUM bei 0); das ist weder das Lorden- noch das Pollak- noch das Shiryaev-Maß, und das Maß steht in jedem Satz der App. Zwei Arten der Änderung (beide nur für AR(1)): **Messversatz δ** (der Sensor liefert ab t = 1 u_t + δ, die Dynamik bleibt) und **Verschiebung des Prozessmittels** um δ (x_t = φ·x_(t−1) + (1 − φ)·δ + s·ε_t; die Reihe läuft allmählich auf das neue Mittel); δ in marginalen Standardabweichungen. Karten: **Shewhart zweiseitig** (Alarm bei |x| > 3), **Shewhart einseitig** (x > 2,7822, ARL₀ 370,398 bei φ = 0), **CUSUM einseitig** (S_t = max(0, S_(t−1) + x_t − k), k = 0,5, Alarm bei S_t ≥ h = 4,0965, ARL₀ auf vier Stellen 370,4 bei φ = 0); der **zweiseitige** CUSUM hat für dasselbe Ziel h = 4,7749, die Demo rechnet ihn mit dem gerundeten h = 4,77 (ARL₀ 368,6 nach der Näherung „einseitig/2“, Simulation 368,3 ± 2,6): **nicht** dieselbe Karte. Ziel ist ARL₀ = 1/(2Φ(−3)) = 370,398, nicht das gesetzte 370. **Residuenkarte** = dieselbe Karte auf e_t = (x_t − φ·x_(t−1))/s mit bekanntem φ; die Karte kennt den Wert x₀ vor dem ersten Beobachtungsintervall (vom Messversatz nicht betroffen), e₁ hat bei Messversatz δ das Mittel δ/s, ab e₂ das Mittel (1 − φ)·δ/s. **Nachgeeicht** heißt: Grenze L_φ bzw. Schwelle h_φ so gesetzt, dass die ARL₀ der Rohdatenkarte wieder 370,398 ist (setzt die Normal-AR(1)-Annahme als bekannt voraus). Alle Schwellen in σ-Einheiten der Beobachtung. Formeln: 📐-Expander der App.

## Methodik

* **Shewhart auf AR(1), exakt:** Zellenkette (Zustand = letzter Wert, Zellen genau zwischen den Grenzen, n = 480 und 960, Richardson (4A(2n) − A(n))/3) **und** Integralgleichung (Gauß-Legendre-Nyström); beide stimmen auf 5 · 10⁻⁴ überein. Die Lauflängenverteilung kommt aus der Kette (n = 960), die Alarm-Klumpen (q) aus einer Quadratur mit Einschluss-Ausschluss aus der Kette als zweitem Weg.
* **CUSUM auf AR(1), numerisch:** zweidimensionale Kette (Zustand = CUSUM-Zelle und letzter Wert; Zellen nach Brook und Evans, die Masse des nächsten Wertes wird exakt auf die Schnittmengen verteilt), dünn besetzt gelöst, Richardson über m = 16 und 32 CUSUM-Zellen bei 128 latenten Zellen (Gitterfehler ∝ 1/m², rund 0,2 %); bei φ = 0 gleich der eindimensionalen Integralgleichung auf 0,1 %. Bei großer Schwelle (h über 10) ist das Gitter zu grob; dort gilt ein feineres Gitter (n = 192, m = 32 und 64).
* **Residuenkarte:** Formel (Shewhart) und Integralgleichung mit erstem Schritt (CUSUM); Nacheichung per Wurzelsuche (Shewhart: Zellenkette, zweiter Weg Integralgleichung; CUSUM: Kette auf dem feinen Gitter, geprüft durch Simulation).
* **M/M/1:** Lag 1 geschlossen (und als Bruch), Lag 1 bis 5 aus einer Zellenkette; Shewhart nach oben bei μ + 3σ als **geschlossene Lösung** der Integralgleichung (zweimaliges Ableiten); CUSUM iid-Sicht (eindimensionale Kette) gegen wirkliche Reihe (zweidimensionale Kette).
* **Schiefe:** Shewhart aus der Verteilungsfunktion (Exponential e⁴, Gamma(4) e¹⁰·3/683 geschlossen), CUSUM-Kette mit Richardson, Vorzeichen-CUSUM exakt als Bruch.
* **Zufall:** nur SplitMix64 (`abw_rng.py`, die Vorlage unverändert; `abw_normal.py` ist die vektorisierte Stapel-Fassung, bitgleich mit der Klasse). Alle Simulationen laufen vektorisiert über Läufe und Zeit (AR(1) per `lfilter`, CUSUM und Lindley-Kette per Spiegelungsformel); `tests/` vergleicht beides mit der Schleife. Läufe ohne Alarm bis tmax heißen zensiert und werden gezählt.
* **Live gegen gefroren:** live gerechnet werden die Shewhart-Karten, die Residuenkarten (Shewhart, CUSUM-Integralgleichung), die Wartezeit-Rechnungen, die Schiefe und alle Simulationen mit Zeitangabe; gefroren (`abw_tables.json`, `tools/abw_make_tables.py --lang`, Stand 2026-11-19) sind CUSUM-Ketten auf dem feinen Gitter, die Nacheichung, die CUSUM-Zellen von Experiment 3, die M/M/1-Ketten, das Gate und die Segmentierung. Live-Läufe: R · tmax höchstens 5 Mio. Zufallszahlen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für: AR(1) mit normaler Randverteilung bzw. die genannte Wartezeit oder Verteilung, **bekannte Parameter**, zero-state, Zähleinheit Beobachtungen bis zum Alarm (ν = 0 bei den Verzögerungen), einseitig oder zweiseitig genannt, ARL₀-Ziel 370,398; ausgedachte Reihen. Die Spalte „Test“ nennt den Test, der die Zahl belegt (Band oder exakt); `test_claims.py` sucht jede Zahl dieser Tabelle neu erzeugt in dieser README.

| Frage | Befund | Test (Datei, Funktion) |
|---|---|---|
| Shewhart zweiseitig auf AR(1), ARL₀ bei φ = 0 / 0,3 / 0,6 / 0,9 (Beobachtungen, exakt, zwei Rechenwege) | 370,398 / 376,383 / 419,377 / 831,783 | `test_oracle_shewhart.py::test_zellenkette_gleich_integralgleichung_auf_5e_4`, `test_claims.py::test_shewhart_zahlen` |
| Symmetrie: φ = −0,6 und −0,9 (zweiseitig); einseitig (Alarm bei x > 2,7822) bei φ = 0 / 0,3 / 0,6 / 0,9 und bei −0,3 / −0,6 / −0,9 | 419,377 und 831,783; einseitig 370,398 / 380,701 / 436,597 / 924,029 und 370,458 / 383,158 / 603,037 | `test_oracle_shewhart.py::test_symmetrie_in_phi_gleichheit_auf_1e_9_nur_zweiseitig`, `test_claims.py::test_shewhart_zahlen` |
| Lauflängenverteilung der zweiseitigen Shewhart-Karte bei φ = 0 / 0,3 / 0,6 / 0,9: P(T ≤ 100); Median; 5-%-Quantil; 95-%-Quantil (exakt, Kette n = 960) | 0,2369 / 0,2336 / 0,2126 / 0,1151; 257 / 261 / 291 / 576; 19 / 20 / 22 / 41; 1 109 / 1 127 / 1 256 / 2 494 | `test_oracle_shewhart.py::test_verteilung_tafelwerte_phi_09`, `test_claims.py::test_shewhart_zahlen` |
| Alarm-Klumpen bei φ = 0 / 0,3 / 0,6 / 0,9: q = P(zweiter Alarm in Folge ǀ Alarm); 1/(p(1 − q)) (Kehrwert der Rate der Klumpenanfänge); bei φ = 0,9: mittlere Klumpenlänge und Anteil des Anstiegs | q 0,0027 / 0,0177 / 0,1035 / 0,4522; 371,4 / 377,1 / 413,1 / 676,1; bei φ = 0,9: 1,83 und 0,66 | `test_oracle_shewhart.py::test_alarmpaar_quadratur_gleich_einschluss_ausschluss_der_kette`, `test_claims.py::test_klumpen` |
| CUSUM einseitig (k = 0,5, h = 4,0965) auf AR(1), ARL₀ bei φ = −0,3 / 0 / 0,1 / 0,3 / 0,6 / 0,8 / 0,9 / 0,95 / 0,98 (zweidimensionale Kette); φ = −0,6 | 3 798 / 370,4 / 229,8 / 114,2 / 62,2 / 55,0 / 63,4 / 83,8 / 137,7; φ = −0,6: 284 222 (nur Kette) | `test_oracle_cusum.py`, `test_claims.py::test_cusum_zahlen` |
| CUSUM, Verhältnis zu 370,4 bei φ = 0,1 / 0,3 / 0,6 / 0,8 / 0,9 / 0,98 und −0,3; P(T ≤ 100) bei φ = 0 / 0,3 / 0,6 / 0,9 | 0,62 / 0,31 / 0,17 / 0,15 / 0,17 / 0,37; −0,3: 10,3; P(T ≤ 100) 0,230 / 0,584 / 0,807 / 0,793 | `test_claims.py::test_cusum_zahlen`, `test_oracle_cusum.py::test_verteilung_der_kette_gegen_simulation_p100` |
| Der CUSUM-Wert ist nicht monoton in φ | fällt bis φ = 0,8 (55,0), danach steigend (63,4 / 83,8 / 137,7); schon φ = 0,1 senkt ihn um 38 % | `test_oracle_cusum.py::test_nicht_monoton_minimum_bei_0_8_im_gitter_und_anstieg_danach`, `test_claims.py::test_cusum_zahlen` |
| Zweiseitiger CUSUM, h = 4,77 (ARL₀ 368,6 bei φ = 0): einseitig exakt / Näherung „einseitig/2“ / Simulation (20 000 Läufe) bei φ = 0 / 0,3 / 0,6 / 0,9 | 737,2 / 170,9 / 78,6 / 70,7 / 368,6 / 85,4 / 39,3 / 35,4 / 368,3 ± 2,6 / 85,4 ± 0,6 / 37,8 ± 0,2 / 27,5 ± 0,2; Abweichung der Näherung +0,1 / +0,1 / +4,0 / +28,6 % | `test_tables.py::test_zweiseitig_naeherung_gegen_simulation_v14_und_neu_gemessen`, `test_oracle_cusum.py::test_zweiseitiger_cusum_ist_simulation_naeherung_gilt_nur_bis_phi_03` |
| Nacheichung bei φ = 0,3 / 0,6 / 0,9: Grenze L_φ (Shewhart, zweiseitig); Schwelle h_φ (CUSUM, feines Gitter); ARL₀ der Kette auf dem feinen Gitter; Simulation (100 000 Läufe) | L_φ 2,9951 / 2,9605 / 2,7112; h_φ 6,114 / 9,969 / 22,099; ARL₀ 370,40 / 370,40 / 370,40; 369,0 ± 1,1 / 371,2 ± 1,1 / 370,8 ± 1,1 (zensiert: 0 / 0 / 1 Läufe) | `test_oracle_residual.py::test_nacheichung_grenze_gegen_integralgleichung_und_tafel`, `test_oracle_residual.py::test_nacheichung_schwelle_gefroren_gegen_kette_und_simulation_aus_der_tabelle` |
| Residuenkarte: ARL₀ für jedes φ (bekanntes φ; zwei Wege: Zellenkette und Simulation der Reihe); H3: ARL₁ bei φ = 0,6, Messversatz δ = 1 mit x₀ / ohne x₀ | 370,398 für φ = 0,3 / 0,6 / 0,9; ARL₁ 150,00 / 152,69 | `test_oracle_residual.py::test_residuenkarte_arl0_ist_370_398_fuer_jedes_phi_zellenkette`, `test_handbeispiele.py::test_h3_residuenkarte_phi_06_messversatz_1` |
| Residuenkarte gegen nachgeeichte Rohkarte, ARL₁ (ν = 0): φ = 0,9, Messversatz δ = 1: Shewhart / CUSUM, je nominell / nachgeeicht / Residuum | Shewhart 153,00 / 88,59 / 223,31; CUSUM 14,73 / 50,55 / 79,60 | `test_claims.py::test_residuenkarte_zellen`, `test_oracle_residual.py::test_arl1_der_residuenkarte_formel_gegen_simulation` |
| dasselbe bei Verschiebung des Prozessmittels, φ = 0,9, δ = 1 | Shewhart 165,99 / 100,29 / 292,57; CUSUM 21,49 / 58,71 / 91,33 | `test_claims.py::test_residuenkarte_zellen` |
| Messversatz δ = 2, φ = 0,9: die Residuenkarte ist schneller | Shewhart 27,70 / 17,86 / 10,68; CUSUM 4,93 / 18,42 / 8,84 | `test_oracle_residual.py::test_messversatz_delta_2_phi_09_residuum_schneller_prozessmittel_langsamer`, `test_claims.py::test_residuenkarte_zellen` |
| Verschiebung des Prozessmittels δ = 2, φ = 0,9: die Residuenkarte bleibt langsam (Shewhart) | Shewhart 43,31 / 31,95 / 172,55 | `test_claims.py::test_residuenkarte_zellen` |
| φ = 0,6, Messversatz δ = 1: nominelle Rohkarte / nachgeeicht / Residuum | Shewhart 60,65 / 55,73 / 150,00; CUSUM 9,84 / 21,27 / 26,16 | `test_claims.py::test_residuenkarte_zellen` |
| M/M/1, Lag 1 bei ρ = 0,5 / 0,8 / 0,9 (exakt, Brüche); Lag 1 bis 5 bei ρ = 0,8 (Kette) | 7/9 / 26/27 / 207/209; ρ = 0,8: 0,9630 / 0,9291 / 0,8978 / 0,8685 / 0,8410 | `test_handbeispiele.py::test_h6_mm1_lag1_als_bruch_und_durch_quadratur`, `test_oracle_queue.py::test_lag_1_bis_5_aus_der_kette_gleich_lag1_geschlossen_und_v2_simulation` |
| M/M/1, Shewhart nach oben bei μ + 3σ, ρ = 0,5 / 0,8 / 0,9: iid-Sicht / wirkliche Reihe (geschlossene Lösung); P(T ≤ 100) iid-Sicht / wirklich | 44,3 / 52,6 / 54,1 / 159,8 / 1 147,9 / 4 669,4; 0,898 / 0,853 / 0,845 / 0,470 / 0,106 / 0,047 | `test_oracle_queue.py::test_shewhart_geschlossene_loesung_gleich_kette_und_tafel`, `test_oracle_queue.py::test_shewhart_p100_gegen_simulation_im_band` |
| M/M/1, CUSUM einseitig auf (W − μ)/σ, ρ = 0,5 / 0,8 / 0,9: iid-Sicht / wirkliche Reihe (zweidimensionale Kette); P(T ≤ 100) iid / wirklich; Median iid / wirklich | 87,3 / 101,4 / 104,0 / 70,2 / 150,1 / 359,9; 0,685 / 0,628 / 0,619 / 0,760 / 0,507 / 0,334; 61 / 71 / 73 / 48 / 98 / 222 | `test_oracle_queue.py::test_cusum_iid_sicht_und_wirkliche_reihe_gegen_tafel`, `test_oracle_queue.py::test_cusum_wirkliche_reihe_gegen_simulation_im_band` |
| Schiefe und Abhängigkeit addieren sich nicht (CUSUM, M/M/1) | Schiefe allein: 370,4 → 87 bis 104; mit Abhängigkeit 70,2 (ρ = 0,5, tiefer), 150,1 und 359,9 (höher); bei ρ = 0,9 ist P(T ≤ 100) 0,334, das 1,45fache des Nominalwerts 0,230 | `test_oracle_queue.py::test_schiefe_allein_senkt_abhaengigkeit_hebt_und_senkt_keine_feste_richtung`, `test_claims.py::test_mm1_zusatz` |
| Gate M/M/2/6 (Angebot 1,6; Zeiteinheit = mittlere Abfertigungsdauer, angenommene Lkw), Randverteilung exakt: Verlust / Mittel / Standardabweichung / Schiefe der Wartezeit; Shewhart 3σ iid-Sicht | 0,0760 / 0,659 / 0,898 / 1,738; 60,1 | `test_oracle_queue.py::test_gate_randverteilung_gegen_geburts_und_sterbeprozess_und_little`, `test_claims.py::test_gate` |
| Gate, Simulation (2 × 2000 Läufe, zwei Zufalls-Seeds): Shewhart 3σ; CUSUM zweiseitig (k = 0,5, h = 4,77); dessen iid-Sicht; Lag 1 bis 5 | 135,4 ± 2,1; 43,0 ± 0,6; 176,4 ± 1,7; 0,782 / 0,587 / 0,413 / 0,256 / 0,179 | `test_tables.py::test_gate_tafel_neu_gemessen_gegen_v2_und_plan_08_im_band`, `test_claims.py::test_gate` |
| Schiefe, Shewhart 3σ zweiseitig, ARL₀: Exponential (e⁴) / Gamma(4) (e¹⁰·3/683) / Lognormal(0; 0,5) / t(5) | 54,60 / 96,75 / 65,01 / 85,29 | `test_oracle_skew.py::test_exponential_e_hoch_4_und_gamma4_geschlossen_gegen_quadratur`, `test_claims.py::test_schiefe` |
| Schiefe, CUSUM einseitig, ARL₀ (dieselben vier Verteilungen) | 104,95 / 159,01 / 120,09 / 275,98 | `test_oracle_skew.py::test_cusum_kette_gleich_simulation_im_band`, `test_claims.py::test_schiefe` |
| Vorzeichen-CUSUM, ARL₀ exakt als Bruch bei H = 3 / 8 / 9 (verteilungsfrei bei stetiger Verteilung) | 14 / 266 / 442 | `test_handbeispiele.py::test_h4_schiefe_geschlossen_und_vorzeichen_cusum_von_hand`, `test_oracle_skew.py::test_vorzeichen_cusum_gleitkomma_gleich_bruch_und_gegen_zufallsbewegung` |
| Normale Werte, Lageverschiebung 1 σ (ν = 0): Vorzeichen-CUSUM (H = 9) / CUSUM; Exponential, Lageverschiebung 2 σ: Vorzeichen-CUSUM | 15,45 / 8,58; Exponential: 9 Beobachtungen (untere Stützstelle, Artefakt) | `test_oracle_skew.py::test_artefakt_exponential_nach_lageverschiebung_meldet_nach_genau_h` |
| Rückblick: Reihen ohne jeden Bruch (n = 200, Mindestlänge 5, 400 Reihen), Anteil mit mindestens einem gefundenen Bruch bei c = 2, φ = 0 / 0,3 / 0,6 / 0,9: Rohdaten; Residuen (φ bekannt) | 4,2 / 34,0 / 88,2 / 98,8 %; 4,0 / 4,5 / 2,0 / 5,5 % | `test_oracle_segment.py`, `test_tables.py::test_segmentierung_gegen_v14_im_band`, `test_claims.py::test_segmentierung` |
| Rückblick, φ = 0,9, Rohdaten, Anteil bei c = 2 / 4 / 6 / 10 (c = 6 und 10: etwa 62 bis 65 % bzw. etwa 31 bis 36 %, eine Nachrechnung mit 2000 Reihen liegt am unteren Ende) | 98,8 / 85,8 / 64,5 / 36,2 % | `test_claims.py::test_segmentierung` |
| Spielzeug Zwei-Zustände-Kette (P(Spitze) = 1/10), φ = 0 / 1/2 / 4/5: Regel A (erste Spitze) / Regel B (zwei Spitzen in Folge) | 10 / 19 / 46 / 110 / 409/11 / 2386/41 | `test_handbeispiele.py::test_h1_zwei_zustaende_kette_von_hand` |
| Spielzeug diskrete Wartezeit: Lag 1 / Lag 2; weitere Beobachtungen aus den Zuständen 0 / 1 / 2; ARL bei Alarm ab W ≥ 3 (iid-Sicht 8) | 5/6 / 13/18; 33 / 30 / 21; 221/8 = 27,625, das 3,45fache | `test_handbeispiele.py::test_h2_diskrete_wartezeit_von_hand` |
| Varianz einer Summe von m AR(1)-Werten: m = 2, φ = 1/2; Grenzwert je Wert bei φ = 4/5 | 3 statt 2; 9 | `test_handbeispiele.py::test_h5_beispiele_und_grenzwert` |
| ARL₀-Ziel; einseitige Grenze; Schwelle des einseitigen CUSUM | 1/(2Φ(−3)) = 370,398; c = 2,7822; h = 4,0965 | `test_oracle_cusum.py::test_schwelle_h_fuer_arl_370_398` |

**Gelesen als Zählung:** „Die ARL₀ steigt für die Shewhart-Karte und fällt für den CUSUM“ gilt nur im Gitter φ = 0,1 bis 0,98 (zweiseitig Shewhart: steigend für φ > 0; CUSUM: fallend bis φ = 0,8, danach steigend, aber bis φ = 0,98 unter 370,4) und **nicht** für negatives φ (CUSUM −0,3: 10,3fach, zweiseitige Shewhart-Zahl symmetrisch) und nicht für den zweiseitigen CUSUM als exakte Zahl (nur Simulation). „Die Residuenkarte erkennt langsamer“ gilt, gezählt über die 18 Zellen je Karte (φ = 0,3 / 0,6 / 0,9, δ = 0,5 / 1 / 2, beide Arten der Änderung, gegen die nachgeeichte Rohkarte), für die Shewhart-Karte in 17 Zellen (Ausnahme: Messversatz δ = 2, φ = 0,9) und für den CUSUM in 13; schneller ist die Residuen-CUSUM-Karte in den 5 Zellen mit δ = 2, außer bei Verschiebung des Prozessmittels mit φ = 0,9. Bei kleinem δ (0,5 und 1; 12 Zellen je Karte) ist die Residuenkarte immer langsamer, um den Faktor 1,4 bis 2,9 (Shewhart) bzw. 1,1 bis 1,6 (CUSUM); diese Faktoren des Plans sind hier nachgerechnet. „Alarme kommen in Klumpen“ ist gemessen (q), die Erklärung des Anstiegs bei φ = 0,9 nur zu etwa zwei Dritteln (0,66; bei φ = 0,6: 0,87; bei φ ≤ 0,5 ist der Anstieg unter 10 % und es wird kein Anteil angegeben).

**Nominalwerte** (φ = 0, die die Verdikte der App als Bezug nennen): P(T ≤ 100) = 0,237 (Shewhart) und 0,230 (CUSUM); Median 257 (Shewhart und CUSUM).

**Rechenzeiten (Messung dieser Maschine, kein Befund; ein BLAS-Thread, Windows, Python 3.14):** Shewhart-Zellenkette live 0,3 bis 0,4 s je φ; CUSUM-Kette (m = 16 und 32, n = 128) 0,3 bis 0,9 s; M/M/1-Lag und Shewhart 0,5 s; die Seite braucht beim ersten Aufruf rund 4 bis 9 s (alles zwischengespeichert), danach unter 1 s je Änderung eines Reglers, der bekannte Werte wiederverwendet; Simulation mit R = 200 und tmax = 10 000 rund 0,1 s. Streamlit Cloud nicht gemessen. Messung dieser README, nicht im Test (der Test begrenzt nur nach oben).

## Ehrliche Grenzen und 🚧 Nicht gemessen

* **Modell:** AR(1) mit normaler Randverteilung, bekannten Parametern (Mittel, Streuung und φ), zero-state, ν = 0; andere Abhängigkeit (AR(2), saisonal, langes Gedächtnis) und andere Änderungen (Streuung, Dynamik) sind nicht gemessen.
* **Maß:** Beobachtungen bis zum Alarm, Zähleinheit je Satz genannt. Das Lorden-, das Pollak- und das Shiryaev-Maß sind bei abhängigen Werten **nicht gemessen**; alle Verzögerungen gelten für ν = 0.
* **Zweiseitiger CUSUM auf AR(1)** ist nur als Simulation belegt (exakt nur einseitig); die Näherung „einseitig/2“ gilt nur bis φ = 0,3. φ = −0,6, 0,95 und 0,98 (CUSUM) sind nicht durch Simulation geprüft (0,95 und 0,98 nur auf dem feinen Gitter).
* **CUSUM-Ketten:** Gitterfehler rund 0,2 %, bei großen Schwellen (h über 10) größer; die nachgeeichte Schwelle bei φ = 0,9 (h = 22) ist auf etwa 1 % genau.
* **Residuenkarte** mit bekanntem φ ist der Bestfall; mit geschätztem φ̂ streut die bedingte ARL₀ (Stück 4b der Reihe). Messversatz und Verschiebung des Prozessmittels sind gewählte Änderungsarten.
* **M/M/1 ist nicht das Gate:** das Gate (M/M/2/6, Simulation) hat andere Größen; EWMA, Zusatzregeln, MEWMA und verteilungsfreie Karten bei abhängigen Werten sind nicht gemessen, ebenso schiefe abhängige Werte außer M/M/1.
* **Ausgedachte Reihen:** ein Nutzen einer Alarmregel in Logistik, Lager oder Fertigung ist nicht belegt.

Nicht gemessen sind außerdem (aus der Planung, hier nicht nachgemessen): **Linux** (die Zahlen sind unter Windows gerechnet; Neu-Rechnen-Tests sind Bänder; `demo_linux_check.py` wurde nicht ausgeführt); der **Gitterfehler** der zweidimensionalen Ketten getrennt vom Richardson-Gewinn (getestet ist nur der Abstand zu feineren Gittern); die Frage, ob der **AR(1)-Filter zum Gate passt** (eine frühere Messung: Shewhart P(T ≤ 100) 0,769 statt 0,549, gelb, nicht neu gemessen); **Ausdünnen** (jeder k-te Lkw) als Mittel gegen die Abhängigkeit (Stück 8).

**Quellen und Stufen** (W = gelesen, sonst „nicht gelesen“; die Stufen stammen aus der Planung, hier nicht neu gelesen). Zwetsloot/Jones-Farmer/Woodall 2024 **(W)**, Abschnitte zu Phase I und II: bei Autokorrelation ist das übliche Vorgehen, ein Modell anzupassen und die Residuen zu überwachen; Aggregation über die Zeit senkt die Autokorrelation und kostet Auflösung. Vanhatalo/Kulahci 2015 **(W)**, Abschnitte 1, 5, 6: die Residuenkarte stabilisiert die ARL₀ **bei perfekt bekanntem Modell**; die Rohdatenkarte mit Standardgrenze zeigt je nach Schätzer der Kovarianz sehr kleine ARL₀ (anderes Experiment als hier). Larsen u. a. (arXiv 1907.05453) **(W, Einleitung, aus der Literatur zitiert)**: Residuen eines AR(1) mit positivem φ erkennen einen Sprung beim ersten Wert danach gut und dann schlechter. Romano u. a. 2022 (DeCAFS) **(W, Abstract, Einleitung)**: Standardstrafen überschätzen die Zahl der Brüche bei autokorreliertem Rauschen. Chakraborti/Graham 2019 **(W)**: verteilungsfreie Karten haben die Lauflänge verteilungsfrei nur bei stetigen **unabhängigen** Daten. Jones-Farmer u. a. 2014 **(W)**: schiefe Verteilungen lassen die empirische Fehlalarmwahrscheinlichkeit der Phase I auf rund 30 % steigen (Zitatstelle). Johnson/Bagshaw 1974 und Alwan/Roberts 1988 **(S, nur Zusammenfassung bzw. Suchtreffer, Wortlaut nicht gelesen)**: positive Autokorrelation erster Ordnung verringere die ARL des CUSUM, negative erhöhe sie; Zeitreihenmodell anpassen und die Karte auf die Residuen anwenden. **Nicht gelesen:** Basseville/Nikiforov, Montgomery, Brook/Evans 1972, Page 1954, Roberts 1959. Keine Neuheit: alles Lehrbuch und Stand der Technik; neu ist die Vorführbarkeit mit Zähleinheit und exakten Orakeln. Was nicht aus den Quellen steht, ist „eigene Nachrechnung in diesem Modell“; es gibt hier **keinen Test aus einer Literaturtabelle**.

## Tests

**304 Tests**, seriell rund 175 s (diese Maschine, Windows, Python 3.14, ein BLAS-Thread; `pytest tests/ -q`; die Zeit hängt stark von der Last des Rechners ab, andere Sitzungen liefen gleichzeitig). Zusammensetzung:

- **Handbeispiele** (`test_handbeispiele.py`, 20): H1 bis H6 des Plans als Brüche, Ganzzahlen und Tafelwerte, jede Zahl von einem Rechenweg ohne den Code der Demo erzeugt (eigene Gauß-Elimination mit Brüchen, Quadratur, abgeschnittene Ketten, Überlebenssummen) und danach mit den Funktionen der Demo verglichen.
- **Orakel** (`test_oracle_shewhart.py` 25, `test_oracle_cusum.py` 20, `test_oracle_queue.py` 29, `test_oracle_skew.py` 19, `test_oracle_residual.py` 19, `test_oracle_segment.py` 8): Zellenkette gegen Integralgleichung (Shewhart auf AR(1), 5·10⁻⁴), Quadratur gegen Einschluss-Ausschluss (Klumpen), zweidimensionale Kette gegen eindimensionale Integralgleichung bei φ = 0 (0,1 %) und gegen Simulation (Band von 4 Standardfehlern, SplitMix64), Spiegelungsformeln gegen Schleifen, Lag 1 geschlossen gegen Quadratur und Brüche, geschlossene Lösung der M/M/1-Integralgleichung gegen Kette und Simulation (Restfehler der Integralgleichung unter 10⁻⁹), Gate gegen den Geburts-und-Sterbeprozess und Little, Residuenkarte gegen eine Zellenkette auf dem letzten Wert und gegen die simulierte Reihe, optimale Zerlegung gegen Aufzählung aller Zerlegungen; dazu Eigenschaften an Zufallsinstanzen (Symmetrie in φ, Monotonie in Grenze, Schwelle und Referenzwert, Lauflängenverteilung monoton, gleicher Zufalls-Seed gleiche Ausgabe) und die Gegenproben für Absolutaussagen („nur bis φ = 0,3“, „nicht monoton“, „addieren sich nicht“).
- **Claims** (`test_claims.py`, 21): jede Zeile der Tabellen „Befunde“ und „Befunde und Korrekturen“ aus der Rechnung bzw. der gefrorenen Tabelle neu erzeugt und wörtlich in dieser README gesucht, Zählaussagen (17 von 18 Zellen u. a.), der Text „Wo die Annahmen enden“ in App und README wortgleich, Testzahl, Pflichtabschnitte, Wortliste.
- **Tabellen** (`test_tables.py`, 17): Aufbau, Zufalls-Seed und R je Zelle (alle verschieden), Beziehungen, Abgleich mit den Vormessungen im Band, Neurechnung kleiner Proben (`tools/abw_make_tables.py --check`).
- **Verdikte** (`test_verdicts.py` 15: reine Funktionen an handgesetzten Zeilen; `test_preset_verdicts.py` 11: je Schnellstart die erwartete Verdikt-Zeile aus der laufenden App), **Randfälle** (15: φ am Rand, δ = 0, R = 1, Zufalls-Seed 0, Zensur, ρ nahe 1, sehr große Schwelle, Schwelle 1, Atom bei 0), **Simulation** (11: Zensur, Obergrenzen, Klumpen, Konsistenz der Läufe), **Presets** (11), **Layout** (25: Schnellstart in der Seitenleiste, fester Kopf, Hervorhebung, jeder Regler hebt sie auf), **App** (18, AppTest: jeder Button, jeder Regler an beiden Grenzen, jede Option jeder Auswahl, Gate-Knopf, Permalink), **Abbildungen** (5), **Zufallsquelle** (12: Bitgleichheit der Stapel-Fassung mit der Klasse, Exponentialzahlen), Footer (2), Laufzeit-Importe (1).
- **Plattform:** Neu-Rechnen-Tests sind Bänder (4 Standardfehler) oder relative Toleranzen, nie bitgleich; Gleitkomma-Letztstellen nur mit Toleranz. Der Lauf unter Linux (`demo_linux_check.py`, Docker) wurde **nicht** ausgeführt; ein unabhängiger Aussagenprüfer (`claim_review.py`) ebenfalls nicht.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: vier Experimente, Tab Schiefe, Rückblick, Grenzen, Formeln |
| `abw_chain.py` | Shewhart auf AR(1) (Zellenkette, Integralgleichung, Klumpen), Residuenkarte, CUSUM bei unabhängigen Werten (Integralgleichung, Brook-Evans-Kette) |
| `abw_cusum.py` | zweidimensionale Kette des CUSUM auf AR(1) und auf der Wartezeit (dünn besetzt, Richardson) |
| `abw_queue.py` | M/M/1 (Lag, geschlossene Lösung, Ketten) und Gate M/M/2/6 (Randverteilung, Simulation) |
| `abw_skew.py` | Schiefe, Vorzeichen-CUSUM mit Brüchen |
| `abw_toy.py` | Spielzeuge von Hand (Zwei-Zustände-Kette, diskrete Wartezeit, Summenvarianz) als Brüche |
| `abw_scenario.py` | Simulationen (Reihen, Läufe, Klumpen, M/M/1, Gate) mit SplitMix64 |
| `abw_segment.py` | optimale Zerlegung und Fehlbrüche bei autokorreliertem Rauschen |
| `abw_normal.py`, `abw_rng.py` | Stapel-Zufallszahlen und die SplitMix64-Vorlage (unverändert) |
| `abw_reference.py`, `abw_tables.json` | gefrorene Tabellen (mit Stand, Zufalls-Seed und R) |
| `abw_evaluation.py` | Einstellungen, Zahlenformat, Verdikte als reine Funktionen |
| `abw_constants.py`, `abw_presets.py`, `abw_layout.py` | Regler-Grenzen und Presets, Permalink und Hervorhebung, fester Kopf |
| `abw_texts.py`, `abw_visualization.py` | Texte (Zahlen aus Rechnung und Tabellen), Plotly-Abbildungen (Achsen fest) |
| `tools/abw_make_tables.py` | erzeugt `abw_tables.json` (`--lang`) und prüft sie mit kleinen Proben (`--check`) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit-Bibliotheken (ungepinnt) und Entwicklungsumgebung |
| `tests/` | Handbeispiele, Orakel, Claims, Tabellen, Verdikte, Randfälle, Presets, Layout, App, Footer, Laufzeit-Importe, Zufallsquelle, Abbildungen |

## Bewusst nicht umgesetzt

* **Geschätztes φ̂ (Phase I bei Autokorrelation):** Stück 4b; hier alle Parameter bekannt.
* **EWMA, Zusatzregeln, MEWMA und verteilungsfreie Karten bei abhängigen Werten**, **andere Abhängigkeit als AR(1)**, **andere Änderungen als Messversatz und Verschiebung des Prozessmittels**.
* **Lorden-, Pollak- und Shiryaev-Maß** bei abhängigen Werten; alle Verzögerungen gelten für ν = 0.
* **Anwendung, Kosten, Prognose-Reset und Bestätigungsregel** (`niveausprung-demo` rechnet Meldungen je Jahr bei Autokorrelation 0,6 mit einem Prognose-Nachführer: anderes Maß, nicht umrechenbar; die Deutung „Autokorrelation erklärt das Kippen“ ist dort eine Lesart, die ein Teil der Zahlen hier stützt, dort aber nicht getestet ist). Batch Means und Einschwingzeit für Intervalle des Mittelwerts stehen in `output-analysis-demo` (Autokorrelation 0,99 bei Auslastung 90 %; die exakte Lag-1-Zahl hier ist 0,9904). Die exakte Gate-Kette und die Abhilfen am Gate (Ausdünnen) gehören zu Stück 8 (`lastwechsel-gate-demo`).
* **Erklärseiten, Grundlagen-Begriffe, Website-Karte und Konzepte-Spezifikation** der Linie: Website-Einbindung, nicht Teil dieser Demo.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python tools/abw_make_tables.py --check     # gefrorene Tabellen gegen eine kleine Neurechnung
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
