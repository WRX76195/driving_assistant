# Mapowanie na pracę magisterską

Uproszczona wersja realizuje część implementacyjną pracy jako demonstrator
offline warstwy komunikatów. Nie jest prototypem ADAS. Słowo „adaptacyjny”
dotyczy wyłącznie wyboru trybu prezentacji i szyku zdania, a nie stanu kierowcy.

## Implementacja

- `app.py`: interfejs użytkownika.
- `src/adaptive_driving_assistant/detector.py`: adapter YOLO dla stałego lokalnego
  modelu `yolo11n.pt`.
- `src/adaptive_driving_assistant/scene_readability.py`: heurystyczny wskaźnik
  wybranych cech technicznych obrazu (historyczna nazwa modułu).
- `src/adaptive_driving_assistant/priority.py`: autorskie reguły priorytetu.
- `src/adaptive_driving_assistant/messages.py`: opisowe komunikaty w trzech
  trybach.
- `src/adaptive_driving_assistant/report.py`: raport JSON.
- `src/adaptive_driving_assistant/pipeline.py`: połączenie modułów dla obrazu,
  bezpieczny zapis przesłanego pliku i unikalne katalogi wynikowe.
- `scripts/analyze_image.py`: wiersz poleceń do analizy pojedynczego obrazu
  bez Streamlit (weryfikacja techniczna, powtarzalne uruchomienia).
- `scripts/prepare_samples.py`: przygotowanie mini zbioru testowego w
  `data/samples/` (obrazy referencyjne i syntetyczne przypadki brzegowe).
- `scripts/analyze_samples.py`: batch analizy mini zbioru →
  `data/samples/batch_results_0.3.6.json`.
- `scripts/monte_carlo_priority.py`: ocena reguł priorytetu (ziarno 20260829).
- `scripts/evaluate_bdd_pairs.py`: zbiór PE1 — 2000 par + ranking AP + `--save-detections`.
- `scripts/evaluate_bdd_official_val.py`: poza zakresem PE1 (oficjalny val 10 000 nie wchodzi do wyników).
- `scripts/compare_priority_policies.py`: VRU-first / area-first / confidence-first.
- `scripts/make_scene_readability_table.py`: Tabela 7.7b.
- `ablacja_imgsz_bdd.py`, `bootstrap_obrazowy_bdd.py`: opcjonalne pomiary z
  Tabeli 8.4 (nie wykonane, nie wymagane do PE1).
- `LICENSE`: GNU AGPL-3.0-or-later.
- `data/samples/`: mini zbiór sześciu plików z pkt 7.4 oraz opcjonalny
  `zidane.jpg` (Ultralytics, poza mini zbiorem). `prepare_samples.py`
  dopisuje `zidane.jpg` do `manifest.json`, jeśli plik jest obecny.

Od wersji 0.3.5 wskaźnik cech obrazu nie wchodzi do punktacji priorytetu.
Od 0.3.6 ten sam wskaźnik ma inną kalibrację (test 7.4.5 przechodzi); nadal
nie punktuje priorytetu. Tryb rozszerzony z objaśnieniem podaje liczbę wykryć,
wynik ufności i pozostałe klasy zamiast metakomentarza o charakterze komunikatu.

## Ewaluacja detektora (PE1 = 2000 par wybranych z całego BDD100K)

- `scripts/evaluate_bdd_pairs.py` — precyzja, kompletność, F1, AP z 19 progów
  **oraz** interpolowane AP z pełnego rankingu ufności. Zrzut detekcji:
  `--save-detections`. Oficjalny split val 10 000 nie jest zbiorem PE1.
  Skrypt grupuje też kompletność według rozmiaru ramki, zasłonięcia, ucięcia
  (`truncated`) i położenia w kadrze (predykat `is_central` z `domain.py`).
  Mapowanie etykiet: `pedestrian`/`rider`/`person` → `person`,
  `bike`/`bicycle` → `bicycle`, `motor`/`motorcycle` → `motorcycle`.
- Zagregowany wynik 0.3.4 (siatka 19 progów): `docs/bdd_eval_metrics_thresholds_0.3.4.json`.
- Przebieg 0.3.6 (ta sama siatka odtworzona + ranking AP): `docs/bdd_eval_metrics_0.3.6.json`.
  AP rankingu: mikro 0,436027; mAP@0,5 0,280977. Siatka: 0,424009 / 0,269988.
- Trzy polityki: `docs/priority_policies_det_0.3.6.json`, `docs/priority_policies_gt_0.3.6.json`.

## Nadal poza zakresem metryk

- trafność wskaźnika cech obrazu względem ocen referencyjnych;
- zasadność progów priorytetu w scenariuszach badawczych z oceną człowieka;
- wpływ na bezpieczeństwo jazdy.

Nie należy opisywać tej wersji jako potwierdzenia poprawy bezpieczeństwa jazdy.
