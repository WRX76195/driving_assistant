# Rejestr testów technicznych

## Wersja 0.3.6 (rekalibracja wskaźnika)

Zmiany zachowania względem 0.3.5:

- nowy wzór S (wagi 0,15 / 0,45 / 0,28 / 0,12, skala ostrości 0,02, kara
  `clip_hi`); test na `bus.jpg` przechodzi;
- priorytet nadal **nie** używa wskaźnika;
- skrypty: ranking AP i trzy polityki priorytetu — wykonane na 2000 parach
  (zbiór PE1). Oficjalny val 10 000 poza zakresem.
- w archiwum: opcjonalny `zidane.jpg` oraz skrypty `ablacja_imgsz_bdd.py` i
  `bootstrap_obrazowy_bdd.py`.

Kontrole automatyczne 0.3.6: `pytest` (68 passed) oraz `ruff check .`.
 `docs/scene_readability_table_7_7_0.3.6.json`.
Batch mini zbioru: `data/samples/batch_results_0.3.6.json`.

## Wersja 0.3.5

Zmiany zachowania względem 0.3.4:

- wskaźnik cech obrazu nie wchodzi do punktacji priorytetu;
- pusta lista detekcji daje priorytet „brak komunikatu” i komunikat
  „Nie wykryto obiektów z obsługiwanych klas.”;
- tryb rozszerzony z objaśnieniem podaje liczbę wykryć, wynik ufności i pozostałe
  klasy zamiast metakomentarza.

Ewaluacja YOLO na 2000 parach BDD100K **nie zmienia się** (ten sam detektor i
ten sam protokół): `docs/bdd_eval_metrics_thresholds_0.3.4.json`.

Kontrole automatyczne 0.3.5: `pytest` oraz `ruff check .` w katalogu projektu.
Batch mini zbioru: `data/samples/batch_results_0.3.5.json`.
Monte Carlo priorytetu: `docs/priority_monte_carlo_0.3.5.json`.

## Wersja 0.3.4 (poprzedni przebieg walidacji)

Data wykonania: 16 sierpnia 2026 r.  
System: Windows 11 (`Windows-11-10.0.26200-SP0`), Python 3.12.10.  
Model: `yolo11n.pt`, SHA-256
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.

## Kontrole automatyczne

| ID | Kontrola | Wynik |
|---|---|---|
| A-001 | `pytest --basetemp=.pytest_tmp` | 68/68 testów zaliczonych (wersja 0.3.6) |
| A-002 | Ruff (`src`, `app.py`, `scripts`, `tests`) | All checks passed |
| A-003 | `compileall` | bez błędów |
| A-004 | pokrycie pakietu | 91% instrukcji; 583 instrukcje, 53 niepokryte (pomiar wersji 0.3.4, orientacyjnie; Coverage.py 7.15.4) |
| A-005 | CLI `scripts/analyze_image.py data/samples/bus.jpg` | wynik zgodny z kontrolą funkcjonalną |
| A-006 | Batch `scripts/analyze_samples.py` | 6/6 obrazów mini zbioru; `data/samples/batch_results_0.3.6.json` (detekcje jak 0.3.5) |
| A-007 | Ewaluacja BDD `scripts/evaluate_bdd_pairs.py` (2000 par, conf 0,35) | mikro P/R/F1: 0,8406 / 0,3225 / 0,4661 (odtworzone w 0.3.6) |
| A-008 | Ewaluacja BDD przy progach 0,05–0,95 | najlepsze mikro-F1: 0,5057 przy conf 0,15; siatka AP 0,4240 / ranking AP 0,4360; `docs/bdd_eval_metrics_0.3.6.json` |

## Pełny przebieg z modelem YOLO

| Pole | Wynik |
|---|---|
| Obraz | `data/samples/bus.jpg`, SHA-256 `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` |
| Konfiguracja | próg 0,35; tryb rozszerzony |
| Detekcje | `bus`: 1; `person`: 4 |
| Wskaźnik cech obrazu | dobry; 0,936405 (kalibracja 0.3.6) |
| Priorytet komunikatu | wysoki; główna klasa `person` |
| Komunikat | „Wykryto osobę w centralnej części analizowanego obrazu.” |
| Artefakty | oznaczony obraz i ścisły raport JSON w `data/output/<znacznik>/` |

Wynik sprawdza integrację komponentów. Osobna ewaluacja precyzji / kompletności / F1
względem etykiet BDD100K jest opisana  pracy i w pliku
`docs/bdd_eval_metrics_thresholds_0.3.4.json`. Nie jest miarą bezpieczeństwa
jazdy ani trafności wskaźnika cech obrazu.

## Test interfejsu

| ID | Scenariusz | Wynik |
|---|---|---|
| UI-001 | przesłanie poprawnego `bus.jpg` | podgląd i analiza poprawne |
| UI-002 | oba przyciski pobierania | raport JSON i oznaczony obraz dostępne |
| UI-003 | usunięcie przesłanego pliku | poprzedni wynik znika; brak nieaktualnych danych |
| UI-004 | plik tekstowy nazwany `.jpg` | kontrolowany komunikat bez stack trace |
| UI-005 | konfiguracja serwera | nasłuchiwanie wyłącznie na `127.0.0.1:8501` |
| UI-006 | zmiana progu wyniku ufności po analizie | poprzedni wynik znika przed ponowną analizą |
| UI-007 | zmiana trybu komunikacji po analizie | poprzedni wynik znika przed ponowną analizą |
| UI-008 | PNG o niebezpiecznie dużych deklarowanych wymiarach | kontrolowany komunikat bez stack trace i błędów konsoli |
