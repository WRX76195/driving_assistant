# Weryfikacja techniczna prototypu (0.3.6)

Procedura poniżej służy potwierdzeniu, że środowisko, testy jednostkowe,
mini zbiór obrazów i pipeline analizy działają zgodnie z oczekiwaniem.
**Nie jest to pomiar jakości modelu YOLO** ani ocena bezpieczeństwa jazdy.

## 1. Instalacja

```powershell
cd adaptive-driving-assistant
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-tested.txt
```

Umieść zweryfikowany plik modelu w katalogu głównym projektu:

```text
adaptive-driving-assistant/yolo11n.pt
```

Oczekiwana suma SHA-256 modelu:

```text
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

## 2. Testy jednostkowe

```powershell
pytest
```

Oczekiwany wynik: wszystkie testy przechodzą. Testy sprawdzają
logikę implementacji, walidację wejścia, serializację raportu i wyliczenie
języka komunikatów — nie metryki detekcji na zbiorze drogowym ani odbiór u ludzi.

## 3. Przygotowanie mini zbioru testowego

```powershell
python scripts/prepare_samples.py
python scripts/analyze_samples.py
```

Skrypt `prepare_samples.py` tworzy katalog `data/samples`, pobiera brakujący
`bus.jpg` (Ultralytics), opcjonalny `zidane.jpg` (poza mini zbiorem z pkt 7.4),
generuje syntetyczne przypadki brzegowe i aktualizuje `manifest.json`. Plik
`street.jpg` jest syntetycznym obrazem wygenerowanym na potrzeby projektu i musi
być obecny lokalnie w archiwum. Jego pochodzenie oraz opis generacyjny zapisano
w `data/samples/PROVENANCE.md`. Batch `analyze_samples.py` analizuje wyłącznie
sześć plików mini zbioru i zapisuje `data/samples/batch_results_0.3.6.json`.

**Uwaga:** sama aplikacja działa offline; sieć jest potrzebna tylko przy pierwszym
pobraniu brakujących obrazów Ultralytics.

## 4. Analiza przykładowego obrazu przez CLI

```powershell
python scripts/analyze_image.py data/samples/bus.jpg
```

Oczekiwane działanie:

- w konsoli pojawia się podsumowanie (liczba obiektów, wskaźnik cech obrazu,
  priorytet, komunikat);
- w podkatalogu `data/output/<znacznik_czasu>/` powstają pliki `bus_oznaczony.jpg`
  i `bus_raport.json`.

Oczekiwana suma SHA-256 pliku `data/samples/bus.jpg`:

```text
c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63
```

Sumę można sprawdzić np. w PowerShell:

```powershell
Get-FileHash data\samples\bus.jpg -Algorithm SHA256
```

## 5. Opcjonalna weryfikacja interfejsu Streamlit

```powershell
streamlit run app.py
```

Prześlij `data/samples/bus.jpg` i porównaj wynik z outputem CLI (te same
moduły pipeline, różny kanał wejścia).

## 6. Co weryfikacja potwierdza, a czego nie

| Potwierdza | Nie potwierdza |
| --- | --- |
| poprawność instalacji i testów | wpływ prototypu na bezpieczeństwo jazdy |
| działanie pipeline na znanym obrazie | trafność heurystyki względem widzialności meteorologicznej |
| powtarzalność sum SHA-256 wejścia i modelu | ocena komunikatu pod kątem komfortu kierowcy |
| precision/recall/F1 na podzbiorze BDD100K (§7) | skuteczność w czasie rzeczywistym / w pojeździe |

## 7. Ewaluacja detektora na podzbiorze BDD100K (opcjonalnie)

Poza weryfikacją techniczną można policzyć precision / recall / F1 względem
etykiet referencyjnych na parach obraz–JSON ze zbioru BDD100K:

```powershell
python scripts/evaluate_bdd_pairs.py --confidence-sweep ui
```

Skrypt szuka `bdd100k_2000_pairs.zip` w `data/`, w katalogu projektu oraz na
Pulpicie. Ścieżkę można też podać jawnie: `--zip "C:\Users\...\bdd100k_2000_pairs.zip"`.
Raport zawiera AP z siatki progów **oraz** AP z pełnego rankingu ufności.
Zrzut detekcji: `--save-detections data/output/bdd_2000_detections.json`.
Trzy polityki priorytetu: `scripts/compare_priority_policies.py`.
Oficjalny val 10 000 leży poza zakresem PE1.

Wynik trafia do `data/output/bdd_eval_metrics.json`. Zagregowany raport z
przebiegu 0.3.6 opisanego w pracy (siatka 19 progów odtworzona + ranking AP)
jest w `docs/bdd_eval_metrics_0.3.6.json`. Odtworzenie siatki 0.3.4:
`docs/bdd_eval_metrics_thresholds_0.3.4.json`. Surowego BDD100K **nie
dołącza się** do archiwum projektu (licencja UC Berkeley / BAIR — użycie
edukacyjne i badawcze non-profit; archiwum ma ok. 135 MB i 2000 zdjęć).
Testy jednostkowe **nie wymagają** tego ZIP — sprawdzają logikę ewaluacji
na syntetycznych parach. Mini zbiór `data/samples` nadal nie służy
do metryk jakości modelu.

Opcjonalne skrypty z Tabeli 8.4 pracy (pomiarów nie wykonano; nie są wymagane
do PE1) uruchamia się na **rozpakowanym** katalogu 2000 par:

```powershell
python ablacja_imgsz_bdd.py --pairs-dir <katalog_2000_par> --imgsz 640 960 1280 --output ablacja_imgsz.json
python bootstrap_obrazowy_bdd.py --pairs-dir <katalog_2000_par> --replicates 2000 --seed 20260827 --output bootstrap.json
```

Raport referencyjny utworzono z 2000 par wybranych przez autora z całego
BDD100K. Skrypt wczytał wszystkie wspólne pary nazw z archiwum, uporządkowane
alfabetycznie. Wszystkie pary wczytano poprawnie; 1992 obrazy zawierały co
najmniej jedną obsługiwaną ramkę referencyjną, a 8 nie zawierało takich ramek.
Archiwum wejściowe miało SHA-256
`96b5d5e9e226e17e3b9f249eb568c033368c8d8041fa62469e3ebcaa284f7f72`.
Wynik opisuje ten zestaw, nie cały BDD100K i nie oficjalny split val.

## 8. Co jeszcze nie jest mierzone

Ewaluacja BDD nie zastępuje walidacji wskaźnika cech obrazu względem ocen
referencyjnych ani badania wpływu komunikatów na odbiorcę.
