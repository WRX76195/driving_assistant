# Mini zbiór testowy obrazów

Katalog `data/samples` zawiera mały, powtarzalny zestaw obrazów do weryfikacji
technicznej prototypu w wersji 0.3.6. Nie służy ocenie jakości modelu YOLO —
umożliwia sprawdzenie, czy pipeline, heurystyka i raport JSON działają
poprawnie na znanych wejściach.

## Skład mini zbioru 

| Plik | Źródło | Przeznaczenie |
| --- | --- | --- |
| `bus.jpg` | Ultralytics | Scena drogowa z autobusem — główny przykład weryfikacji CLI |
| `street.jpg` | obraz syntetyczny wygenerowany za pomocą narzędzia OpenAI | Zagęszczona scena uliczna — wiele klas jednocześnie |
| `dark.png` | syntetyczny | Niska jasność — test heurystycznego wskaźnika cech obrazu |
| `bright.png` | syntetyczny | Wysoka jasność — test heurystycznego wskaźnika cech obrazu |
| `blur.png` | syntetyczny | Rozmycie — test heurystyki ostrości |
| `empty_scene.png` | syntetyczny | Brak obiektów — test komunikatu bez detekcji |

Metadane (sumy SHA-256, opisy) znajdują się w `manifest.json`. Szczegółowe
pochodzenie syntetycznego pliku `street.jpg`, data utworzenia i użyty opis
generacyjny znajdują się w `PROVENANCE.md`. Manifest jest odświeżany przez
skrypt `scripts/prepare_samples.py`.

## Przygotowanie

Z katalogu głównego projektu:

```powershell
python scripts/prepare_samples.py
```

Skrypt:

- pobiera brakujący `bus.jpg` z sieci;
- pobiera brakujący opcjonalny `zidane.jpg` (Ultralytics), poza mini zbiorem;
- generuje obrazy syntetyczne, jeśli ich brakuje;
- wymaga obecności lokalnego `street.jpg` w `data/samples` (syntetyczny obraz jest dołączony do archiwum);
- aktualizuje `manifest.json` (sześć plików mini zbioru oraz `zidane.jpg`, jeśli jest).

**Uwaga:** sama aplikacja Streamlit / CLI działa offline. Dostęp do sieci jest
potrzebny wyłącznie przy pierwszym przygotowaniu brakujących obrazów Ultralytics.

## Użycie z CLI

```powershell
python scripts/analyze_image.py data/samples/bus.jpg
python scripts/analyze_image.py data/samples/street.jpg --mode rozszerzony-z-objasnieniem
python scripts/analyze_samples.py
```

Wyniki pojedynczej analizy trafiają do `data/output`. Podsumowanie batcha:
`data/samples/batch_results_0.3.6.json`.

## Użycie ze Streamlit

```powershell
streamlit run app.py
```

Szczegółową procedurę weryfikacji opisano w [docs/weryfikacja.md](../../docs/weryfikacja.md).
