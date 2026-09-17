# Warstwa komunikacji — analiza obrazu drogowego 0.3.6

Demonstrator offline szablonowej warstwy komunikatów. Nie jest prototypem ADAS.
Aplikacja analizuje wyłącznie pojedyncze obrazy JPG, JPEG i PNG przesłane przez
użytkownika. Tryb prezentacji i szyk zdania wynikają z jawnych szablonów oraz
ręcznego wyboru użytkownika, a nie ze stanu kierowcy. Aplikacja:

1. wykrywa wybrane obiekty drogowe lokalnym modelem YOLO;
2. oblicza heurystyczny wskaźnik wybranych cech technicznych obrazu;
3. wyznacza priorytet komunikatu **wyłącznie z detekcji** (wskaźnik nie punktuje);
4. generuje opisowy komunikat w jednym z trzech trybów;
5. zapisuje i udostępnia do pobrania raport JSON oraz oznaczony obraz.

Prototyp nie jest certyfikowanym systemem ADAS. Nie działa w czasie rzeczywistym,
nie steruje pojazdem, nie przewiduje kolizji, nie mierzy odległości i nie
diagnozuje użytkownika.

Kod własny jest udostępniany na licencji GNU AGPL-3.0-or-later (`LICENSE`) ze
względu na zależność od biblioteki Ultralytics.

## Instalacja na Windows

```powershell
cd adaptive-driving-assistant
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-tested.txt
```

Minimum to Python 3.11 (`pyproject.toml`). Walidację wersji 0.3.6 z pracy
wykonano na Pythonie 3.12.10.

## Lokalny model YOLO

Aplikacja używa stałego modelu `yolo11n.pt`. Plik wag musi znajdować się w
głównym katalogu projektu przed uruchomieniem analizy:

```text
adaptive-driving-assistant/yolo11n.pt
```

Aplikacja nie pobiera modelu automatycznie z internetu i nie udostępnia pola do
zmiany modelu w interfejsie. Przed analizą sprawdza jego sumę SHA-256:

```text
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

## Uruchomienie

```powershell
streamlit run app.py
```

Dołączona konfiguracja ogranicza nasłuchiwanie do `127.0.0.1` i maksymalny
rozmiar przesyłanego pliku do 20 MB.

## Zakres aplikacji

- Wejście: jeden zweryfikowany obraz JPG/JPEG/PNG, maksymalnie 20 MB i 40 mln
  pikseli. Rozszerzenie musi odpowiadać rzeczywistemu formatowi.
- Detekcja: lokalny adapter Ultralytics YOLO dla klas `person`, `car`, `bus`,
  `truck`, `bicycle`, `motorcycle`.
- Heurystyczny wskaźnik cech obrazu: jasność, kontrast, ostrość, nasycenie,
  gęstość krawędzi i udział kanałów nasyconych do 255. Od 0.3.6 kalibracja
  przechodzi test 7.4.5 (skrajne rozmycie i prześwietlenie → poziom ograniczony).
  Wynik trafia do raportu i może pojawić się w trybie rozszerzonym; **nie zmienia
  priorytetu komunikatu**.
- Priorytet komunikatu: jawne reguły wyboru głównej detekcji i kolejności
  informacji, nie ocena ryzyka.
- Komunikaty: trzy ręcznie wybierane tryby `minimalny`, `rozszerzony`,
  `rozszerzony z objaśnieniem`. Tryb z objaśnieniem podaje liczbę wykryć,
  wynik ufności głównej detekcji i pozostałe klasy — nie metakomentarz.

Wskaźnik opisuje wybrane cechy techniczne pliku. Nie mierzy czytelności sceny
dla człowieka, widzialności drogowej ani bezpieczeństwa jazdy.

Priorytet jest kategorią logiki doboru komunikatu. Nie jest oceną ryzyka kolizji
ani zaleceniem manewru. Obiekt główny jest wybierany jedną wspólną regułą dla
priorytetu, komunikatu i raportu.

Raport zawiera wersję aplikacji i środowiska, parametry analizy, sumy SHA-256
wejścia, modelu i obrazu wynikowego, pełne detekcje, wynik heurystyki, przesłanki
priorytetu oraz komunikat. Przycisk „Usuń wynik i jego lokalne pliki” usuwa
artefakty bieżącej analizy z katalogu `data`.

## Testy

```powershell
pytest
ruff check .
```

Testy sprawdzają logikę implementacji, w tym wyczerpujące wyliczenie języka
komunikatów. Nie są to metryki skuteczności modelu ani ocena odbioru u ludzi.

Opcjonalne pomiary z Tabeli 8.4 pracy (ablacja `imgsz` i bootstrap obrazowy)
nie wchodzą do PE1. Skrypty leżą w katalogu głównym projektu:
`ablacja_imgsz_bdd.py` oraz `bootstrap_obrazowy_bdd.py`. Mini zbiór w
`data/samples` ma sześć plików z pkt 7.4; archiwum zawiera dodatkowo
`zidane.jpg` (Ultralytics).

## Zweryfikowane środowisko

- Windows 11;
- Python 3.12.10;
- Ultralytics 8.4.102;
- Torch 2.13.0;
- Streamlit 1.59.2;
- NumPy 2.5.1;
- Pillow 12.3.0.

Dokładne wersje bezpośrednich i krytycznych zależności zapisano w
`requirements-tested.txt`.

## Komponenty zewnętrzne

Model YOLO11n i biblioteka Ultralytics nie są autorskim modelem projektu.
Informacje o źródle, cytowaniu i licencji znajdują się w
`THIRD_PARTY_NOTICES.md`.
