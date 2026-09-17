# Architektura

Projekt jest lokalnym demonstratorem warstwy komunikacyjnej działającej offline.
Pierwszy ekran aplikacji jest narzędziem analizy pojedynczego obrazu, a nie panelem
systemu czasu rzeczywistego.

## Komponenty

1. Detekcja obiektów przez gotowy lokalny model YOLO:
   `src/adaptive_driving_assistant/detector.py`.
2. Heurystyczny wskaźnik cech technicznych obrazu:
   `src/adaptive_driving_assistant/scene_readability.py`.
3. Autorska logika priorytetu komunikatu:
   `src/adaptive_driving_assistant/priority.py`.
4. Generowanie opisowych komunikatów w trzech trybach:
   `src/adaptive_driving_assistant/messages.py`.

`app.py` zawiera tylko interfejs Streamlit oraz wywołania modułów. Pipeline
obrazu znajduje się w `src/adaptive_driving_assistant/pipeline.py`.

Model `yolo11n.pt` jest stałym lokalnym plikiem w katalogu projektu. Aplikacja
sprawdza jego obecność przed analizą i nie pobiera wag automatycznie.

## Przepływ

1. Użytkownik przesyła obraz JPG/JPEG/PNG.
2. Aplikacja sprawdza rozszerzenie, rzeczywisty format, rozmiar i liczbę pikseli,
   a następnie zapisuje obraz lokalnie pod bezpieczną, unikalną nazwą.
3. YOLO zwraca detekcje wybranych klas drogowych.
4. Moduł wskaźnika liczy wybrane cechy techniczne obrazu.
5. Jedna reguła wybiera główną detekcję i kategorię priorytetu komunikatu
   (wyłącznie na podstawie detekcji; wskaźnik cech nie punktuje).
6. Generator wykorzystuje główną detekcję, priorytet i ręcznie wybrany tryb do
   utworzenia opisowego komunikatu. Tryb z objaśnieniem dodaje liczbę wykryć,
   wynik ufności i pozostałe klasy.
7. Pipeline zapisuje oznaczony obraz i ścisły raport JSON z sumami SHA-256 oraz
   manifestem środowiska.
8. Interfejs udostępnia oba artefakty do pobrania i umożliwia usunięcie plików
   bieżącej analizy.

Wskaźnik opisuje wybrane cechy techniczne pliku. Nie mierzy czytelności sceny
dla człowieka, widzialności drogowej ani bezpieczeństwa jazdy.
