# Metodyka

Testy techniczne sprawdzają poprawność implementacji. Osobna ewaluacja
detektora na oznaczonym podzbiorze BDD100K (precision / recall / F1 oraz
interpolowane AP) jest opisana w punkcie 7.7 pracy. Siatka 19 progów z
przebiegu 0.3.4: `docs/bdd_eval_metrics_thresholds_0.3.4.json`. Przebieg
0.3.6 (ta sama siatka odtworzona + AP z pełnego rankingu ufności):
`docs/bdd_eval_metrics_0.3.6.json`.

## Testowane elementy

- heurystyczny wskaźnik cech technicznych obrazu;
- reguły priorytetu komunikatu;
- trzy tryby komunikacji;
- brak zakazanych zwrotów dyrektywnych;
- struktura raportu JSON (w tym wymiary obrazu);
- geometria detekcji;
- pipeline pojedynczego obrazu;
- bezpieczny zapis przesłanego obrazu;
- unikalne katalogi wynikowe;
- obecność stałego lokalnego modelu YOLO;
- odrzucanie nieobsługiwanych wejść;
- odrzucanie plików z fałszywym rozszerzeniem, zbyt małych obrazów i podmienionych wag;
- odrzucanie obrazów o niebezpiecznie dużych deklarowanych wymiarach;
- unieważnianie wyniku po zmianie trybu komunikacji lub progu confidence;
- sprzątanie częściowych artefaktów po błędzie analizy;
- kontrola spójności tablic detektora i ograniczanie ramek do granic obrazu;
- spójność obiektu głównego w priorytecie i komunikacie;
- wpływ priorytetu na szyk tekstu;
- zgodność raportu ze ścisłym standardem JSON;
- przypadki brzegowe tablic NumPy i małych obrazów.
- obliczenia IoU, dopasowanie ramek 1:1 i obsługa wieloprogowej ewaluacji BDD.

Precision, recall i F1-score podaje się wyłącznie dla ewaluacji na oznaczonym
zbiorze (protokół BDD100K w §7.7) — nie jako wynik testów jednostkowych ani
kontroli na `data/samples`.

## Poziomy walidacji

1. Testy jednostkowe i integracyjne z kontrolowanym detektorem sprawdzają reguły
   autorskiej warstwy.
2. Test funkcjonalny z rzeczywistym `yolo11n.pt` oraz batch na mini zbiorze
   `data/samples` sprawdzają kompletność potoku.
3. Ewaluacja na 2000 parach BDD100K mierzy jakość detektora względem etykiet
   (precision / recall / F1 przy IoU 0,5 i progach wyniku ufności 0,05–0,95 oraz
   interpolowane AP z pełnego rankingu ufności) — bez redystrybucji surowego
   zbioru.
4. Ręczny test interfejsu sprawdza upload, obsługę błędu, czyszczenie stanu oraz
   dostępność obu artefaktów do pobrania.
