# Ograniczenia

- Prototyp analizuje wyłącznie pojedyncze obrazy JPG, JPEG i PNG przesłane przez użytkownika.
- System nie działa w czasie rzeczywistym.
- System nie steruje pojazdem.
- System nie przewiduje kolizji.
- System nie mierzy odległości.
- System nie diagnozuje użytkownika.
- System nie jest certyfikowanym ADAS.
- System nie potwierdza poprawy bezpieczeństwa jazdy.

Heurystyczny wskaźnik opisuje wybrane cechy techniczne pliku. Nie mierzy
czytelności sceny dla człowieka, widzialności drogowej ani bezpieczeństwa jazdy.

Priorytet jest kategorią logiki doboru komunikatu. Nie jest oceną ryzyka kolizji
ani zaleceniem manewru. Od wersji 0.3.5 heurystyczny wskaźnik cech obrazu nie
wchodzi do punktacji priorytetu. Od 0.3.6 wskaźnik jest zrekalibrowany pod test
7.4.5; to nie jest ocena czytelności dla człowieka.

## Ograniczenia walidacji

- Testy komunikatów weryfikują zgodność implementacji z regułami zdefiniowanymi
  przez autora; nie mierzą odbioru komunikatów przez kierowców.
- Nie przeprowadzono badania z użytkownikami ani kalibracji eksperckiej HMI.
- Wagi priorytetu oraz wagi i progi wskaźnika cech obrazu ustalono ręcznie.
  Analiza wrażliwości pokazuje wpływ tych stałych, ale nie potwierdza ich
  trafności psychologicznej ani bezpieczeństwa.
- Ewaluację detektora wykonano na 2000 parach wybranych przez autora z całego
  BDD100K (archiwum zamknięte, pary uporządkowane alfabetycznie). Wyniki opisują
  ten zestaw, nie cały zbiór i nie oficjalny split val.
