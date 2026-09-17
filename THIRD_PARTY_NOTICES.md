# Informacje o komponentach zewnętrznych

Prototyp korzysta z gotowego modelu YOLO11n oraz biblioteki Ultralytics. Plik
`yolo11n.pt` nie został wytrenowany w ramach tej pracy. Jego zweryfikowana suma
SHA-256 wynosi:

```text
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

Dokumentacja i warunki licencyjne Ultralytics:

- https://docs.ultralytics.com/models/yolo11/
- https://www.ultralytics.com/license

Ultralytics udostępnia oprogramowanie na warunkach AGPL-3.0 oraz w wariancie
Enterprise. Dołączenie tego pliku nie nadaje odrębnej licencji na publiczną lub
komercyjną dystrybucję całego projektu. Przed wykorzystaniem innym niż ocena
akademicka należy samodzielnie sprawdzić obowiązki licencyjne wszystkich
komponentów i wag.

## BDD100K (ewaluacja offline)

Opcjonalna ewaluacja detektora może korzystać z podzbioru BDD100K
(obrazy + etykiety detekcji). Dane i etykiety pobrane ze źródeł BDD100K
podlegają warunkom The Regents of the University of California / BAIR Open
Research Commons: dozwolone użycie edukacyjne, badawcze i non-profit;
wykorzystanie komercyjne poza członkostwem BDD/BAIR Commons wymaga osobnego
uzgodnienia z UC Berkeley OTL. Surowego zbioru nie redystrybuuje się wraz z
tym repozytorium — w projekcie pozostaje skrypt `scripts/evaluate_bdd_pairs.py`
oraz zagregowany raport metryk.

Cytowanie:

Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., &
Darrell, T. (2020). BDD100K: A Diverse Driving Dataset for Heterogeneous
Multitask Learning. CVPR.

Dokumentacja licencji: https://doc.bdd100k.com/license.html

## Syntetyczny obraz `street.jpg`

Plik `data/samples/street.jpg` nie pochodzi z zewnętrznego banku zdjęć. Został
wygenerowany 16 sierpnia 2026 r. za pomocą narzędzia generowania obrazów OpenAI
na potrzeby technicznej weryfikacji prototypu, a następnie zapisany jako JPEG
o wymiarach 800 x 533 piksele. Obraz nie przedstawia rzeczywistego zdarzenia
drogowego i nie służy do oceny jakości modelu YOLO. Szczegółowe pochodzenie,
użyty opis generacyjny oraz suma SHA-256 znajdują się w
`data/samples/PROVENANCE.md`. Warunki dotyczące wygenerowanych treści opisują
aktualne Warunki korzystania z usług OpenAI: https://openai.com/policies/terms-of-use/
