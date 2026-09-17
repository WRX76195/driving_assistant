"""Stałe konfiguracyjne prototypu offline.

Moduł centralizuje ścieżki katalogów, limity bezpieczeństwa wejścia,
parametry modelu YOLO oraz teksty wyjaśniające ograniczenia demonstratora.
Wartości są współdzielone przez potok, interfejs Streamlit i narzędzia CLI.
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "Warstwa komunikacji — analiza obrazu drogowego"
APP_VERSION = "0.3.6"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
TOOLING_CONFIG_DIR = PROJECT_ROOT / "models"
# Katalog na lokalne pliki konfiguracyjne narzędzi (Ultralytics/Matplotlib),
# nie na cache wag YOLO — model leży w katalogu głównym projektu.
ULTRALYTICS_CONFIG_DIR = TOOLING_CONFIG_DIR
MATPLOTLIB_CONFIG_DIR = TOOLING_CONFIG_DIR / "matplotlib"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MIN_IMAGE_SIDE = 3

DEFAULT_YOLO_MODEL = "yolo11n.pt"
YOLO_MODEL_PATH = PROJECT_ROOT / DEFAULT_YOLO_MODEL
EXPECTED_YOLO_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
SUPPORTED_OBJECT_CLASSES = {
    "person",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
}

DEFAULT_OBJECT_CONFIDENCE = 0.35

SCENE_READABILITY_EXPLANATION = (
    "Wskaźnik opisuje wybrane cechy techniczne pliku. Nie mierzy czytelności "
    "sceny dla człowieka, widzialności drogowej ani bezpieczeństwa jazdy."
)

PRIORITY_EXPLANATION = (
    "Priorytet jest kategorią logiki doboru komunikatu. Nie jest oceną ryzyka "
    "kolizji ani zaleceniem manewru. Heurystyczny wskaźnik cech obrazu nie "
    "wchodzi do punktacji priorytetu."
)

DEFAULT_LIMITATIONS = [
    "Prototyp demonstracyjny, nie certyfikowany system ADAS.",
    "Prototyp analizuje wyłącznie pojedyncze obrazy JPG, JPEG i PNG przesłane przez użytkownika.",
    "System nie działa w czasie rzeczywistym.",
    "System nie steruje pojazdem, nie przewiduje kolizji i nie mierzy odległości.",
    "System nie ocenia cech ani stanu użytkownika.",
    SCENE_READABILITY_EXPLANATION,
    PRIORITY_EXPLANATION,
]
