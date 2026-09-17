"""Przygotowanie mini zbioru testowego obrazów w katalogu data/samples."""

from __future__ import annotations

import json
import urllib.request
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

REMOTE_SAMPLES = {
    "bus.jpg": "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
}

OPTIONAL_REMOTE_SAMPLES = {
    "zidane.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
}

LOCAL_SAMPLES = {
    "street.jpg": (
        "Syntetyczna scena uliczna wygenerowana na potrzeby projektu za pomocą narzędzia generowania obrazów OpenAI (2026-08-16).",
        "Kontrola detekcji wielu klas, wyboru obiektu głównego i priorytetu.",
    ),
}

SAMPLE_METADATA = {
    "bus.jpg": (
        "Przykładowa scena drogowa z autobusem (Ultralytics).",
        "Weryfikacja CLI, detekcji pojazdów i raportu JSON.",
    ),
    "street.jpg": LOCAL_SAMPLES["street.jpg"],
    "dark.png": (
        "Syntetyczny ciemny obraz o niskiej jasności.",
        "Test heurystycznego wskaźnika cech obrazu — oczekiwany niski wynik.",
    ),
    "bright.png": (
        "Syntetyczny jasny obraz o wysokiej jasności.",
        "Test heurystycznego wskaźnika cech obrazu — skrajna jasność.",
    ),
    "blur.png": (
        "Syntetyczny rozmyty obraz sceny.",
        "Test heurystycznego wskaźnika cech obrazu — niska ostrość.",
    ),
    "empty_scene.png": (
        "Jednolity obraz bez obiektów drogowych.",
        "Test komunikatu i priorytetu przy braku detekcji YOLO.",
    ),
}

OPTIONAL_SAMPLE_METADATA = {
    "zidane.jpg": (
        "Obraz demonstracyjny Ultralytics (poza mini zbiorem z punktu 7.4).",
        "Opcjonalna kontrola detekcji; nie wchodzi do przebiegów §7.4.",
    ),
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_if_missing(filename: str, url: str) -> None:
    destination = SAMPLES_DIR / filename
    if destination.is_file():
        print(f"Pominięto pobieranie (plik istnieje): {filename}")
        return
    print(f"Pobieranie: {filename}")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def generate_synthetics() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    dark_path = SAMPLES_DIR / "dark.png"
    if not dark_path.is_file():
        Image.new("RGB", (640, 480), (15, 18, 22)).save(dark_path)

    bright_path = SAMPLES_DIR / "bright.png"
    if not bright_path.is_file():
        Image.new("RGB", (640, 480), (245, 248, 252)).save(bright_path)

    blur_path = SAMPLES_DIR / "blur.png"
    if not blur_path.is_file():
        base = Image.new("RGB", (640, 480), (120, 130, 140))
        draw = ImageDraw.Draw(base)
        draw.rectangle((100, 200, 540, 400), fill=(80, 90, 100))
        draw.rectangle((250, 150, 400, 350), fill=(200, 50, 50))
        base.filter(ImageFilter.GaussianBlur(radius=8)).save(blur_path)

    empty_path = SAMPLES_DIR / "empty_scene.png"
    if not empty_path.is_file():
        Image.new("RGB", (640, 480), (180, 185, 190)).save(empty_path)


def build_manifest() -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    entries = {**SAMPLE_METADATA, **OPTIONAL_SAMPLE_METADATA}
    ordered = list(SAMPLE_METADATA) + [
        name for name in OPTIONAL_SAMPLE_METADATA if name not in SAMPLE_METADATA
    ]
    for filename in ordered:
        description, expected_use = entries[filename]
        path = SAMPLES_DIR / filename
        if not path.is_file():
            continue
        manifest.append(
            {
                "filename": filename,
                "sha256": file_sha256(path),
                "description": description,
                "expected_use": expected_use,
            }
        )
    return manifest


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in REMOTE_SAMPLES.items():
        download_if_missing(filename, url)
    for filename, url in OPTIONAL_REMOTE_SAMPLES.items():
        download_if_missing(filename, url)
    generate_synthetics()

    missing_local = [
        name for name in LOCAL_SAMPLES if not (SAMPLES_DIR / name).is_file()
    ]
    if missing_local:
        print(
            "UWAGA: brak lokalnych plików wymaganych w mini zbiorze pracy: "
            + ", ".join(missing_local)
        )
        print("Uzupełnij je w data/samples przed pełną kontrolą §7.4.")

    manifest = build_manifest()
    manifest_path = SAMPLES_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Zapisano manifest: {manifest_path} ({len(manifest)} pozycji)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
