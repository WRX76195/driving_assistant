from __future__ import annotations

import ast
from pathlib import Path


def test_production_code_has_no_withdrawn_entry_points_or_engines() -> None:
    root = Path(__file__).resolve().parents[1]
    files = [root / "app.py", *sorted((root / "src").rglob("*.py"))]
    forbidden_import_roots = {"cv2", "whisper", "pyaudio", "sounddevice"}
    forbidden_calls = {"VideoCapture", "weather_classifier"}

    for path in files:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        assert not imports.intersection(forbidden_import_roots)
        assert not calls.intersection(forbidden_calls)
        assert "rtsp://" not in text.lower()


def test_streamlit_interface_has_no_yolo_model_text_input() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "app.py").read_text(encoding="utf-8")

    assert "st.sidebar.text_input" not in text
    assert "model_name" not in text
