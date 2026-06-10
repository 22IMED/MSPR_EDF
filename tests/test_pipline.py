from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_project_structure():
    assert (PROJECT_ROOT / "preprocessing").exists()
    assert (PROJECT_ROOT / "pipeline").exists()
    assert (PROJECT_ROOT / "main.py").exists()
