from pathlib import Path


def test_project_structure():
    assert Path("preprocessing").exists()
    assert Path("pipeline").exists()
    assert Path("main.py").exists()
