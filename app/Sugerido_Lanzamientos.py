"""Entrada compatible para la aplicación integrada de Planeación Cosnova."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"),
               run_name="__main__")
