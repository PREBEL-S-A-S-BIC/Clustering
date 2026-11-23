# -*- coding: utf-8 -*-
"""
_preparar_entorno.py - Prepara el entorno antes de abrir la aplicacion.

Lo ejecutan los archivos .bat justo antes de arrancar Streamlit.

Unica tarea: crear el archivo de credenciales de Streamlit si no existe.
En su primera ejecucion, Streamlit pide un correo electronico por consola y se
queda esperando la respuesta; la aplicacion nunca llegaria a abrir. Crear el
archivo vacio evita esa pregunta para siempre.

Vive en un archivo propio y no como `python -c "..."` dentro del .bat porque las
comillas anidadas dentro de un .bat no sobreviven al interprete de cmd: la linea
fallaba en silencio y la app quedaba colgada en el prompt del correo.
"""

from pathlib import Path
import argparse
import sys
import urllib.request
import webbrowser


def abrir_si_activa():
    """Reutiliza el servidor local que ya está atendiendo la aplicación."""
    try:
        with urllib.request.urlopen(
            "http://localhost:8502/_stcore/health", timeout=2
        ) as respuesta:
            if respuesta.read().strip() != b"ok":
                return False
    except OSError:
        return False
    webbrowser.open("http://localhost:8502")
    return True


def preparar_credenciales_streamlit() -> bool:
    """Crea ~/.streamlit/credentials.toml si falta. Devuelve True si lo creo."""
    ruta = Path.home() / ".streamlit" / "credentials.toml"
    if ruta.exists():
        return False

    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text('[general]\nemail = ""\n', encoding="utf-8")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--abrir-si-activa", action="store_true")
    args = parser.parse_args()
    if args.abrir_si_activa:
        sys.exit(0 if abrir_si_activa() else 1)
    try:
        if preparar_credenciales_streamlit():
            print("Configuracion inicial de Streamlit creada.")
    except OSError as e:
        # No es critico: si falla, Streamlit solo preguntara el correo una vez.
        print(f"Aviso: no se pudo crear la configuracion de Streamlit ({e})")
