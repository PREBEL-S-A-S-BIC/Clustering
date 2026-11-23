@echo off
REM ===========================================================================
REM  Aplicacion integrada Cosnova - ventas, clusters y lanzamientos
REM
REM  La informacion se actualiza por fuera con Actualizar_Clustering_SAP.bat.
REM ===========================================================================

chcp 65001 >nul
cd /d "%~dp0"

title Planeacion Cosnova - NO CERRAR mientras uses la aplicacion

echo.
echo  ============================================================
echo    PLANEACION COSNOVA
echo  ============================================================
echo.

REM Se prefiere el entorno virtual del proyecto (vive fuera de esta carpeta por
REM el limite de 260 caracteres de Windows). Si no existe, se usa el del sistema.
set "PYEXE=C:\venvs\cosnova\Scripts\python.exe"
if exist "%PYEXE%" (
    echo  Usando el entorno virtual del proyecto.
) else (
    set "PYEXE=%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe"
    if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe" (
        echo  Usando el Python del proyecto instalado en este equipo.
    ) else (
        set "PYEXE=python"
    )
)
echo.

"%PYEXE%" --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] No se encontro Python en este equipo.
    echo  Instalalo desde https://www.python.org/downloads/ y marca
    echo  la casilla "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

"%PYEXE%" "app\_preparar_entorno.py" --abrir-si-activa
if not errorlevel 1 (
    echo  La aplicacion esta activa. Se abrio en el navegador.
    exit /b 0
)

"%PYEXE%" -c "import streamlit, pandas, sklearn" >nul 2>&1
if errorlevel 1 (
    echo  Instalando librerias por unica vez, puede tardar unos minutos...
    "%PYEXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo  [ERROR] No se pudieron instalar las librerias.
        pause
        exit /b 1
    )
)

REM Evita el prompt de email de Streamlit en la primera ejecucion, que dejaria
REM la consola esperando una respuesta y el dashboard nunca abriria.
"%PYEXE%" "app\_preparar_entorno.py"

echo  Abriendo el dashboard en tu navegador...
echo.
echo  Si no se abre solo, entra a:  http://localhost:8502
echo.
echo  NO CIERRES esta ventana mientras lo uses.
echo.

"%PYEXE%" -m streamlit run "dashboard\app.py" --server.port 8502 --server.headless false

if errorlevel 1 (
    echo.
    echo  [ERROR] El dashboard se cerro de forma inesperada.
    echo.
    pause
)
