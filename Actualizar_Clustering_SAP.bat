@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYMODELO=%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe"
set "PYSAP=%USERPROFILE%\AppData\Local\anaconda3\envs\Ajuste_demanda\python.exe"
set "CREDENCIALES=%~dp0..\..\Ajustes_Demanda\sap_credentials.json"
echo Actualizando ventas SAP y entrenando con el catalogo del Excel.
echo El corte predeterminado es el ultimo mes cerrado.
"%PYMODELO%" run_pipeline.py --actualizar-sap --python-sap "%PYSAP%" --credenciales "%CREDENCIALES%" --sin-figuras %*
set "RESULTADO=%ERRORLEVEL%"
if not "%RESULTADO%"=="0" (
    echo La actualizacion termino con error. Revisa el mensaje anterior.
) else (
    echo Actualizacion completada. Recarga el dashboard.
)
pause
exit /b %RESULTADO%
