@echo off
rem ==========================================================================
rem Empaqueta AutoPyTune en dist\AutoPyTune (one-folder, doble-click, sin instalar
rem dependencias). El COLLECT de PyInstaller recrea dist\, por eso data/ y los
rem extras se copian DESPUES del build. Correr desde la raiz del proyecto.
rem ==========================================================================
cd /d "%~dp0"

echo === PyInstaller (one-folder, ventana) ===
.venv\Scripts\pyinstaller.exe AutoPyTune.spec --noconfirm
if errorlevel 1 goto :error

echo === Copiando data (instrumentales + karaoke) ===
xcopy /E /I /Y "data\01_instrumentales" "dist\AutoPyTune\data\01_instrumentales" >nul
if errorlevel 1 goto :error
xcopy /E /I /Y "data\07_karaoke" "dist\AutoPyTune\data\07_karaoke" >nul
if errorlevel 1 goto :error

echo === Copiando extras (lanzador + LEEME) ===
copy /Y "empaquetado\Abrir AutoPyTune.bat" "dist\AutoPyTune\" >nul
copy /Y "empaquetado\LEEME.txt" "dist\AutoPyTune\" >nul

echo.
echo LISTO. Copia la carpeta entera:  dist\AutoPyTune
goto :eof

:error
echo.
echo *** FALLO el empaquetado ***
exit /b 1
