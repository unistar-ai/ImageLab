@echo off
setlocal
cd /d "%~dp0"

rem 定位 ComfyUI：优先环境变量 IMAGELAB_COMFY_HOME，
rem 否则在本目录的同级找 ComfyUI_windows_portable / ComfyUI。
if "%IMAGELAB_COMFY_HOME%"=="" set "IMAGELAB_COMFY_HOME=%~dp0..\ComfyUI_windows_portable"
if not exist "%IMAGELAB_COMFY_HOME%\ComfyUI\main.py" if not exist "%IMAGELAB_COMFY_HOME%\main.py" set "IMAGELAB_COMFY_HOME=%~dp0..\ComfyUI"

set "PY=%IMAGELAB_COMFY_HOME%\python_embeded\pythonw.exe"
if not exist "%PY%" set "PY=%IMAGELAB_COMFY_HOME%\python_embeded\python.exe"
if not exist "%PY%" set "PY=python"

start "" "%PY%" "%~dp0app.py" --comfy-home "%IMAGELAB_COMFY_HOME%"

exit /b 0
