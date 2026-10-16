@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo 未找到 python 命令。请先安装 Python 3.10 以上版本，安装时勾选 "Add Python to PATH"。
    pause
    exit /b 1
)
python -m fcs_sim.webgui
pause
