@echo off
REM JALEBI installer for Windows: runs the interactive installer with the Python on PATH.
REM   install.bat            interactive
REM   install.bat --yes      accept all defaults
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%~dp0install.py" %*
) else (
  python "%~dp0install.py" %*
)
