@echo off
REM No "uv" and no network access needed -- avoids the corporate self-signed-cert
REM %~dp0 = folder this .cmd lives in.
"%~dp0.venv\Scripts\plot_digitizer.exe" --image "%~dp0tests\CMZ54xxB Zender diode - V_Z-I_R.png" --grid --method trace --debug-svg -v
pause
