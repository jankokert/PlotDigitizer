@echo off
REM No "uv" and no network access needed -- avoids the corporate self-signed-cert
REM %~dp0 = folder this .cmd lives in.
"%~dp0.venv\Scripts\plot_digitizer.exe" --image "%~dp0tests\USFF_1206 - with axis labels.png" --debug-files
pause
