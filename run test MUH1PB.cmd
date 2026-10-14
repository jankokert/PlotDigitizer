@echo off
REM Extract Figures 3 and 4 from the datasheet PDF and digitize them.
"%~dp0.venv\Scripts\plot_digitizer.exe" --pdf "%~dp0tests\MUH1PB,PC,PD.pdf" --figure 3 4 --debug-files
pause
