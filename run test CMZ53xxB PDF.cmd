@echo off
REM Locate the plot by its title (drawn as outlines -> OCR), extract it and digitize (vector mode).
"%~dp0.venv\Scripts\plot_digitizer.exe" --pdf "%~dp0tests\CMZ53xxB.pdf" --figure "Zener Voltage CMZ5334B thru CMZ5350B" --debug-files
pause
