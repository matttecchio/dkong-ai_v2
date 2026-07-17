@echo off
rem kill titled farm supervisor windows first, then their MAMEs
taskkill /F /FI "WINDOWTITLE eq DK FARM supervisor*" 2>nul
taskkill /F /IM mame.exe 2>nul
copy /Y C:\mame\dkstates\dk_farm.ps1 C:\mame\dk_farm.ps1
copy /Y C:\mame\dkstates\bridge.lua C:\mame\bridge.lua
powershell -ExecutionPolicy Bypass -File C:\mame\dk_farm.ps1
pause
