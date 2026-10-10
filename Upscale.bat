@echo off
rem Drag one or more videos onto this file to upscale them to 4K (live-action model).
rem For anime/cartoons, run:  py -3.11 upscale.py "video.mp4" --model anime
if "%~1"=="" (
  echo Drag a video file onto Upscale.bat, or run: py -3.11 "%~dp0upscale.py" --help
  pause
  exit /b
)
:loop
if "%~1"=="" goto done
py -3.11 "%~dp0upscale.py" "%~1"
shift
goto loop
:done
pause
