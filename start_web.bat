@echo off
title Upscale Pro 4K - Web Server
echo ========================================================
echo        Starting Upscale Pro 4K Local Web Studio
echo ========================================================
echo Open your browser at: http://127.0.0.1:8000
echo.
py -3.11 web_server.py --host 127.0.0.1 --port 8000
pause
