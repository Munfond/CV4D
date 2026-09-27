@echo off
title VinFast ADAS 4D-OccFusion Web Portal
color 0B
cls
echo ======================================================================
echo    VINFAST ADAS 4D-OCCFUSION LAB - LOCAL WEB DASHBOARD
echo ======================================================================
echo  Dang khoi dong may chu Web Server tai dia chi:
echo  http://localhost:8080
echo.
echo  Trinh duyet web se tu dong duoc mo sau 1 giay...
echo  (Nhan Ctrl+C de dung server khi khong con su dung)
echo ======================================================================
echo.

py web_app.py --port 8080
pause
