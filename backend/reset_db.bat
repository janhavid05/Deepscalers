@echo off
echo Stopping any running Django server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq runserver*" 2>nul

echo Waiting for processes to stop...
timeout /t 2 /nobreak >nul

echo Removing old database...
if exist db.sqlite3 del /F db.sqlite3

echo Creating new database...
python manage.py migrate

echo Database reset complete!
pause 