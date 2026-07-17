@echo off
cd /d "%~dp0"
cmake -S . -B build-win || exit /b 1
cmake --build build-win --config Release || exit /b 1
build-win\Release\xrsps-native.exe %*
