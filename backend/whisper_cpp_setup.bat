@echo off
echo Setting up Whisper.cpp for Windows...

REM Create directory for Whisper.cpp
if not exist "whisper_cpp" mkdir whisper_cpp
cd whisper_cpp

REM Clone Whisper.cpp repository
echo Cloning Whisper.cpp repository...
git clone https://github.com/ggerganov/whisper.cpp.git .

REM Build Whisper.cpp using CMake (Windows preferred method)
echo Building Whisper.cpp...
cmake -B build
cmake --build build --config Release

REM Download the base model
echo Downloading base.en model...
powershell -Command "& {$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -OutFile models/ggml-base.en.bin}"

REM Return to original directory
cd ..

echo Whisper.cpp setup complete!
echo Model saved at: whisper_cpp\models\ggml-base.en.bin
echo Executable at: whisper_cpp\build\Release\main.exe
echo.
echo Update WHISPER_EXECUTABLE in whisper_utils.py to point to: whisper_cpp\build\Release\main.exe 