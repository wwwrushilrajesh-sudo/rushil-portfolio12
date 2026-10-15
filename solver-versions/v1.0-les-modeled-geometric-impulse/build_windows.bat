@echo off
setlocal
cd /d "%~dp0"
where nvcc >nul 2>nul
if errorlevel 1 (
    echo nvcc was not found. Install the CUDA Toolkit and open an x64 Native Tools Command Prompt.
    exit /b 1
)
where cl >nul 2>nul
if errorlevel 1 (
    echo MSVC cl was not found. Run this file from an x64 Native Tools Command Prompt for Visual Studio.
    exit /b 1
)
rem CUDA Toolkit must support the installed MSVC version; do not override its checks.
rem No Python headers, pybind11, CMake, or CuPy are needed.
nvcc -O2 -std=c++17 --expt-relaxed-constexpr --fmad=false -shared -Xcompiler "/MD /EHsc" backend.cu -o cylinder_solver.dll -lcufft -lcusolver -lcublas
if errorlevel 1 exit /b 1
echo Built cylinder_solver.dll. You can now run main.py in Spyder.
