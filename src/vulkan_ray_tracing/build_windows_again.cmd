@echo off
setlocal enabledelayedexpansion

:: Exit immediately on error
set ERRLEVEL=0

:: === Set up variables ===
set BUILD_DIR=build
set EXT_PATH=external

:: Set Vulkan SDK (make sure VULKAN_SDK is set in your env)
set "VULKAN_SDK=%VK_SDK_PATH%"
set "VULKAN_SDK_INCLUDE=%VULKAN_SDK%\Include"
set "VULKAN_SDK_LIB=%VULKAN_SDK%\Lib"

echo Vulkan SDK Path: %VULKAN_SDK%
echo Include: %VULKAN_SDK_INCLUDE%
echo Lib: %VULKAN_SDK_LIB%

:: Clean and recreate build directory
cd %BUILD_DIR%

:: === Handle external dependencies ===

:: SPIR-V Headers
set SPIRV_HEADERS_BIN=%EXT_PATH%\spirv-headers
@REM xcopy /E /I /Y ..\%SPIRV_HEADERS_BIN% %SPIRV_HEADERS_BIN%
@REM mkdir %SPIRV_HEADERS_BIN%\build
@REM cd %SPIRV_HEADERS_BIN%\build
@REM cmake ..
@REM cmake --build . --config Release
@REM cd ..\..\..

:: SPIR-V Tools
set SPIRV_BIN=%EXT_PATH%\spirv-tools
@REM xcopy /E /I /Y ..\%SPIRV_BIN% %SPIRV_BIN%
@REM mkdir %SPIRV_BIN%\build
@REM cd %SPIRV_BIN%\build
@REM cmake -DSPIRV-Headers_SOURCE_DIR=%CD%\..\..\spirv-headers ..
@REM cmake --build . --config Release
@REM cd ..\..\..

:: GLSLang
set GLSLANG_BIN=%EXT_PATH%\glslang
@REM xcopy /E /I /Y ..\%GLSLANG_BIN% %GLSLANG_BIN%
@REM mkdir %GLSLANG_BIN%\build
@REM cd %GLSLANG_BIN%\build
@REM cmake ^
@REM   -DENABLE_OPT=OFF ^
@REM   -DSPIRV_TOOLS_INCLUDE_DIR=%CD%\..\..\spirv-tools\include ^
@REM   -DSPIRV_TOOLS_LIB_DIR=%CD%\..\..\spirv-tools\build ^
@REM   ..
@REM cmake --build . --config Release
@REM cd ..\..\..

:: VCPKG
set VCPKG_BIN=%EXT_PATH%\vcpkg
@REM xcopy /E /I /Y ..\%VCPKG_BIN% %VCPKG_BIN%
@REM cd %VCPKG_BIN%
@REM call bootstrap-vcpkg.bat
@REM vcpkg integrate install
@REM vcpkg install tbb
@REM cd ..\..

:: pybind11
set PYBIND11_BIN=%EXT_PATH%\pybind11
@REM xcopy /E /I /Y ..\%PYBIND11_BIN% %PYBIND11_BIN%

:: GLM
set GLM_BIN=%EXT_PATH%\glm
@REM xcopy /E /I /Y ..\%GLM_BIN% %GLM_BIN%

:: === Configure with CMake ===
echo Building main Vulkan project...
for /f "delims=" %%i in ('where python') do (
    if not defined PYTHON_EXECUTABLE set PYTHON_EXECUTABLE=%%i
)
echo python is %PYTHON_EXECUTABLE%

cmake ^
  -G "NMake Makefiles" ^
  -DENABLE_OPT=OFF ^
  -DGLM_INCLUDE_DIR=%CD%\%EXT_PATH%\glm ^
  -DCMAKE_PREFIX_PATH=%CD%\%EXT_PATH%\vcpkg\installed\x64-windows ^
  -DBASE_DIRECTORY=%CD%\%EXT_PATH% ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DVulkan_INCLUDE_DIRS=%VULKAN_SDK_INCLUDE% ^
  -DVulkan_LIBRARIES=%VULKAN_SDK_LIB% ^
  -DPython3_EXECUTABLE=%PYTHON_EXECUTABLE% ^
  ..

:: === Build the project ===
nmake install

echo Build complete. You can now run your Vulkan app.
